"""Read-only ATen-copy attribution from existing Kineto trace; no GPU use.

Each device activity is counted once at its launching leaf ATen operation.
_to_copy/to are enclosing wrappers, not additional device work. Reported
logical tensor bytes are estimates, never measured DRAM traffic.
"""

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path


ROOT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/kernel_audit')
SOURCE = ROOT / 'torch_profile/denoise_history48_current16.json.gz'


def main():
    trace = json.load(gzip.open(SOURCE))
    events = trace['traceEvents']
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation')]
    stacks = defaultdict(list)
    parents = {}
    for e in sorted(cpu, key=lambda e: (e['ts'], -e['dur'])):
        stack = stacks[e['tid']]
        while stack and stack[-1]['ts'] + stack[-1]['dur'] < e['ts'] + e['dur'] - .02:
            stack.pop()
        parents[id(e)] = tuple(stack)
        stack.append(e)
    external = {e['args']['External id']: e for e in cpu if 'External id' in e.get('args', {})}
    device = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    linked = defaultdict(list)
    for e in device:
        op = external.get(e.get('args', {}).get('External id'))
        if op is not None:
            linked[id(op)].append(e)
    sizes = {'float': 4, 'c10::BFloat16': 2, 'c10::Half': 2, 'double': 8, 'long int': 8,
             'long': 8, 'int': 4, 'bool': 1, 'unsigned char': 1}
    rows = []
    for op in cpu:
        if op['name'] != 'aten::copy_':
            continue
        args = op['args']
        dims, dtypes = args.get('Input Dims', []), args.get('Input type', [])
        dst, src = dims[:2]
        dst_type, src_type = dtypes[:2]
        ancestors = parents[id(op)]
        names = [e['name'] for e in ancestors]
        scopes = [n for n in names if n.startswith('audit/')]
        scope = scopes[-1] if scopes else 'unlabelled'
        under_to_copy = 'aten::_to_copy' in names
        is_cast = dst_type != src_type
        # Rank alone is insufficient: timestep Linear has a 2D [16,256]
        # activation. Match the explicit weight/bias slots of an enclosing
        # Linear/Conv call to avoid mislabelling it as a parameter cast.
        parameter_roles = []
        for ancestor in ancestors:
            if ancestor['name'] not in ('aten::linear', 'aten::conv3d'):
                continue
            ancestor_dims = ancestor.get('args', {}).get('Input Dims', [])
            for index, role in ((1, 'weight'), (2, 'bias')):
                if len(ancestor_dims) > index and ancestor_dims[index] and src == ancestor_dims[index]:
                    parameter_roles.append(role)
        parameter = (is_cast and dst_type == 'c10::BFloat16' and src_type == 'float' and parameter_roles)
        if scope == 'audit/attention_cast_cat_and_flash' and src == [1, 19200, 12, 128]:
            category = 'full_visible_K_fp32_to_bf16'
        elif scope == 'audit/attention_cast_cat_and_flash' and src == [1, 4800, 12, 128]:
            category = 'current_Q_fp32_to_bf16'
        elif parameter:
            category = 'model_weight_or_bias_fp32_to_bf16'
        elif scope == 'audit/kv_current_write':
            category = 'resident_current_KV_same_dtype_write'
        elif scope == 'audit/rope_coefficients':
            category = 'rope_coefficient_construction_copy'
        elif scope == 'audit/rope_apply':
            category = 'rope_rotation_layout_copy'
        elif scope == 'audit/rmsnorm':
            category = 'RMSNorm_activation_dtype_conversion'
        elif (scope == 'audit/attention_layout_and_other' and src == [1, 4800, 1536]
              and src_type == 'c10::BFloat16' and dst_type == 'float'):
            category = 'attention_output_roundtrip_bf16_to_fp32'
        elif (scope == 'audit/attention_output_gemm' and src == [1, 4800, 1536]
              and src_type == 'float' and dst_type == 'c10::BFloat16'):
            category = 'attention_output_roundtrip_fp32_to_bf16'
        elif is_cast:
            category = 'other_activation_dtype_conversion'
        else:
            category = 'other_same_dtype_layout_copy'
        activities = linked[id(op)]
        logical = None
        if src_type in sizes and dst_type in sizes:
            logical = math.prod(src) * sizes[src_type] + math.prod(dst) * sizes[dst_type]
        rows.append({'category': category, 'scope': scope, 'scope_chain': scopes,
                     'dst_shape': dst, 'src_shape': src, 'dst_dtype': dst_type, 'src_dtype': src_type,
                     'dst_stride': args.get('Input Strides', [[], []])[0],
                     'src_stride': args.get('Input Strides', [[], []])[1],
                     'under_to_copy_wrapper': under_to_copy,
                     'parameter_role': parameter_roles[-1] if parameter else None,
                     'device_activity_count': len(activities),
                     'device_time_us': sum(e['dur'] for e in activities),
                     'logical_read_plus_write_bytes': logical,
                     'kernel_names': [e['name'] for e in activities],
                     'external_id': args.get('External id')})
    grouped = defaultdict(lambda: {'cpu_copy_calls': 0, 'device_activities': 0,
                                  'gpu_us': 0., 'logical_read_plus_write_bytes': 0,
                                  'cpu_calls_without_correlated_device_activity': 0})
    shaped = defaultdict(lambda: {'cpu_copy_calls': 0, 'device_activities': 0, 'gpu_us': 0.})
    for r in rows:
        g = grouped[r['category']]
        g['cpu_copy_calls'] += 1
        g['device_activities'] += r['device_activity_count']
        g['gpu_us'] += r['device_time_us']
        g['logical_read_plus_write_bytes'] += r['logical_read_plus_write_bytes'] or 0
        g['cpu_calls_without_correlated_device_activity'] += r['device_activity_count'] == 0
        key = (r['category'], r['scope'], r['src_dtype'], r['dst_dtype'], str(r['src_shape']), str(r['dst_shape']))
        s = shaped[key]
        s['cpu_copy_calls'] += 1
        s['device_activities'] += r['device_activity_count']
        s['gpu_us'] += r['device_time_us']
    total_us = sum(e['dur'] for e in device)
    copy_us = sum(r['device_time_us'] for r in rows)
    categories = []
    for name, values in sorted(grouped.items(), key=lambda item: -item[1]['gpu_us']):
        categories.append({'category': name, **values, 'gpu_ms': values['gpu_us']/1000,
                           'share_of_all_device_time_percent': values['gpu_us']/total_us*100})
    shape_groups = [dict(zip(('category', 'scope', 'src_dtype', 'dst_dtype', 'src_shape', 'dst_shape'), key),
                         **values, gpu_ms=values['gpu_us']/1000)
                    for key, values in sorted(shaped.items(), key=lambda item: -item[1]['gpu_us'])]
    copy_wrappers = [e for e in cpu if e['name'] == 'aten::_to_copy']
    under_wrapper = [r for r in rows if r['under_to_copy_wrapper']]
    # Independent cross-check against root's already-exported device-activity CSV.
    with SOURCE.with_name('denoise_history48_current16_kernels.csv').open() as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_copy = [r for r in csv_rows if r['op'] == 'aten::copy_']
    csv_copy_us = sum(float(r['duration_us']) for r in csv_copy)
    assert len(csv_copy) == sum(r['device_activity_count'] for r in rows)
    assert abs(csv_copy_us - copy_us) < 1e-5
    with SOURCE.with_name('denoise_history48_current16_operators.csv').open() as handle:
        op_csv = list(csv.DictReader(handle))
    operator_checks = {}
    for name in ('aten::copy_', 'aten::_to_copy', 'aten::to', 'aten::cat', 'aten::clone', 'aten::contiguous', 'aten::index'):
        selected = [r for r in op_csv if r['op'] == name]
        operator_checks[name] = {'cpu_calls': sum(int(r['count']) for r in selected),
                                 'self_gpu_us': sum(float(r['self_gpu_us']) for r in selected)}
    assert abs(operator_checks['aten::copy_']['self_gpu_us'] - copy_us) < 1e-5
    other_movers = defaultdict(lambda: {'device_activities': 0, 'gpu_us': 0.})
    for row in csv_rows:
        if row['op'] in ('aten::cat', 'aten::index'):
            group = other_movers[row['op'], row['scope']]
            group['device_activities'] += 1
            group['gpu_us'] += float(row['duration_us'])
    report = {
        'source': str(SOURCE), 'scope': 'One real denoise forward, history48 latent including sink, current16 latent; resident KV and chunkwise FlashAttention enabled.',
        'attribution': 'Device External id -> launching leaf CPU aten::copy_; innermost audit scope assigns disjoint categories. Ancestor _to_copy/to time is not added again.',
        'all_device_activities': len(device), 'all_device_time_ms': total_us/1000,
        'aten_copy_cpu_calls': len(rows), 'aten_copy_device_activities': len(csv_copy),
        'aten_copy_gpu_ms': copy_us/1000, 'aten_copy_share_percent': copy_us/total_us*100,
        'aten_to_copy_cpu_calls': len(copy_wrappers),
        'to_copy_descendant_copy_device_activities': sum(r['device_activity_count'] for r in under_wrapper),
        'to_copy_descendant_copy_gpu_ms': sum(r['device_time_us'] for r in under_wrapper)/1000,
        'aten_to_copy_direct_device_activities': sum(len(linked[id(e)]) for e in copy_wrappers),
        'explicit_gpu_memcpy_activities': sum(e.get('cat') == 'gpu_memcpy' for e in device),
        'attention_output_roundtrip_gpu_ms': sum(r['device_time_us'] for r in rows
            if r['category'].startswith('attention_output_roundtrip')) / 1000,
        'attention_output_roundtrip_copy_calls': sum(r['category'].startswith('attention_output_roundtrip') for r in rows),
        'cross_check_copy_csv_gpu_us': csv_copy_us,
        'operator_csv_cross_checks': operator_checks,
        'other_explicit_movement_outside_copy_total': [
            {'op': key[0], 'scope': key[1], **values}
            for key, values in sorted(other_movers.items(), key=lambda item: -item[1]['gpu_us'])],
        'categories': categories, 'shape_groups': shape_groups, 'copies': rows,
        'caveats': [
            'Profiler durations are disjoint device activity sums from an instrumented single forward, not E2E wall time.',
            'Logical read+write bytes count explicit tensor operands using recorded shapes/dtypes; they are NOT measured DRAM traffic and do not model L2 reuse, transactions, or cache-line effects.',
            'copy_ kernel work includes casts and strided copies. Absence of explicit gpu_memcpy activities does not imply absence of data movement.',
            'Other explicit movement is reported separately: cat/index kernels are outside copy_ totals. Arithmetic/attention kernels also move memory; this is not a complete measured-HBM accounting.',
            'Three initial copy_ CPU calls have no correlated device event in this trace; do not assume their actual GPU cost is zero.',
            'FP32-to-BF16 full-visible-K conversion includes current K as well as committed history; historical fraction is14400/19200 tokens.',
        ],
    }
    (ROOT / 'memory_copy_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    text = [report['scope'], report['attribution'], '',
            f"All device activities: {len(device)}, summed time {total_us/1000:.6f}ms.",
            f"aten::copy_: {len(rows)} CPU calls, {len(csv_copy)} correlated device activities, {copy_us/1000:.6f}ms ({copy_us/total_us*100:.3f}%).",
            f"aten::_to_copy: {len(copy_wrappers)} CPU wrapper calls, no additional direct device activities; descendant copy cost {report['to_copy_descendant_copy_gpu_ms']:.6f}ms (already included above).", '',
            'category,cpu_calls,device_activities,gpu_ms,all_device_percent,logical_read_write_GB']
    for r in categories:
        text.append(f"{r['category']},{r['cpu_copy_calls']},{r['device_activities']},{r['gpu_ms']:.6f},{r['share_of_all_device_time_percent']:.3f},{r['logical_read_plus_write_bytes']/1e9:.6f}")
    text.extend(['', 'Shape groups (largest first):'])
    for r in shape_groups:
        text.append(f"{r['category']} | {r['scope']} | {r['src_dtype']}->{r['dst_dtype']} | {r['src_shape']} | calls={r['cpu_copy_calls']} | GPU={r['gpu_ms']:.6f}ms")
    text.extend(['', 'Other explicit movement, outside the copy_ total:'])
    for r in report['other_explicit_movement_outside_copy_total']:
        text.append(f"{r['op']} | {r['scope']} | device activities={r['device_activities']} | GPU={r['gpu_us']/1000:.6f}ms")
    text.extend(['', *report['caveats']])
    (ROOT / 'memory_copy_audit.txt').write_text('\n'.join(text) + '\n')
    print('\n'.join(text[:8 + len(categories)]))


if __name__ == '__main__':
    main()
