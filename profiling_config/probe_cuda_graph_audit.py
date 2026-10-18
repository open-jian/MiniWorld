"""Isolated CUDA Graph replay probe on real, fixed-shape streaming calls.

No production path is changed. Repeated denoising/commit *forwards* overwrite
only current scratch; this probe never commits or evicts between repetitions.
The actual sampler runs normally until both representative shapes are seen.
Graphs cannot be reused across history lengths, cache banks, or addresses as-is.
"""
import argparse
from pathlib import Path
import statistics
import time

import torch
import benchmark_episode_chunk_flash as bench


class FinishedProbe(Exception):
    pass


def timed(fn, reps=5):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return dict(cuda_ms=start.elapsed_time(end)/reps, wall_ms=(time.perf_counter()-wall)*1000/reps)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,_=bench.load_model()
    data=bench.load_prepared()
    forward=model.net.forward_with_cache
    report=dict(scope='Single-branch fixed-address full model forward; excludes sampler, VAE, graph construction and input-copy cost; NOT E2E',
                settings=dict(history_chunks_excluding_sink=23.5,sink_latents=1,chunk_latents=2),
                results={})
    def wrapper(*a,**kw):
        cache=kw.get('resident_cache')
        h=cache.length//300 if cache else 0
        c=a[0].shape[2]
        name=f'current{c}_'+('commit' if kw.get('return_kv') else 'denoise')
        if h==48 and c in (16,2) and name not in report['results']:
            original=lambda:forward(*a,**kw)
            for _ in range(3):
                original()
            torch.cuda.synchronize()
            reference=original()[0].clone()
            try:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_result=original()
                graph.replay()
                torch.cuda.synchronize()
                actual=graph_result[0]
                parity=dict(bitwise_equal=bool(torch.equal(actual,reference)),
                            max_abs_diff=float((actual-reference).abs().max()))
                records=[]
                for rnd in range(5):
                    for mode in (('eager','graph') if rnd%2==0 else ('graph','eager')):
                        records.append(dict(round=rnd,mode=mode,**timed(original if mode=='eager' else graph.replay)))
                med={m:statistics.median(r['cuda_ms'] for r in records if r['mode']==m) for m in ('eager','graph')}
                report['results'][name]=dict(**parity,median_cuda_ms=med,speedup=med['eager']/med['graph'],rounds=records)
                # Restore normal Python cache bookkeeping and independent
                # output ownership before the sampler receives its result.
                del actual,graph_result,graph
            except RuntimeError as exc:
                report['results'][name]=dict(error=str(exc))
            bench.dump(args.out/'summary.json',report)
            print(name,report['results'][name],flush=True)
            if len(report['results'])==2:
                raise FinishedProbe
        return forward(*a,**kw)
    model.net.forward_with_cache=wrapper
    with torch.autocast('cuda',dtype=torch.bfloat16):
        try:
            model.generate_eval_latents_streaming(data['latents'],data['cond_seq'],total_len=125,history_len=1,
                 max_cache_chunks=24,inflight_chunks=8,sink_frames=1,noise=data['noise'],use_resident_kv_cache=True)
        except FinishedProbe:
            pass
    print('DONE',flush=True)


if __name__=='__main__':
    main()
