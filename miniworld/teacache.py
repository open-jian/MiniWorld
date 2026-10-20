"""Opt-in, rollout-local TeaCache trunk-residual reuse for MiniWorld.

The first-block input proxy and residual mechanism follow the input-aware
Hunyuan implementation of TeaCache (ali-vilab/TeaCache, commit 7c10efc).
MiniWorld supplies its own fitted coefficients, explicit CFG branch and actual
window/history identity. Prefill and KV commits must set ``commit=True`` (or
bypass the runtime and invalidate its state). No model weights are changed.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Hashable, Optional

import torch


SOURCE_COMMIT = '7c10efc4702c6b619f47805f7abe4a7a08085aa0'

# Speed-oriented default; keep each dataset's own calibrated coefficients.
# Measured at threshold 0.4: 8 videos per dataset, 253 RGB frames, 240x320,
# all 252 generated frames evaluated, one retained run per case/setting.
# Mean GT metric changes (TeaCache - Original):
#   DROID: PSNR -0.1987 dB, SSIM -0.00307, LPIPS +0.00206.
#   RE10K: PSNR -0.1532 dB, SSIM +0.00583, LPIPS +0.02441.
# RE10K's mean SSIM improved; GT averages can still hide visible output drift.
# LPIPS against Original output (mean / worst case):
#   DROID 0.01523 / 0.02674; RE10K 0.12797 / 0.28096.
# If the quality loss or visible drift is too large, lower
# --teacache_threshold (e.g. 0.2, 0.1 or 0.05), especially on RE10K.
# A smaller threshold refreshes more often, trading reuse/speed for fidelity.
DEFAULT_TEACACHE_THRESHOLD = 0.4


@dataclass(frozen=True)
class TeaCacheContext:
    runtime: object
    branch: str
    key: tuple
    force_full: bool = False
    trace_metadata: Optional[dict] = None


@dataclass(frozen=True)
class TeaCacheConfig:
    enabled: bool = False
    threshold: float = DEFAULT_TEACACHE_THRESHOLD
    # Descending powers, as in numpy.polyval; never import another model's fit.
    coefficients: tuple = ()
    calibration: bool = False
    trace: bool = False
    max_reuse_steps: int = 0
    calibration_samples: int = 65536
    max_trace_records: int = 10000

    def __post_init__(self):
        if any(type(value) is not bool for value in
               (self.enabled, self.calibration, self.trace)):
            raise ValueError('TeaCache enabled, calibration and trace must be booleans')
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('TeaCache threshold must be finite and nonnegative')
        coefficients = tuple(float(value) for value in self.coefficients)
        if any(not math.isfinite(value) for value in coefficients):
            raise ValueError('TeaCache coefficients must be finite')
        object.__setattr__(self, 'coefficients', coefficients)
        if self.enabled and not self.calibration and not coefficients:
            raise ValueError('Enabled TeaCache requires MiniWorld-calibrated coefficients')
        for name, minimum in (('max_reuse_steps', 0), ('calibration_samples', 1),
                              ('max_trace_records', 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'TeaCache {name} must be an integer >= {minimum}')


@dataclass
class _BranchState:
    context: Hashable
    signature: tuple
    previous_proxy: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    accumulated: float = 0.
    reuse_streak: int = 0
    previous_output_sample: Optional[torch.Tensor] = None
    previous_residual_sample: Optional[torch.Tensor] = None
    proxy_relative_l1: Optional[float] = None
    forward_index: int = 0
    last_full_call_index: Optional[int] = None
    last_full_outer_step: Optional[int] = None


@dataclass(frozen=True)
class TeaCacheDecision:
    should_compute: bool
    reason: str
    branch: str
    _owner: object = field(repr=False)
    _state: Optional[_BranchState] = field(default=None, repr=False)
    _version: int = field(default=0, repr=False)
    _bypass: bool = field(default=False, repr=False)
    _trace_record: Optional[dict] = field(default=None, repr=False)

    def reuse(self, trunk_input):
        """Add the most recently computed residual to this call's fresh input."""
        if self.should_compute or self._state is None or self._state.residual is None:
            raise RuntimeError('This TeaCache decision requires a full forward')
        self._owner._check_decision(self)
        residual = self._state.residual
        if _tensor_signature(trunk_input) != _tensor_signature(residual):
            raise ValueError('TeaCache residual and current trunk input do not match')
        return trunk_input + residual


def _tensor_signature(value):
    return (tuple(value.shape), value.dtype, value.device)


def _relative_l1(current, previous):
    # Do reductions in FP32 even when the denoiser runs in BF16. This is the
    # scalar synchronization intrinsic to the Python-controlled TeaCache gate.
    current, previous = current.detach().float(), previous.detach().float()
    return float(((current - previous).abs().mean() /
                  previous.abs().mean().clamp_min(1e-12)).item())


def _sample(value, count):
    # A bounded, deterministic probe of the complete flattened tensor. At most
    # the previous output/residual probes survive, never an on-device trace.
    flat = value.detach().reshape(-1)
    stride = max(1, math.ceil(flat.numel() / count))
    return flat[::stride].float().clone()


class TeaCacheRuntime:
    """Stateful gate owned by one generation, with separate state per CFG branch.

    Integration contract::

        decision = runtime.evaluate(branch, (win_sf, win_ef, history_epoch),
                                    proxy, trunk_input, force_full=last_step)
        if decision.should_compute:
            trunk_output = all_blocks(trunk_input)
            runtime.store(decision, trunk_input, trunk_output)
        else:
            trunk_output = decision.reuse(trunk_input)
        prediction = current_output_head(trunk_output)

    Calibration runs the exact full branch and samples adjacent true trunk
    outputs for fitting. ``store`` never replaces the caller's full output with
    a subtract-and-add reconstruction, preserving full-path numerical parity.
    """
    def __init__(self, config, branches=('cond', 'uncond')):
        self.config = config
        self.branches = tuple(branches)
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError('TeaCache branches must be nonempty and unique')
        self.reset()

    def reset(self):
        """Discard all tensors and metrics at the start of a new rollout."""
        self._states = {}
        self._versions = {branch: 0 for branch in self.branches}
        self._counts = Counter()
        self._branch_counts = {branch: Counter() for branch in self.branches}
        self._reasons = Counter()
        self.records = []
        self.decision_records = []

    def invalidate(self, branch=None):
        """Invalidate residual/proxy history after explicit history mutation."""
        branches = self.branches if branch is None else (branch,)
        for name in branches:
            if name not in self._versions:
                raise ValueError(f'Unknown TeaCache branch: {name}')
            self._versions[name] += 1
            if self._states.pop(name, None) is not None:
                self._counts['invalidations'] += 1

    def _record_decision(self, branch, reason, full, *, bypass=False):
        key = 'full_forwards' if full else 'reused_forwards'
        self._counts[key] += 1
        self._branch_counts[branch][key] += 1
        self._reasons[reason] += 1
        if bypass:
            self._counts['commit_forwards'] += 1
        else:
            self._counts['denoising_forwards'] += 1

    def _trace_decision(self, branch, context, reason, full, metadata, *,
                        state=None, commit=False, delta=None, prediction=None,
                        accumulated_before=0., accumulated_after_gate=0.):
        """Record existing gate scalars only; never inspect/reduce a tensor."""
        if not self.config.trace:
            return None
        call_index = self._counts['full_forwards'] + self._counts['reused_forwards'] - 1
        branch_counts = self._branch_counts[branch]
        last_full = None if state is None else state.last_full_call_index
        last_step = None if state is None else state.last_full_outer_step
        safe_scalar = lambda value: (value if value is None or math.isfinite(value) else None)
        row = dict(metadata or {})
        row.update(call_index=call_index,
            branch_call_index=branch_counts['full_forwards'] + branch_counts['reused_forwards'] - 1,
            branch=branch, outer_step=row.get('outer_step'),
            phase=row.get('phase', 'kv_commit' if commit else 'denoise'),
            window_start=context[0], window_end=context[1], history_epoch=context[2],
            decision='full' if full else 'reuse', reason=reason,
            last_full_call_index=last_full,
            reference_call_index=None if full else last_full,
            reference_outer_step=None if full else last_step,
            proxy_relative_l1=safe_scalar(delta),
            predicted_relative_change=safe_scalar(prediction),
            predicted_change=None if prediction is None or not math.isfinite(prediction)
                             else max(0., prediction),
            accumulated_before=accumulated_before,
            accumulated_after_gate=accumulated_after_gate,
            accumulated_after=0. if state is None else state.accumulated,
            reuse_streak=0 if state is None else state.reuse_streak)
        if len(self.decision_records) < self.config.max_trace_records:
            self.decision_records.append(row)
        else:
            self._counts['decision_records_dropped'] += 1
        return row

    def evaluate(self, branch, context, proxy, trunk_input, *, force_full=False,
                 commit=False, trace_metadata=None):
        """Decide before executing blocks; ``context`` must identify actual frames.

        The caller must not substitute cache length/position offset for global
        window/history identity: those values saturate while the frames change.
        ``commit`` also covers prefill; no proxy is inspected in that path.
        """
        if not self.config.enabled:
            return TeaCacheDecision(True, 'disabled', branch, self, _bypass=True)
        if branch not in self._versions:
            raise ValueError(f'Unknown TeaCache branch: {branch}')
        if commit:
            self.invalidate(branch)
            self._record_decision(branch, 'kv_commit', True, bypass=True)
            self._trace_decision(branch, context, 'kv_commit', True, trace_metadata,
                                 commit=True)
            return TeaCacheDecision(True, 'kv_commit', branch, self, _bypass=True)
        hash(context)
        signature = (_tensor_signature(proxy), _tensor_signature(trunk_input))
        state = self._states.get(branch)
        if state is None or state.context != context or state.signature != signature:
            if state is not None:
                self._counts['invalidations'] += 1
            state = self._states[branch] = _BranchState(context, signature)
        self._versions[branch] += 1
        version = self._versions[branch]
        state.forward_index += 1
        delta = (None if state.previous_proxy is None else
                 _relative_l1(proxy, state.previous_proxy))
        state.proxy_relative_l1 = delta
        # Update on every call, including reused calls, as TeaCache requires.
        state.previous_proxy = proxy.detach().clone()
        accumulated_before = state.accumulated
        prediction = None

        if self.config.calibration:
            full, reason = True, 'calibration'
            self._counts['calibration_forwards'] += 1
        elif force_full:
            full, reason = True, 'forced'
        elif state.residual is None or delta is None:
            full, reason = True, 'cold_start'
        elif not math.isfinite(delta):
            full, reason = True, 'nonfinite_gate'
        elif self.config.max_reuse_steps and state.reuse_streak >= self.config.max_reuse_steps:
            full, reason = True, 'reuse_limit'
        else:
            prediction = 0.
            for coefficient in self.config.coefficients:
                prediction = prediction * delta + coefficient
            if not math.isfinite(prediction):
                full, reason = True, 'nonfinite_gate'
            else:
                # A fitted polynomial can be negative away from its data. A
                # negative estimated error must not decrease the accumulator.
                state.accumulated += max(0., prediction)
                full = state.accumulated >= self.config.threshold
                reason = 'threshold' if full else 'reuse'
        accumulated_after_gate = state.accumulated
        if full:
            state.accumulated = 0.
            state.reuse_streak = 0
        else:
            state.reuse_streak += 1
        self._record_decision(branch, reason, full)
        row = self._trace_decision(branch, context, reason, full, trace_metadata,
            state=state, delta=delta, prediction=prediction,
            accumulated_before=accumulated_before,
            accumulated_after_gate=accumulated_after_gate)
        return TeaCacheDecision(full, reason, branch, self, state, version,
                                _trace_record=row)

    def _check_decision(self, decision):
        if (decision._owner is not self or
                self._states.get(decision.branch) is not decision._state or
                self._versions.get(decision.branch) != decision._version):
            raise RuntimeError('Stale or foreign TeaCache decision')

    def store(self, decision, trunk_input, trunk_output):
        """Save a full computation; no-op for disabled/prefill/commit calls."""
        if decision._owner is not self:
            raise RuntimeError('Foreign TeaCache decision')
        if decision._bypass:
            return
        if not decision.should_compute:
            raise RuntimeError('Cannot store a skipped TeaCache forward')
        self._check_decision(decision)
        state = decision._state
        if (_tensor_signature(trunk_input) != state.signature[1] or
                _tensor_signature(trunk_input) != _tensor_signature(trunk_output)):
            raise ValueError('TeaCache full forward changed trunk shape/dtype/device')
        residual = (trunk_output - trunk_input).detach()
        if decision._trace_record is not None:
            state.last_full_call_index = decision._trace_record['call_index']
            state.last_full_outer_step = decision._trace_record['outer_step']
        if self.config.calibration:
            output_sample = _sample(trunk_output, self.config.calibration_samples)
            residual_sample = _sample(residual, self.config.calibration_samples)
            if state.previous_output_sample is not None and state.proxy_relative_l1 is not None:
                row = dict(branch=decision.branch, context=state.context,
                    forward_index=state.forward_index,
                    proxy_relative_l1=state.proxy_relative_l1,
                    output_relative_l1=_relative_l1(output_sample, state.previous_output_sample),
                    residual_relative_l1=_relative_l1(residual_sample, state.previous_residual_sample),
                    sample_elements=output_sample.numel())
                finite = all(math.isfinite(row[key]) for key in
                    ('proxy_relative_l1', 'output_relative_l1', 'residual_relative_l1'))
                if finite and len(self.records) < self.config.max_trace_records:
                    self.records.append(row)
                elif not finite:
                    self._counts['nonfinite_records_dropped'] += 1
                else:
                    self._counts['trace_records_dropped'] += 1
            state.previous_output_sample = output_sample
            state.previous_residual_sample = residual_sample
        else:
            state.residual = residual

    def summary(self):
        counts = {name: self._counts[name] for name in
            ('full_forwards', 'reused_forwards', 'denoising_forwards',
             'commit_forwards', 'calibration_forwards', 'invalidations',
             'trace_records_dropped', 'nonfinite_records_dropped',
             'decision_records_dropped')}
        tensors = [value for state in self._states.values() for value in
            (state.previous_proxy, state.residual, state.previous_output_sample,
             state.previous_residual_sample) if value is not None]
        return dict(enabled=self.config.enabled,
            mode='calibration' if self.config.calibration else 'teacache',
            config=asdict(self.config), **counts,
            reason_counts=dict(self._reasons),
            branches={key: dict(value) for key, value in self._branch_counts.items()},
            records=list(self.records),
            decision_records=list(self.decision_records),
            trace_schema_version=1,
            cached_tensor_bytes=sum(value.numel() * value.element_size() for value in tensors),
            source_commit=SOURCE_COMMIT,
            gate='First-block normalized time/action-modulated input; full relative L1',
            calibration_target='Adjacent true trunk-output relative L1, deterministic bounded sample',
            adaptation='MiniWorld coefficients; explicit branch/window/history; nonnegative predicted change; full KV writes')


def fit_calibration(records, degree=4):
    """Fit MiniWorld's proxy/output-change relation from finite dense trace rows.

    This is an offline screening fit, not a closed-loop quality guarantee. Keep
    its metadata with the checkpoint/rollout settings in the caller's artifact.
    """
    import numpy as np

    if type(degree) is not int or degree < 0:
        raise ValueError('TeaCache polynomial degree must be a nonnegative integer')
    pairs = [(float(row['proxy_relative_l1']), float(row['output_relative_l1']))
             for row in records]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)
             and x >= 0 and y >= 0]
    if not pairs:
        raise ValueError('No finite TeaCache calibration pairs')
    x, y = np.asarray(pairs, dtype=np.float64).T
    actual_degree = min(degree, len(np.unique(x)) - 1)
    if actual_degree == 0:
        coefficients = [float(y.mean())]
    else:
        # Scale the fit domain internally before converting back to ordinary
        # descending coefficients, avoiding an unnecessarily ill-scaled solve.
        polynomial = np.polynomial.Polynomial.fit(x, y, actual_degree).convert()
        coefficients = polynomial.coef[::-1].tolist()
    predicted = np.maximum(0., np.polyval(coefficients, x))
    if not np.isfinite(coefficients).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite TeaCache calibration fit')
    return dict(coefficients=coefficients, fit_metadata=dict(
        requested_degree=degree, degree=actual_degree, samples=len(pairs),
        input_min=float(x.min()), input_max=float(x.max()),
        target='output_relative_l1',
        mae=float(np.abs(predicted - y).mean()),
        rmse=float(np.sqrt(np.mean((predicted - y) ** 2))),
        prediction_clamp='nonnegative',
        note='Dense-trace fit; validate quality and E2E speed in closed-loop inference'))
