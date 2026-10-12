"""Reproducible failure/straggler configurations; process actions stay in training."""

from dataclasses import replace


def with_worker_failure(config, step, rank=None):
    """Inject one abrupt worker exit before a zero-based step in attempt zero."""
    return replace(config, fail_step=step, fail_rank=config.fail_rank if rank is None else rank)


def with_straggler(config, seconds, start=0, stop=None, rank=None):
    """Add a controlled per-step delay to one worker in a bounded step range."""
    return replace(
        config,
        straggler_seconds=seconds,
        straggler_start=start,
        straggler_stop=stop,
        straggler_rank=config.straggler_rank if rank is None else rank,
    )


def calibration_scenarios(config, delay_seconds):
    """Successful-run scenarios used by held-out cost-model validation."""
    return [
        ("baseline", config),
        ("checkpoint-every-step", replace(config, checkpoint_every=1)),
        ("straggler", with_straggler(config, delay_seconds, start=config.steps // 3)),
    ]
