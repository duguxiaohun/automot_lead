"""Resolve warmup against the actual optimizer update budget, not microbatches."""
import math


def resolve_warmup(total_updates, *, steps=None, ratio=0.1):
    if type(total_updates) is not int or total_updates < 1 or not 0 <= ratio < 1:
        raise ValueError('invalid optimizer budget or warmup ratio')
    if steps is not None:
        if type(steps) is not int or not 0 <= steps < total_updates:
            raise ValueError(f'explicit warmup steps must be >=0 and < {total_updates} optimizer updates; '
                             'reduce --warmup-steps or omit it to use --warmup-ratio')
        resolved = steps
    else:
        resolved = min(total_updates-1, math.ceil(total_updates*ratio))
    return dict(total_optimizer_updates=total_updates, warmup_updates=resolved,
                warmup_source='explicit_steps' if steps is not None else 'ratio',
                requested_ratio=ratio, effective_ratio=resolved/total_updates)
