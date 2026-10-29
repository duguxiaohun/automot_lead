"""Explicit finite budgets; changing a budget never changes labels or prompts."""

PROFILES = ('bounded', 'phase1_40k', 'phase2_10k')


def profile(phase, name='bounded'):
    if phase not in (1, 2) or name not in PROFILES:
        raise ValueError('Unknown phase/budget profile')
    if name == 'phase1_40k' and phase != 1:
        raise ValueError('phase1_40k is only available for Phase1; Phase2 stays independent')
    if name == 'phase2_10k' and phase != 2:
        raise ValueError('phase2_10k is only available for Phase2')
    extended = name == 'phase1_40k'
    longer_phase2 = name == 'phase2_10k'
    return dict(name=name, phase=phase, quota=3360 if extended else (768 if phase == 1 else 1024),
                epochs=6 if longer_phase2 else 3, world_size=4, grad_accum=1, max_steps=0, max_frame_repeat=10,
                epoch_step_limit=13440 if extended else 4096,
                total_step_limit=40320 if extended else 12288,
                epoch_hours_limit=24 if extended else 10,
                expected_updates=40320 if extended else (10752 if longer_phase2 else (9216 if phase == 1 else 5376)),
                train_timeout=432000 if extended or longer_phase2 else 172800)
