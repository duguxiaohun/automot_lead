"""Explicit suite scope; each selected group keeps the original full budget."""
BRANCHES=('bev_only','qwen_simple')


def groups(branch='both'):
    if branch not in ('both',*BRANCHES):raise ValueError('unknown_ABC_branch')
    return tuple(f'{v}_{b}' for v in 'abc' for b in BRANCHES if branch=='both' or branch==b)


def admitted(gate):
    # Historical receipts are explicitly the original six-group suite.
    selected=gate.get('selected_groups',list(groups()))
    if not isinstance(selected,list) or not any(selected==list(groups(b)) for b in ('both',*BRANCHES)):
        raise ValueError('invalid_admitted_group_scope')
    return tuple(selected)


def resolve(branch,previous=None):
    selected=groups(branch) if branch is not None else admitted(previous or {})
    if previous is not None and selected!=admitted(previous):
        raise ValueError('existing_suite_group_scope_changed_use_new_output')
    return selected


def require_group(gate,variant,branch):
    selected=admitted(gate)
    if f'{variant}_{branch}' not in selected:raise ValueError('worker_group_outside_admitted_scope')
    return selected
