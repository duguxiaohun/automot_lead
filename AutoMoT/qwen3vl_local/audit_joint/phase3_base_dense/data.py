"""Derived references only: no RGB copies, relabeling or inferred missing targets."""
from collections import Counter, defaultdict
from dataclasses import replace, asdict
from .baseline import digest
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity, physical_group_for_item
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import FRAME_DT_SECONDS

ACTIONS = ('DECELERATE', 'STOP', 'RESUME', 'LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT', 'KEEP')
LETTERS = tuple('ABCDEF')
POLICY = 'phase3_dense_strict_frame_bank_v5'


def premise(row, *, history=True):
    """Only input-time premises already used by Phase3, never answer values."""
    from qwen3vl_local.sft_new_loop_phase3.prompts import _scene_context_block
    from qwen3vl_local.sft_new_loop_phase3.navigation_goal import render_navigation_goal
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
    r=row if history else replace(row,context_detail='')
    spec=train._make_item(r,seed=20260904,action_output_mode='choice').spec
    allowed=(*CONTEXT_BY_ID[r.context_id].action_keys,'KEEP')
    options=', '.join(LETTERS[ACTIONS.index(a)]+'='+a for a in ACTIONS if a in allowed)
    return (_scene_context_block(spec)+f'\nCurrent measured speed: {r.current_speed_mps} m/s.\n'
            +render_navigation_goal(spec.goal_xy)+'\nAllowed choices for this scene: '+options+'.')


def key(row):
    return (row.scenario, row.route_id, row.split, row.context_id, row.prompt_road_structure)


class Pool:
    def __init__(self, rows, auxiliary=(), offsets=(0,4,8)):
        self.rows = list(rows)
        self.anchor_rows=list(rows)
        self.offsets=tuple(offsets)
        self.by_key = {}
        split_groups = defaultdict(set)
        for r in rows:
            item = train._make_item(r, seed=20260904, action_output_mode='choice')
            split_groups[physical_group_for_item(item)].add(r.split)
            if train.choice_rejection_reason(item.spec) is not None:
                continue
            k = (*key(r), r.frame_id)
            if k in self.by_key:
                raise ValueError('Duplicate dense label identity')
            self.by_key[k] = r
        if any(len(s)>1 for s in split_groups.values()):
            raise ValueError('Physical route group crosses dataset splits')
        routes={(r.scenario,r.route_id,r.split) for r in rows}
        self.aux_added=0
        for r in auxiliary:
            item=train._make_item(r,seed=20260904,action_output_mode='choice')
            group=physical_group_for_item(item)
            if ((r.scenario,r.route_id,r.split) not in routes or split_groups.get(group)!={r.split}):
                raise ValueError('Auxiliary labels must belong to an existing base route/group in the same split')
            if train.choice_rejection_reason(item.spec) is not None:continue
            k=(*key(r),r.frame_id)
            if k in self.by_key:
                if asdict(r)!=asdict(self.by_key[k]):
                    raise ValueError('Auxiliary label conflicts with frozen base observation/label')
            else:
                self.by_key[k]=r;self.aux_added+=1

    def at(self, row, frame):
        return self.by_key.get((*key(row), frame))

    @staticmethod
    def action(row):
        if row is None:
            return None
        item = train._make_item(row, seed=20260904, action_output_mode='choice')
        return train.choice_action_for_answers(item.spec)

    def future(self, row, offsets=(0,4,8)):
        if not offsets or offsets[0] != 0 or list(offsets) != sorted(set(offsets)):
            raise ValueError('Offsets must start at zero and strictly increase')
        targets = [self.at(row, row.frame_id+offset) for offset in offsets]
        # No future images, speed, road/context text or true actions enter the input.
        return dict(mode='future', id=digest(case_identity(train._make_item(row, seed=20260904))),
            premises=[premise(row)],
            images=list(row.history_rgb_paths), offsets_seconds=[x*FRAME_DT_SECONDS for x in offsets],
            actions=[self.action(r) for r in targets], frames=[row.frame_id+x for x in offsets],
            anchor=row.frame_id, event=row.context_id, split=row.split,
            source_ids=[digest(case_identity(train._make_item(r,seed=20260904))) if r else None for r in targets])

    def windows(self, row, mode='stream'):
        if mode not in ('stream','windows','windows_prefix','windows_bank'):
            raise ValueError(mode)
        targets = [self.at(row, f) for f in range(row.frame_id-4, row.frame_id+1)]
        if any(r is None for r in targets):
            return None
        images = list(targets[0].history_rgb_paths)
        if len(images) != 4:
            raise ValueError('Expected original four-frame histories')
        for r in targets[1:]:
            if list(r.history_rgb_paths[:3]) != images[-3:]:
                raise ValueError('Noncontiguous RGB history; cannot form an eight-image segment')
            images.append(r.history_rgb_paths[-1])
        if len(set(images)) != 8:
            raise ValueError('Repeated initialization images in segment')
        return dict(mode=mode, id=digest(case_identity(train._make_item(row,seed=20260904))),
            premises=[premise(r,history=False) for r in targets],
            images=images, actions=[self.action(r) for r in targets],
            offsets_seconds=[0]*5, frames=[r.frame_id for r in targets], anchor=row.frame_id,
            event=row.context_id, split=row.split,
            source_ids=[digest(case_identity(train._make_item(r,seed=20260904))) for r in targets])

    def record(self, row, mode):
        return self.future(row,self.offsets) if mode=='future' else self.windows(row,mode)

    def eligible(self, mode, split):
        selected=[]; coverage=Counter(); targets=Counter()
        for row in self.anchor_rows:
            if row.split != split:
                continue
            if (*key(row),row.frame_id) not in self.by_key:continue
            record=self.record(row,mode)
            coverage['candidate/'+row.context_id]+=1
            if record is None:
                coverage['missing_segment/'+row.context_id]+=1
                continue
            selected.append(row)
            coverage['eligible/'+row.context_id]+=1
            for i,action in enumerate(record['actions']):
                targets[f'{i}/{action or "MISSING"}']+=1
        return selected,dict(coverage=coverage,targets=targets,
            label_policy='exact existing native labels in same split/route/context/road; missing masked, never KEEP',
            current_action_in_prompt=False, future_rgb_in_forecast=False)
