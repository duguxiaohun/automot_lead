"""事件实例状态与动作先验。模型只回答条件；执行器回执与模型答案分离。"""
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from .taxonomy import LONGITUDINAL, TEMPLATE_STATES, applicable, template_for, transitions

ANSWERS = ('YES', 'NO', 'UNKNOWN', 'INVALID')
STARTS = {'depart', 'enter', 'return', 'proceed', 'release'}
RESTRICT = {'hold', 'restrict', 'renewed_restriction', 're_yield'}
ACTION_TEXT = {'KEEP':'保持当前走廊和适当速度', 'STOP':'制动至停车或继续等待',
               'DECELERATE':'减速建立间距', 'RESUME':'恢复行进',
               'LANE_CHANGE_LEFT':'进入已确定的左侧走廊',
               'LANE_CHANGE_RIGHT':'进入已确定的右侧走廊', 'UNCOND':'该事件不再提供动作约束'}


@dataclass
class Episode:
    event: str
    instance_id: str
    branch: str = 'default'
    state: str = ''
    longitudinal: str = 'STABLE'
    direction: str = ''
    return_direction: str = ''
    return_required: bool = True
    target_corridor: str = ''
    return_corridor: str = ''
    last_frame: int = -1
    suspended: bool = False
    needs_recheck: bool = False
    uncertain: bool = False
    stalled_observations: int = 0
    stall_limit: int = 120
    history: list = field(default_factory=list)
    waiting_observations: int = 0
    unexecuted_observations: int = 0
    wait_reason: str = ""

    def __post_init__(self):
        template = template_for(self.event, self.branch)
        if not self.state:
            self.state = TEMPLATE_STATES[template][0]
        if self.state not in TEMPLATE_STATES[template] or self.longitudinal not in LONGITUDINAL:
            raise ValueError('invalid episode state')
        if self.direction not in ('','LEFT','RIGHT','FORWARD') or self.return_direction not in ('','LEFT','RIGHT'):
            raise ValueError('invalid direction')
        if template in ('bypass','lane') and self.direction == 'FORWARD':
            raise ValueError('lateral template requires LEFT/RIGHT')
        if type(self.return_required) is not bool or not self.instance_id or self.stall_limit < 1:
            raise ValueError('invalid event instance')
        if template == 'bypass' and not self.return_required and self.state == 'RETURN':
            raise ValueError('no-return branch has no RETURN state')

    @property
    def finished(self):
        # DONE closes only the event axis. Remaining longitudinal constraints stay alive.
        return self.state == 'DONE' and self.longitudinal in ('STABLE','FOLLOW')

    @property
    def current_target_corridor(self):
        returned = (self.state == 'DONE' and self.return_required
                    and template_for(self.event,self.branch) == 'bypass')
        return self.return_corridor if self.state == 'RETURN' or returned else self.target_corridor

    def refresh_uncommitted(self):
        if self.history and self.history[-1]['pending_start'] and not self.history[-1]['committed']:
            record = self.history[-1]
            for axis,value in record['rollback'].items():
                setattr(self,axis,value)
            record['expired_start'] = list(record['pending_start'])
            record['pending_start'] = []
            self.unexecuted_observations += 1
            self.wait_reason = 'permission_not_executed'
            if self.unexecuted_observations >= self.stall_limit:
                self.needs_recheck = True

    def questions(self):
        """在下一观察构题前取消未执行的旧机会，不能将过期YES作为历史事实。"""
        self.refresh_uncommitted()
        if self.suspended or self.needs_recheck or self.finished:
            return ()
        return tuple(e for e in transitions(self.event,self.branch,self.return_required)
                     if applicable(e,self.state,self.longitudinal))

    def acknowledge(self, frame_id):
        """执行端确认已开始执行；不允许仅因模型说YES就自动确认。"""
        if self.needs_recheck or self.suspended:
            raise ValueError('cannot acknowledge an unresolved instance')
        if not self.history or frame_id != self.history[-1]['frame_id']:
            raise ValueError('acknowledgement must match latest decision')
        if not self.history[-1]['pending_start'] or self.history[-1]['committed']:
            raise ValueError('no pending start')
        self.history[-1]['committed'] = True
        self.unexecuted_observations = 0
        self.wait_reason = ''

    def confirm_execution(self, receipt):
        """Causal execution/tracker receipt; may observe a maneuver that already started.

        Binds instance + latest decision + exact edge set, including after snapshot recovery.
        Delayed receipts for an expired decision are rejected; re-observe and bind the new
        decision instead. A model YES is never an execution observation.
        """
        if (receipt.get('instance_id') != self.instance_id or
                receipt.get('source') not in ('executor','causal_tracker') or
                not receipt.get('evidence_id') or receipt.get('successor_observed') is not True):
            raise ValueError('unverified execution receipt')
        decision,observed,started = (receipt.get(k) for k in
                                    ('decision_frame','observed_frame','started_frame'))
        if (any(type(v) is not int for v in (decision,observed,started)) or
                not 0 <= started <= observed or observed != self.last_frame or decision != self.last_frame):
            raise ValueError('receipt must bind current decision and causal observation')
        pending = self.history[-1]['pending_start'] if self.history else []
        if set(receipt.get('edges',[])) != set(pending) or not pending or self.history[-1]['committed']:
            raise ValueError('receipt transition mismatch or duplicate')
        self.acknowledge(decision)
        self.history[-1]['execution_receipt'] = dict(receipt)

    def advance(self, frame_id, answers, *, context_valid=True, execution_committed=False, progress_fault=False):
        """Reject malformed input without expiring an earlier valid permission."""
        candidate = deepcopy(self)
        result = candidate._advance(frame_id, answers, context_valid=context_valid,
                                    execution_committed=execution_committed, progress_fault=progress_fault)
        self.__dict__.update(candidate.__dict__)
        return result

    def _advance(self, frame_id, answers, *, context_valid=True, execution_committed=False, progress_fault=False):
        if type(frame_id) is not int or frame_id <= self.last_frame:
            raise ValueError('stale or duplicate observation')
        if context_valid is not None and type(context_valid) is not bool:
            raise ValueError('invalid context validity')
        if type(progress_fault) is not bool:
            raise ValueError('invalid external progress fault')
        if type(execution_committed) is not bool:
            raise ValueError('invalid execution acknowledgement')
        questions = {e.key:e for e in self.questions()}
        if set(answers)-questions.keys() or any(v not in ANSWERS for v in answers.values()):
            raise ValueError('answer does not match pending transitions')
        self.last_frame = frame_id
        if progress_fault:
            self.wait_reason = 'external_progress_fault'
            self.needs_recheck = True
            return self.prior()
        if context_valid is not True or 'INVALID' in answers.values():
            self.wait_reason = 'invalid_context'
            self.needs_recheck = True
            return self.prior()
        # Missing answers are uncertainty, never silently NO.
        self.uncertain = set(answers) != set(questions) or 'UNKNOWN' in answers.values()
        if self.uncertain or self.suspended or self.needs_recheck:
            if self.uncertain:
                self.wait_reason = 'uncertain_observation'
            self.stalled_observations += 1
            if self.stalled_observations >= self.stall_limit:
                self.needs_recheck = True
            return self.prior()
        accepted = [e for k,e in questions.items() if answers[k] == 'YES']
        keys = {e.key for e in accepted}
        completing = questions.get('complete')
        progress_completion = ('complete' in keys and completing is not None
                               and 'stable_progress_established' in completing.criteria)
        if keys & RESTRICT and (keys & (STARTS | {'stable'}) or progress_completion):
            self.wait_reason = 'contradictory_restriction_and_progress'
            self.needs_recheck = True
            return self.prior()
        # stationary wait also implies restriction; consume only the stronger longitudinal edge.
        if 'hold' in keys:
            accepted = [e for e in accepted if e.axis == 'event' or e.key == 'hold']
        if any(sum(e.axis == axis for e in accepted) > 1 for axis in ('event','longitudinal')):
            self.needs_recheck = True
            return self.prior()
        for e in accepted:
            if e.key in ('depart','enter','return'):
                side = self.return_direction if e.key == 'return' else self.direction
                corridor = self.return_corridor if e.key == 'return' else self.target_corridor
                if not side or not corridor:
                    self.needs_recheck = True
                    return self.prior()
        before = (self.state,self.longitudinal)
        rollback = {}
        for e in accepted:
            updates = {}
            if e.axis == 'event':
                updates['state'] = e.target
                if e.key in ('proceed','depart','enter','return') and self.longitudinal in ('APPROACH','FOLLOW','HOLD'):
                    updates['longitudinal'] = 'RECOVER'
                elif e.key == 're_yield' and self.longitudinal != 'HOLD':
                    updates['longitudinal'] = 'APPROACH'
            else:
                updates['longitudinal'] = e.target
            for axis,value in updates.items():
                if e.key in STARTS:
                    rollback.setdefault(axis,getattr(self,axis))
                else:
                    # Independent confirmed fact supersedes an implicit start-side change.
                    rollback.pop(axis,None)
                setattr(self,axis,value)
        self.stalled_observations = 0
        self.waiting_observations = 0 if accepted else self.waiting_observations + 1
        self.wait_reason = 'permission_pending_execution' if keys & STARTS and not execution_committed else '' if accepted else 'confirmed_no_transition'
        if not keys & STARTS or execution_committed:
            self.unexecuted_observations = 0
        self.history.append(dict(frame_id=frame_id,before=list(before),after=[self.state,self.longitudinal],
                                 accepted=[e.key for e in accepted],committed=execution_committed,
                                 rollback=rollback,pending_start=sorted(keys & STARTS)))
        return self.prior()

    def prior(self):
        status = 'RECHECK' if self.needs_recheck else 'SUSPENDED' if self.suspended else 'UNKNOWN' if self.uncertain else 'DONE' if self.finished else 'ACTIVE'
        lon = {'STABLE':'KEEP','APPROACH':'DECELERATE','FOLLOW':'KEEP','HOLD':'STOP','RECOVER':'RESUME'}[self.longitudinal]
        side = self.direction if self.state in ('DEPART','CROSS') else self.return_direction if self.state == 'RETURN' else ''
        lat = f'LANE_CHANGE_{side}' if side in ('LEFT','RIGHT') else 'KEEP'
        primary = 'STOP' if lon == 'STOP' else lat if lat != 'KEEP' else lon
        if status != 'ACTIVE':
            retained = lon if status != 'DONE' and lon in ('STOP','DECELERATE') else None
            primary, lon, lat = ('UNCOND' if status == 'DONE' else retained), retained, None
        return dict(status=status,action=primary,text=ACTION_TEXT.get(primary),longitudinal=lon,
                    lateral=lat,event=self.event,state=self.state,instance_id=self.instance_id,
                    event_stage_completed=self.state=='DONE',instance_complete=self.finished,wait_reason=self.wait_reason,
                    target_corridor=self.current_target_corridor)

    def to_dict(self):
        return asdict(self)


def combine_priors(episodes):
    """Pure aggregation: retain braking and name the instances needing resolution.

    Runtime persists aggregate conflicts on the named instances. Exporting priors alone
    does not mutate episodes, but exposes the same actionable recheck list.
    """
    live = [e for e in episodes if not e.finished or e.needs_recheck]
    priors = [e.prior() for e in live]
    maneuvers = [e for e in live if e.state in ('DEPART','CROSS','RETURN')]
    targets = {(e.return_direction if e.state=='RETURN' else e.direction,
                e.current_target_corridor) for e in maneuvers}
    conflicts = {e.instance_id for e in maneuvers} if len(targets)>1 else set()
    conflicts |= {e.instance_id for e in live if e.needs_recheck and e.wait_reason=='lateral_target_conflict'}
    conflicts = sorted(conflicts)
    rechecks = sorted(set(conflicts) | {e.instance_id for e in live if e.needs_recheck or e.suspended})
    details = dict(instances=priors,conflict_instances=conflicts,recheck_instances=rechecks)
    if rechecks or any(p['status'] != 'ACTIVE' for p in priors):
        braking = {p['longitudinal'] for p in priors}
        for ep in live:
            record = ep.history[-1] if ep.history else {}
            if record.get('pending_start') and not record.get('committed'):
                old_lon = record.get('rollback',{}).get('longitudinal')
                braking.add({'HOLD':'STOP','APPROACH':'DECELERATE'}.get(old_lon))
        retained = next((a for a in ('STOP','DECELERATE') if a in braking),None)
        return dict(status='RECHECK' if rechecks else 'UNKNOWN',action=retained,
                    text=ACTION_TEXT.get(retained),**details)
    # Longitudinal constraints survive the primary lateral projection.
    actions = {p['action'] for p in priors} | {p['longitudinal'] for p in priors}
    primary = next((a for a in ('STOP','DECELERATE','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','RESUME','KEEP') if a in actions),'UNCOND')
    return dict(status='ACTIVE' if priors else 'DONE',action=primary,text=ACTION_TEXT[primary],**details)
