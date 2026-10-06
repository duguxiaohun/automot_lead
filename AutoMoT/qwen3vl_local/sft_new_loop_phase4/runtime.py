"""Phase1/2只负责建立/重新核验事件；Phase4实例内部自主循环。"""
from .controller import Episode, combine_priors
from copy import deepcopy
from .identity import digest
from .route_context import route_identity
from .route_prompts import prompt
from .observation import observation_contract, check_observation_contract, validate_observation


class Phase4Loop:
    def __init__(self, *, rgb_mode=4):
        self.episodes = {}
        self.observation_contract = observation_contract(rgb_mode)

    def establish(self, episode, *, verified):
        if verified is not True:
            raise ValueError('upstream event and route context must be verified')
        if episode.instance_id in self.episodes:
            raise ValueError('instance already exists; do not reset completed history')
        self.episodes[episode.instance_id] = episode

    def aggregate(self):
        """Persist cross-instance conflicts so public revalidation can resolve them."""
        prior=combine_priors(list(self.episodes.values()))
        for ident in prior['conflict_instances']:
            ep=self.episodes[ident]
            ep.refresh_uncommitted()
            ep.needs_recheck=True
            ep.wait_reason='lateral_target_conflict'
        return combine_priors(list(self.episodes.values()))

    def revalidate(self, instance_id, replacement, *, verified):
        candidate = deepcopy(self)
        candidate._revalidate(instance_id, deepcopy(replacement), verified=verified)
        self._commit(candidate)

    def _revalidate(self, instance_id, replacement, *, verified):
        self.aggregate()
        old = self.episodes[instance_id]
        if verified is not True or replacement.instance_id != instance_id:
            raise ValueError('invalid revalidation')
        if not (old.needs_recheck or old.suspended):
            raise ValueError('ordinary Phase4 progress does not need Phase1/2')
        if route_identity(old) != route_identity(replacement):
            raise ValueError('changed route target/boundary requires a new event instance')
        replacement.last_frame = old.last_frame
        replacement.history = old.history + [dict(revalidated=True,pending_start=[],committed=True)]
        self.episodes[instance_id] = replacement
        self.aggregate()

    def tick(self, observation, images, predictor, *, context_valid=None, execution_receipts=(), progress_faults=None, maneuver_clearances=()):
        """Commit the entire observation only after every prediction/receipt succeeds.

        Predictor receives staged episodes and must be read-only outside the loop;
        external inference/logging side effects cannot be rolled back here.
        """
        candidate = deepcopy(self)
        result = candidate._tick(observation, images, predictor, context_valid=context_valid,
                                 execution_receipts=execution_receipts, progress_faults=progress_faults,
                                 maneuver_clearances=maneuver_clearances)
        self._commit(candidate)
        return result

    def _commit(self, candidate):
        # Preserve public Episode references on successful commits as well as failures.
        for ident, ep in candidate.episodes.items():
            self.episodes[ident].__dict__.update(ep.__dict__)

    def _tick(self, observation, images, predictor, *, context_valid=None, execution_receipts=(), progress_faults=None, maneuver_clearances=()):
        validate_observation(self.observation_contract,observation,len(images))
        context_valid = {} if context_valid is None else context_valid
        progress_faults = {} if progress_faults is None else progress_faults
        for values, allowed in ((context_valid, (True, False, None)), (progress_faults, (True, False))):
            if (not isinstance(values, dict) or set(values) - self.episodes.keys()
                    or any(not any(v is a for a in allowed) for v in values.values())):
                raise ValueError('invalid per-instance loop context or progress fault')
        for ep in self.episodes.values():
            if not ep.finished and observation['frame_id'] <= ep.last_frame:
                raise ValueError('stale loop observation')
        receipts = {}
        safety = {}
        if not isinstance(maneuver_clearances, (list, tuple)):
            raise ValueError('maneuver clearances must be a sequence')
        for clearance in maneuver_clearances:
            if not isinstance(clearance, dict) or clearance.get('instance_id') not in self.episodes:
                raise ValueError('unknown maneuver clearance instance')
            safety.setdefault(clearance['instance_id'], []).append(clearance)
        for receipt in execution_receipts:
            ident=receipt.get('instance_id')
            if ident not in self.episodes or ident in receipts:
                raise ValueError('unknown or duplicate receipt instance')
            receipts[ident]=receipt
        for ep in self.episodes.values():
            ep.refresh_uncommitted()
        self.aggregate()
        for ident,ep in self.episodes.items():
            if ep.finished or ep.needs_recheck or ep.suspended:
                continue
            if observation['frame_id'] <= ep.last_frame:
                raise ValueError('stale loop observation')
            questions = ep.questions()
            if ep.needs_recheck:
                continue
            answers = {e.key:predictor(ep,e.key,observation,images) for e in questions}
            ep.advance(observation['frame_id'],answers,context_valid=context_valid.get(ident,True),
                       progress_fault=progress_faults.get(ident,False),
                       maneuver_clearances=safety.pop(ident, ()))
            if ident in receipts:
                ep.confirm_execution(receipts.pop(ident))
        if receipts:
            raise ValueError('receipt for inactive instance')
        if safety:
            raise ValueError('clearance for inactive instance')
        prior=self.aggregate()
        return dict(prior=prior,
                    recheck_instances=prior['recheck_instances'],
                    completed_instances=[i for i,e in self.episodes.items() if e.finished])

    def acknowledge(self, instance_id,frame_id, *, segment_id=None, route_context_id=None):
        candidate = deepcopy(self)
        candidate.aggregate()
        candidate.episodes[instance_id].acknowledge(frame_id,segment_id=segment_id,route_context_id=route_context_id)
        self._commit(candidate)

    def snapshot(self):
        return dict(version='phase4_loop_v11',observation_contract=self.observation_contract,
                    episodes={k:e.to_dict() for k,e in self.episodes.items()})

    @classmethod
    def restore(cls,snapshot):
        if snapshot.get('version') != 'phase4_loop_v11':
            raise ValueError('snapshot protocol mismatch')
        c=check_observation_contract(snapshot.get('observation_contract'))
        loop=cls(rgb_mode=c['rgb_mode'])
        for ident,fields in snapshot['episodes'].items():
            ep=Episode(**fields)
            if ep.instance_id != ident:
                raise ValueError('snapshot identity mismatch')
            loop.episodes[ident]=ep
        return loop


def predict_with_bundle(bundle,episode,edge_key,observation,images):
    import torch
    from .route_prompts import messages,parse_answer
    validate_observation(getattr(bundle,'observation_contract',None),observation,len(images))
    msgs=messages(episode,edge_key,observation,images)
    text=bundle.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=bundle.processor(text=[text],images=images,return_tensors='pt',padding=True)
    inputs={k:v.to(bundle.device) for k,v in inputs.items()}
    with torch.inference_mode():
        out=bundle.model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
    raw=bundle.tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    try:
        return parse_answer(raw)
    except ValueError:
        return 'UNKNOWN'


def handoff(phase1_answers,phase2_answers,route_events,episode_specs, *, rgb_mode=4):
    """接收现有Phase1/2严格解析后的bool字典，以及导航确认的R-E事件。

    HIGHWAY不是R-E3、RS2不是R-E2，不能由道路类别自动创建机动。
    UE7按用户给定/上游确认故障前提建立，不从某一帧灯色反推故障。
    每个参与者实例的分支、目标走廊和初始阶段由因果跟踪/导航填入spec。
    """
    if any(type(v) is not bool for v in [*phase1_answers.values(),*phase2_answers.values()]):
        raise ValueError('handoff requires strictly parsed complete bool answers')
    if episode_specs or phase2_answers or route_events:
        required_p1={'HIGHWAY','STATIC_OBSTACLE','VULNERABLE','TRAFFIC_LIGHT_ABNORMAL','RS1','RS2','RS4','RS5'}
        if not required_p1 <= phase1_answers.keys():
            raise ValueError('missing required Phase1 context judgments')
        required={'UE1','UE3','UE5','UE6','INVALID_EVENT_CONTEXT'}
        if not required <= phase2_answers.keys():
            raise ValueError('missing required Phase2 validity/event judgments')
    if phase2_answers.get('INVALID_EVENT_CONTEXT') is True:
        raise ValueError('invalid upstream context requires recheck')
    established={event for key,event in [('STATIC_OBSTACLE','U-E2'),('VULNERABLE','U-E4'),('TRAFFIC_LIGHT_ABNORMAL','U-E7')]
                 if phase1_answers.get(key) is True}
    established.update('U-E'+key[2:] for key in ('UE1','UE3','UE5','UE6') if phase2_answers.get(key) is True)
    if any(e not in ('R-E2','R-E3','R-E5') for e in route_events):
        raise ValueError('invalid navigation event')
    established.update(route_events)
    loop=Phase4Loop(rgb_mode=rgb_mode)
    for spec in episode_specs:
        ep=Episode(**spec)
        if ep.event not in established:
            raise ValueError('episode was not established upstream')
        loop.establish(ep,verified=True)
    if established!={e.event for e in loop.episodes.values()}:
        raise ValueError('established event lacks tracked instance/route branch')
    return loop
