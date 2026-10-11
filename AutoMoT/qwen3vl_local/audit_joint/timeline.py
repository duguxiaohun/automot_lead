"""G1 只读账本：核验既有回放，保留全帧、未匹配实例和条件区间。

python -m qwen3vl_local.audit_joint.timeline --replay REPLAY --output-dir NEW_DIR
不重跑老师、不认证物理事件、学生可观察性、最终入库或真实执行许可。
"""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import stat
from types import SimpleNamespace

from .io import digest, file_sha, write_json


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def bound_file(root, name):
    """回执中的文件及中间目录不能逃逸或改指。"""
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('invalid receipt path')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('receipt symlink is not allowed')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('receipt requires regular files')
    return path


def verify_inputs(root):
    """仅核验冻结回放字节；不读取或认证当前外部 RGB。"""
    receipt_path = bound_file(root, 'receipt.json')
    receipt_sha = file_sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())['files']
    required = {'request.json', 'source.json', 'summary.json',
                'phase3_timeline.jsonl', 'phase4_questions.jsonl'}
    if not required <= receipt.keys():
        raise ValueError('replay receipt lacks required files')
    for name, proof in receipt.items():
        path = bound_file(root, name)
        if path.stat().st_size != proof['bytes'] or file_sha(path) != proof['sha256']:
            raise ValueError('replay artifact drift: ' + name)
    summary = json.loads((root / 'summary.json').read_text())
    if summary.get('status') != 'complete':
        raise ValueError('incomplete replay')
    request = json.loads((root / 'request.json').read_text())
    source = json.loads((root / 'source.json').read_text())
    if source['request_sha256'] != file_sha(root / 'request.json'):
        raise ValueError('request binding mismatch')
    requested = [(r['scenario'], r['route_id']) for r in request['routes']]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('empty or duplicate requested routes')
    summaries = {(r['scenario'], r['route_id']): r for r in summary['routes']}
    if len(summaries) != len(summary['routes']) or set(summaries) != set(requested):
        raise ValueError('route summary inventory mismatch')
    for route in request['routes']:
        frames = route['rgb_frames']
        if (not frames or any(type(f) is not int or f < 0 for f in frames)
                or frames != sorted(set(frames))):
            raise ValueError('invalid requested frame inventory')
        item = summaries[route['scenario'], route['route_id']]
        if item['raw_file'] not in receipt or item['sha256'] != receipt[item['raw_file']]['sha256']:
            raise ValueError('raw route is not bound to receipt')
    if summary['total_frames'] != sum(len(r['rgb_frames']) for r in request['routes']):
        raise ValueError('requested total frame mismatch')
    return request, source, summaries, receipt_sha


def matching_protocol(source):
    """执行映射来自已绑定源码；拒绝用新映射解释旧回放。"""
    from qwen3vl_local.sft_new_loop_phase3 import context_taxonomy, source_mapping
    from qwen3vl_local.sft_new_loop_phase4 import taxonomy, route_context
    for module in (taxonomy, route_context):
        if source['producer']['sources'].get(Path(module.__file__).name) != file_sha(module.__file__):
            raise ValueError('matching source differs from replay: ' + module.__name__)
    source_mapping.mapping_contract_hash.cache_clear()
    if source_mapping.mapping_contract_hash() != source['phase3_mapping_sha256']:
        raise ValueError('Phase3 mapping differs from replay')
    mapping = dict(taxonomy.CONTEXT_TO_EVENT)
    if mapping != {c.context_id: c.source_event for c in context_taxonomy.ACTION_CONTEXTS}:
        raise ValueError('Phase3/4 context mapping disagreement')
    return dict(version='joint_timeline_v1', frame_tolerance=0, context_to_event=mapping,
                sources={m.__name__: file_sha(m.__file__) for m in
                         (taxonomy, route_context, context_taxonomy, source_mapping)},
                phase3_mapping_sha256=source['phase3_mapping_sha256'],
                matching='same recording/frame/event only yields candidates; no participant evidence means unmatched',
                absence='no Phase4 question is UNKNOWN, including when another event has questions',
                semantics='Phase3 context gate versus Phase4 established question instances; not physical event truth',
                permission='null without deployment permission evidence; causal_tracker is offline observation',
                lag_alert_frames=None)


def presence(left, right):
    if left is None or right is None:
        return 'any_unknown'
    return {(True, True): 'both_present', (True, False): 'only_phase3',
            (False, True): 'only_phase4', (False, False): 'both_absent'}[left, right]


def index_frames(records, route):
    indexed = {}
    expected = set(route['rgb_frames'])
    for row in records:
        f = row['frame_id']
        if (type(f) is not int or f not in expected or f in indexed
                or (row['scenario'], row['route_id']) != (route['scenario'], route['route_id'])
                or row['physical_group'] != route['physical_group']):
            raise ValueError('unexpected/duplicate frame or recording identity')
        indexed[f] = row
    return indexed


def join_frames(route, p3_records, p4_records, mapping):
    """请求全帧轴的外连接；缺记录保留为 missing，不当无事件。"""
    p3 = index_frames(p3_records, route)
    p4 = index_frames(p4_records, route)
    for f in route['rgb_frames']:
        left, right = p3.get(f), p4.get(f)
        contexts = left.get('contexts', []) if left else []
        if set(contexts) - mapping.keys():
            raise ValueError('unknown Phase3 context')
        left_events = {mapping[c] for c in contexts}
        left_known = bool(left and left.get('source_annotation_present') is True
                          and left.get('mapping') is not None
                          and (left.get('annotation_repair') or {}).get('repaired_rs') != 'UNKNOWN')
        questions = right.get('questions', []) if right else []
        right_events = {q['episode']['event'] for q in questions}
        if right_events - set(mapping.values()):
            raise ValueError('unknown Phase4 event')
        # 同录制同帧还须校验当前 RGB 的内容绑定，不能跨 Rep/改过的源连接。
        current = []
        if left and right:
            current = [s for s in right.get('sources', []) if s['kind'] == 'rgb' and s['frame_id'] == f]
            if current and any(s['sha256'] != left['rgb_identity']['sha256'] for s in current):
                raise ValueError('same-frame RGB identity mismatch')
        events = []
        for event in mapping.values():
            ids = sorted({q['episode']['instance_id'] for q in questions if q['episode']['event'] == event})
            a = event in left_events if left_known else None
            b = True if ids else None
            status = ('ambiguous_match' if a and len(ids) > 1 else
                      'unmatched_participant_evidence' if a and ids else 'unmatched')
            events.append(dict(event=event, phase3_presence=a, phase4_presence=b,
                               presence_status=presence(a, b), candidate_instance_ids=ids,
                               match_status=status, confirmed_instance_id=None,
                               comparable=False, label_error=None))
        yield dict(scenario=route['scenario'], route_id=route['route_id'], physical_group=route['physical_group'],
                   split=route['split'], frame_id=f, phase3_record_present=left is not None,
                   phase4_record_present=right is not None, phase3_context_known=left_known,
                   phase3_action_evidence_present=bool(left and left.get('evidence')),
                   phase3_contexts=contexts, events=events,
                   recording_rgb_binding='same_frame_sha_verified' if current else 'not_available',
                   phase3_raw_labels=left.get('raw_labels') if left else None,
                   phase3_longitudinal_decision=(left.get('evidence') or {}).get('longitudinal_decision') if left else None,
                   phase3_native_candidate_count=len(left.get('native_eligible_candidates', [])) if left else None,
                   phase3_record_sha256=digest(left) if left else None,
                   phase4_record_sha256=digest(right) if right else None,
                   phase4_reasons=right.get('reasons', []) if right else ['missing_record'],
                   physical_onset=None, physical_completion=None, final_supervision_coverage=None)


class ConditionIntervals:
    """流式记录原始三态片段；停止问不改写为 NO，答案不等于许可。"""
    def __init__(self, route, write):
        self.route, self.write = route, write
        self.active = {}
        self.last_frame = None

    def close(self, key, reason, observed_frame):
        item = self.active.pop(key)
        item.update(end_reason=reason, end_observed_frame=observed_frame,
                    right_censored=reason != 'target_changed')
        self.write(item)

    def observe(self, frame, record, admitted):
        if self.last_frame is not None and frame != self.last_frame + 1:
            for key in list(self.active):
                self.close(key, 'observation_gap', frame)
        self.last_frame = frame
        questions = record.get('questions', []) if record else []
        ended = {x['instance_id']: x['reason'] for x in record.get('instance_ends', [])} if record else {}
        traces = {x['instance_id']: x for x in record.get('traces', [])} if record else {}
        present = set()
        for q in questions:
            ep = q['episode']; key = (ep['instance_id'], q['edge'], q['rgb_mode'])
            if key in present:
                raise ValueError('duplicate instance/edge/mode at frame')
            present.add(key)
            if q['rule_target'] not in ('YES', 'NO', 'UNKNOWN'):
                raise ValueError('invalid rule target')
            signature = digest(dict(episode=ep, slice=q['slice']))
            previous = self.active.get(key)
            if previous and (previous['context_sha256'] != signature or previous['target'] != q['rule_target']):
                reason = ('question_context_changed' if previous['context_sha256'] != signature else
                          'became_unknown' if q['rule_target'] == 'UNKNOWN' else 'target_changed')
                self.close(key, reason, frame)
            if key not in self.active:
                self.active[key] = dict(scenario=self.route['scenario'], route_id=self.route['route_id'],
                    physical_group=self.route['physical_group'], instance_id=key[0], event=ep['event'],
                    edge=key[1], rgb_mode=key[2], episode=ep, slice=q['slice'], target=q['rule_target'],
                    context_sha256=signature, start_frame=frame, end_frame=frame, frame_count=0,
                    left_boundary='first_observed_in_this_context_not_physical_onset',
                    first_question_id=q['question_id'], last_question_id=q['question_id'],
                    gate_target_counts=Counter(), first_offline_execution_observed_frame=None,
                    first_offline_state_sync_frame=None, permission_granted=None,
                    model_response_frame=None, deployment_execution_frame=None,
                    lag_alert_frames=None, label_error=None)
            item = self.active[key]
            item['end_frame'] = frame; item['frame_count'] += 1
            item['last_question_id'] = q['question_id']
            item['gate_target_counts'][admitted[q['question_id']]['admission_target']] += 1
            trace = traces.get(key[0], {})
            receipt = trace.get('execution_receipt') or {}
            if key[1] in receipt.get('edges', []) and receipt.get('successor_observed') is True:
                if item['first_offline_execution_observed_frame'] is None:
                    item['first_offline_execution_observed_frame'] = frame
            if trace.get('observed_state_sync') and item['first_offline_state_sync_frame'] is None:
                item['first_offline_state_sync_frame'] = frame
        for key in list(self.active):
            if key not in present:
                reason = ('missing_record' if record is None else
                          'instance_end:' + ended[key[0]] if key[0] in ended else
                          'question_not_observed')
                self.close(key, reason, frame)

    def finish(self):
        for key in list(self.active):
            self.close(key, 'route_end', None)


def route_groups(path, question=False):
    key = lambda r: ((r['question'] if question else r)['scenario'],
                     (r['question'] if question else r)['route_id'])
    yield from groupby(rows(path), key)


def take_route(groups, current, key):
    if current is not None and current[0] == key:
        result = list(current[1])
        return result, next(groups, None)
    return [], current


def validate_question(q, route, frame):
    """只检查原始题身份/因果范围及冻结边定义，不重算老师答案。"""
    from qwen3vl_local.sft_new_loop_phase4.route_context import episode_edge
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import applicable
    if (q['scenario'], q['route_id'], q['physical_group'], q['frame_id']) != (
            route['scenario'], route['route_id'], route['physical_group'], frame):
        raise ValueError('question recording/frame mismatch')
    if q['question_id'] != digest({k: v for k, v in q.items() if k != 'question_id'}):
        raise ValueError('question identity mismatch')
    ep = SimpleNamespace(**q['episode'])
    edge = episode_edge(ep, q['edge'])
    if not applicable(edge, ep.state, ep.longitudinal):
        raise ValueError('inapplicable question edge')
    if q['slice'] not in ('readiness', 'catchup'):
        raise ValueError('invalid question slice')
    if set(q['facts']) != set(edge.criteria):
        raise ValueError('question criterion mismatch')
    if any(v is not None and type(v) is not bool for v in q['facts'].values()):
        raise ValueError('criterion must be tri-state')
    for field in ('input_sources', 'causal_sources', 'instance_sources'):
        if ((field != 'instance_sources' and not q.get(field))
                or any(type(s.get('frame_id')) is not int or s['frame_id'] > frame for s in q[field])):
            raise ValueError('missing or future question evidence')
    if q['rgb_mode'] not in (2, 4) or q['history_frames'] != [s['frame_id'] for s in q['input_sources']]:
        raise ValueError('question input frame mismatch')
    if len(q['history_frames']) != q['rgb_mode'] or q['history_frames'][-1] != frame:
        raise ValueError('question input mode/anchor mismatch')
    if (q['history_frames'] != sorted(set(q['history_frames']))
            or any(s['kind'] != 'rgb' for s in q['input_sources'])):
        raise ValueError('question input order/kind mismatch')
    def check_nested(value):
        # STOP 等历史证据也在因果包络内，不能只验顶层的七帧 sources。
        if isinstance(value, dict):
            if {'path', 'sha256', 'frame_id'} <= value.keys():
                if type(value['frame_id']) is not int or not 0 <= value['frame_id'] <= frame:
                    raise ValueError('future or invalid nested causal evidence')
                kind = value.get('kind')
                if kind not in ('rgb', 'metas', 'bboxes'):
                    raise ValueError('unknown causal source kind')
                extension = 'jpg' if kind == 'rgb' else 'pkl'
                expected = f"{route['scenario']}/{route['route_id']}/{kind}/{value['frame_id']:04d}.{extension}"
                if value['path'] != expected:
                    raise ValueError('causal source recording/path mismatch')
            for child in value.values():
                check_nested(child)
        elif isinstance(value, list):
            for child in value:
                check_nested(child)
    check_nested(q)


def question_view(q, gate):
    """复制已有字段和引用；没有保存的学生文字/批准不从当前模板补造。"""
    return dict(question_id=q['question_id'], episode=q['episode'], edge=q['edge'],
        actor_id=q.get('instance', {}).get('actor_id'), actor_ids=q.get('instance', {}).get('actor_ids'),
        rgb_mode=q['rgb_mode'], history_frames=q['history_frames'],
        offsets=[f - q['frame_id'] for f in q['history_frames']], input_sources=q['input_sources'],
        input_sha256=q.get('input_sha256'), causal_sources_sha256=digest(q['causal_sources']),
        causal_source_until=max(s['frame_id'] for s in q['causal_sources']),
        facts=q['facts'], raw_rule_target=q['rule_target'], state_rule_target=q['state_rule_target'],
        gate_target=gate['admission_target'], gate_reason=gate['admission_reason'], slice=q['slice'],
        visible_condition_scope=q.get('visible_condition_scope'),
        final_dataset_target=None, finite_input_judgment=None, exact_student_prompt=None,
        input_scope='native replay RGB references; no student view/tokenizer/processor certification',
        criterion_provenance='whole question causal envelope; per-criterion observability not adjudicated')


def run(replay, output):
    replay, output = Path(replay).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError('use a fresh timeline directory')
    if output.is_relative_to(replay):
        raise ValueError('output must be outside the source replay')
    request, source, summaries, receipt_sha = verify_inputs(replay)
    protocol = matching_protocol(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'progress.json', dict(status='in_progress'))
    try:
        write_json(output / 'matching_protocol.json', protocol)
        p3groups = iter(route_groups(replay / 'phase3_timeline.jsonl'))
        qgroups = iter(route_groups(replay / 'phase4_questions.jsonl', question=True))
        left_group, q_group = next(p3groups, None), next(qgroups, None)
        counts = Counter(); match_counts = Counter(); ends = Counter()
        event_counts = Counter({event + '/' + status: 0 for event in protocol['context_to_event'].values()
                               for status in ('both_present', 'only_phase3', 'only_phase4', 'both_absent', 'any_unknown')})
        with (output / 'timeline.jsonl').open('x') as timeline, \
                (output / 'discrepancy_queue.jsonl').open('x') as queue, \
                (output / 'condition_intervals.jsonl').open('x') as intervals:
            def write_interval(item):
                ends[item['end_reason']] += 1
                emit(intervals, item)
            for route in request['routes']:
                key = route['scenario'], route['route_id']
                left, left_group = take_route(p3groups, left_group, key)
                flat, q_group = take_route(qgroups, q_group, key)
                admitted = {r['question']['question_id']: r for r in flat}
                if len(admitted) != len(flat):
                    raise ValueError('duplicate admitted question ID')
                raw = iter(rows(replay / summaries[key]['raw_file']))
                header = next(raw)
                if header.get('kind') != 'diagnostic_header' or header.get('route') != route:
                    raise ValueError('raw route header mismatch')
                right = list(raw); indexed = index_frames(right, route); seen = set()
                tracker = ConditionIntervals(route, write_interval)
                for frame in join_frames(route, left, right, protocol['context_to_event']):
                    f = frame['frame_id']; record = indexed.get(f)
                    frame['phase4_questions'] = []
                    frame['phase4_raw_file'] = summaries[key]['raw_file']
                    frame['phase3_raw_file'] = 'phase3_timeline.jsonl'
                    for q in record.get('questions', []) if record else []:
                        validate_question(q, route, f)
                        current = [s for s in record.get('sources', []) if s['kind'] == 'rgb' and s['frame_id'] == f]
                        if current and q['input_sources'][-1] not in current:
                            raise ValueError('question/current frame RGB binding mismatch')
                        ident = q['question_id']
                        if ident in seen or ident not in admitted or q != admitted[ident]['question']:
                            raise ValueError('raw/admission question identity mismatch')
                        if admitted[ident]['admission_target'] not in ('YES', 'NO', 'UNKNOWN'):
                            raise ValueError('invalid admission target')
                        seen.add(ident)
                        frame['phase4_questions'].append(question_view(q, admitted[ident]))
                        counts['raw_questions'] += 1
                        counts['gate_' + admitted[ident]['admission_target']] += 1
                    tracker.observe(f, record, admitted)
                    counts['requested_frames'] += 1
                    counts['both_records'] += int(frame['phase3_record_present'] and frame['phase4_record_present'])
                    counts['same_frame_rgb_bound'] += int(frame['recording_rgb_binding'] == 'same_frame_sha_verified')
                    counts['phase3_action_frames'] += int(frame['phase3_action_evidence_present'])
                    counts['phase3_context_known_frames'] += int(frame['phase3_context_known'])
                    emit(timeline, frame)
                    for event in frame['events']:
                        event_counts[event['event'] + '/' + event['presence_status']] += 1
                        if event['phase3_presence'] is True or event['phase4_presence'] is True:
                            match_counts[event['match_status']] += 1
                            emit(queue, dict(scenario=key[0], route_id=key[1], frame_id=f,
                                             physical_group=route['physical_group'], **event))
                tracker.finish()
                if seen != admitted.keys():
                    raise ValueError('admission questions missing from raw frames')
        if left_group is not None or q_group is not None:
            raise ValueError('unexpected or out-of-order route records')
        # 成功前重新核验输入及映射，禁止混入同时进行的回放/源码修改。
        if verify_inputs(replay)[3] != receipt_sha or matching_protocol(source) != protocol:
            raise ValueError('source changed during timeline audit')
        result = dict(status='complete', g1_status='in_progress', source_replay=str(replay),
            replay_receipt_sha256=receipt_sha, source_contract=source,
            matching_protocol_sha256=file_sha(output / 'matching_protocol.json'),
            implementation_sha256=file_sha(__file__), counts=dict(counts),
            event_presence_counts=dict(event_counts), match_counts=dict(match_counts),
            interval_end_counts=dict(ends), confirmed_instance_matches=0, comparable_propositions=0,
            final_supervision_coverage=None, external_rgb_reverified=False, independent_review=False,
            denominator_units=dict(frames='recording × frame', presence='recording × frame × event',
                questions='recording × frame × instance × edge × RGB mode',
                intervals='native rule target in unchanged question context; modes counted separately'),
            missing_records=counts['requested_frames'] - counts['both_records'],
            scope='selected recorded routes; gate proposals and offline conditions only; no correctness or safety verdict')
        write_json(output / 'summary.json', result)
        write_json(output / 'receipt.json', dict(files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
            for p in output.iterdir() if p.name != 'progress.json'}))
        status = dict(status='complete', g1_status='in_progress')
    except BaseException as error:
        status = dict(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        temp = output / 'progress.tmp'
        write_json(temp, status)
        temp.replace(output / 'progress.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = run(args.replay, args.output_dir)
    print(json.dumps({k: result[k] for k in ('status', 'g1_status', 'counts', 'missing_records')}))


if __name__ == '__main__':
    main()
