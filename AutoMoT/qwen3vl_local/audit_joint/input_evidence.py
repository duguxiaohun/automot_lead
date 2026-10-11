"""只读盘点精确学生视图与老师历史来源；不推断有限RGB是否足够。

python -m qwen3vl_local.audit_joint.input_evidence --view VIEW --output-dir NEW
输出题级输入/证据索引及 criterion 支持计数，不产生训练目标或盲审结果。
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .io import digest, file_sha, write_json


def source_records(value):
    """递归保留STOP/实例等旧历史，不只看最近七帧来源。"""
    if isinstance(value, dict):
        if {'path', 'sha256', 'frame_id', 'kind'} <= value.keys():
            yield {k: value[k] for k in ('path', 'sha256', 'frame_id', 'kind')}
        for child in value.values():
            yield from source_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_records(child)


def evidence_envelope(question, student):
    """保留来源与输入的区别；额外来源不等于学生不可判断。"""
    anchor = student['observation']['frame_id']
    if (question['scenario'], question['route_id'], question['frame_id'], question['edge'], question['episode']) != (
            student['scenario'], student['route_id'], anchor, student['edge'], student['episode']):
        raise ValueError('teacher/student semantic identity mismatch')
    unique = {}
    for source in source_records(question):
        if type(source['frame_id']) is not int or not 0 <= source['frame_id'] <= anchor:
            raise ValueError('future or invalid teacher source')
        key = source['path']
        if key in unique and unique[key] != source:
            raise ValueError('contradictory source identities')
        unique[key] = source
    if not unique:
        raise ValueError('teacher evidence envelope is missing')
    history = student['observation']['history_frames']
    for path, sha, frame in zip(student['images'], student['image_sha256'], history, strict=True):
        if unique.get(path) != dict(path=path, sha256=sha, frame_id=frame, kind='rgb'):
            raise ValueError('student RGB is not bound to teacher causal evidence')
    sources = sorted(unique.values(), key=lambda s: (s['frame_id'], s['kind'], s['path']))
    control = list(source_records(question.get('control_evidence', {})))
    rgb_frames = sorted({s['frame_id'] for s in sources if s['kind'] == 'rgb'})
    return dict(source_count=len(sources), source_envelope_sha256=digest(sources),
        observed_until=max(s['frame_id'] for s in sources), earliest_source_frame=sources[0]['frame_id'],
        teacher_rgb_frames=rgb_frames, teacher_rgb_frames_not_shown=sorted(set(rgb_frames) - set(history)),
        sources_before_visible_start=sum(s['frame_id'] < history[0] for s in sources),
        control_source_frames=sorted({s['frame_id'] for s in control}),
        control_sources_before_visible_start=any(s['frame_id'] < history[0] for s in control),
        privileged_source_kinds=sorted({s['kind'] for s in sources} - {'rgb'}),
        sources=sources, finite_input_judgment=None, label_error=None,
        scope='source coverage only; sources may be redundant; not a proof of necessity or unobservability')


def inspect_pair(original, short):
    """同题两输入必须来自官方变换；人工参考不能被冒充老师证据。"""
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import short_row
    from qwen3vl_local.sft_new_loop_phase4.controller import Episode
    from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key
    from qwen3vl_local.sft_new_loop_phase4.route_prompts import prompt, SYSTEM
    if short != short_row(original):
        raise ValueError('short input differs from bound official transform')
    proof = original.get('teacher_provenance')
    if original['label_basis'] == 'weak_rule_teacher' and not proof:
        raise ValueError('weak row is missing teacher provenance')
    output = []
    for mode, row in ((2, short), (4, original)):
        rendered = prompt(Episode(**row['episode']), row['edge'], row['observation'])
        if digest(rendered) != row['prompt_sha256']:
            raise ValueError('rendered student prompt identity mismatch')
        if model_input_key(row) != row['model_input_sha256']:
            raise ValueError('student model input identity mismatch')
        question = proof['question'] if proof else None
        envelope = evidence_envelope(question, row) if question else None
        output.append(dict(id=row['id'], scenario=row['scenario'], route_id=row['route_id'],
            physical_group=row['physical_group'], split=row['split'], rgb_mode=mode,
            event=row['episode']['event'], edge=row['edge'], branch=row['episode'].get('branch', 'default'),
            slice=row['slice'], frame_id=row['observation']['frame_id'],
            source_row_sha256=digest(original), student_row_sha256=digest(row),
            original_label_basis=original['label_basis'], student_label_basis=row['label_basis'],
            reference_target=row['target'], teacher_rule_target=question['rule_target'] if question else None,
            question_id=question['question_id'] if question else None,
            input=dict(history_frames=row['observation']['history_frames'], images=row['images'],
                       file_sha256=row['image_sha256'], pixel_sha256=row['image_rgb_sha256'],
                       system=SYSTEM, prompt=rendered, prompt_sha256=row['prompt_sha256'],
                       model_input_sha256=row['model_input_sha256']),
            facts=question['facts'] if question else None, evidence=envelope,
            criterion_provenance_granularity='question_envelope_only' if question else 'not_available',
            actual_input_newly_reviewed=False, independent_approval=False,
            observability_status='not_adjudicated', tokenizer_processor_verified=False))
    return output


def summarize(records):
    """分报题数/物理组数；不把多criterion或两输入累加成独立样本。"""
    totals = Counter()
    cells = defaultdict(lambda: dict(counts=Counter(), groups=set()))
    for row in records:
        prefix = f"{row['split']}/rgb{row['rgb_mode']}"
        totals[prefix + '/rows'] += 1
        totals[prefix + '/source/' + row['original_label_basis']] += 1
        envelope = row['evidence']
        if envelope is None:
            totals[prefix + '/teacher_provenance_not_available'] += 1
        else:
            totals[prefix + '/teacher_provenance_available'] += 1
            totals[prefix + '/history_before_input'] += int(envelope['sources_before_visible_start'] > 0)
            totals[prefix + '/control_history_before_input'] += int(envelope['control_sources_before_visible_start'])
            totals[prefix + '/extra_teacher_rgb_frames'] += int(bool(envelope['teacher_rgb_frames_not_shown']))
        for criterion, value in (row['facts'] or {}).items():
            key = '/'.join((prefix, row['event'], row['edge'], row['branch'], row['slice'], criterion))
            cell = cells[key]
            cell['counts']['rows'] += 1
            cell['counts'][str(value)] += 1
            cell['groups'].add(row['physical_group'])
    return dict(counts=dict(totals), criterion_cells={key: dict(counts=dict(cell['counts']),
        physical_groups=len(cell['groups'])) for key, cell in sorted(cells.items())},
        unknown_observability='all rows remain not_adjudicated; no UNKNOWN rate or accuracy inferred',
        criterion_source_assignment='question-level envelope is not per-criterion dependency tracing')


def run(view, output):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract
    view, output = Path(view).absolute(), Path(output).absolute()
    descriptor_path = view / 'student_view.json'
    descriptor = json.loads(descriptor_path.read_text())
    parent = Path(descriptor['parent']['logical_path'])
    if output.exists():
        raise FileExistsError('use a fresh evidence inventory directory')
    if any(output.resolve().is_relative_to(p.resolve()) for p in (view, parent)):
        raise ValueError('output must be outside view and parent dataset')
    source_sha = file_sha(descriptor_path)
    source_target = str(descriptor_path.resolve())
    source_contract = contract()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'progress.json', dict(status='in_progress'))
    try:
        original, manifest = data.load_dataset(view, rgb_mode=4)
        # 以相同正式选择集逐行调用官方 short_row，避免重建或另订筛选。
        with (output / 'input_evidence.jsonl').open('x') as stream:
            def records():
                for split, items in original.items():
                    for row in items:
                        if row['split'] != split:
                            raise ValueError('row split mismatch')
                        for record in inspect_pair(row, data.short_row(row)):
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                            yield record
            result = summarize(records())
        parent_manifest = json.loads((parent / 'manifest.json').read_text())
        if (file_sha(descriptor_path) != source_sha or str(descriptor_path.resolve()) != source_target
                or str(parent.resolve()) != descriptor['parent']['resolved_path']
                or data.parent_binding(parent, parent_manifest) != descriptor['parent']['files']
                or data.exposure_inventory()[1] != descriptor['exposure_files'] or contract() != source_contract):
            raise ValueError('view/parent/exposure/source changed during inventory')
        result.update(status='complete', view=str(view), view_sha256=source_sha,
            view_resolved_path=source_target, input_contract=source_contract,
            implementation_sha256=file_sha(__file__), counts_by_split=manifest['student_view']['counts'],
            production_changed=False, external_rgb_reverified=False, independent_review=False,
            g1_status='in_progress', g2_status='not_run')
        write_json(output / 'summary.json', result)
        write_json(output / 'receipt.json', dict(files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
            for p in output.iterdir() if p.name != 'progress.json'}))
        status = dict(status='complete')
    except BaseException as error:
        status = dict(status='failed', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        write_json(output / 'progress.tmp', status)
        (output / 'progress.tmp').replace(output / 'progress.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--view', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = run(args.view, args.output_dir)
    print(json.dumps(dict(status=result['status'], counts=result['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
