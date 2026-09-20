#!/usr/bin/env python3
"""从已有评测包准备逐帧人工审计；先排异常路线，未来帧仅作离线取证。

不会运行模型或修改标签。输出保存原图/原 meta 指纹、实际输入核验及逐帧表；
拼图不是人工审计结论，观察结果必须另写手工笔记。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageDraw
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, label_actions
from qwen3vl_local.sft_new_loop_phase3.primary_action import primary_answers, primary_action
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import PRIMARY_CHOICE_VERSION


def digest(path):
    """记录原始文件身份，不把压缩拼图当作模型实际输入。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_action(row, answers):
    """v15 的空动作位显示 KEEP；解析失败不得在 RGB 审计图上冒充保持。"""
    if not answers or any(value not in ("YES", "NO") for value in answers.values()):
        return "UNPARSED"
    positive = "+".join(key for key, value in answers.items() if value == "YES")
    spec = row.get("prompt_spec", {})
    empty = ("KEEP" if spec.get("action_output_mode") == "choice"
             and spec.get("primary_action_version") in (
                 PRIMARY_CHOICE_VERSION, "primary_choice_v2_explicit_domain_keep") else "NONE")
    return positive or empty


def compare_action_labels(row, labels):
    """分开检查原始证据和主要动作，避免把合法的组合投影报告成错标。"""
    valid = row['gt']['INVALID_ACTION_CONTEXT'] == 'NO'
    raw = row.get('action_answers', row['gt'])
    allowed = [k for k in row['gt'] if k not in ('INVALID_ACTION_CONTEXT', 'KEEP')]
    if labels is not None and ('KEEP' in raw or 'KEEP' in row['gt']):
        labels = {**labels, 'KEEP': valid and primary_action(labels, allowed) == 'NONE'}
    raw_mismatches = [k for k, v in raw.items() if k != 'INVALID_ACTION_CONTEXT'
                      and valid and (labels is None or labels[k] != (v == 'YES'))]
    projected = (primary_answers(labels, allowed)
                 if labels is not None and row.get('prompt_spec', {}).get('action_output_mode') == 'choice'
                 else labels)
    target_mismatches = [k for k, v in row['gt'].items() if k != 'INVALID_ACTION_CONTEXT'
                         and valid and (projected is None or projected.get(k) != (v == 'YES'))]
    return raw_mismatches, target_mismatches


def prepare(bundle, data_root, output, extra_ids=(), only_ids=None):
    """先核验全评测路线，再准备精选错例、十情境对照和显式补查案例。"""
    rows = {r['case_index']: r for f in sorted((bundle/'lora_production').glob('cases*.jsonl'))
            for line in f.read_text().splitlines() if line.strip() for r in [json.loads(line)]}
    routes = sorted({(r['scenario'], r['route_id']) for r in rows.values()})
    allowed, checks = set(), []
    print(f'discover: {len(rows)} cases, {len(routes)} routes', flush=True)
    for i, (scenario, route) in enumerate(routes, 1):
        run = data_root/scenario/route
        excluded, info = is_abnormal_lead_route(run, scenario)
        info.update(excluded=excluded, missing=not run.is_dir())
        checks.append(info)
        if not excluded and run.is_dir():
            allowed.add((scenario, route))
        if i % 20 == 0 or i == len(routes):
            print(f'route [{i}/{len(routes)}]', flush=True)
    selected = set(extra_ids)
    for f in (bundle/'lora_production_audit_samples').glob('*/*/case.json'):
        selected.add(json.loads(f.read_text())['case_index'])
    # 每个context补两个不同route的错例和一个正确对照，避免只看一个错误桶。
    for context in sorted({r['context_id'] for r in rows.values()}):
        for correct, limit in ((False, 2), (True, 1)):
            seen = set()
            for i, r in sorted(rows.items()):
                if r['context_id'] != context or r['all_ok'] != correct or r['gt']['INVALID_ACTION_CONTEXT'] != 'NO':
                    continue
                if r['route_id'] not in seen:
                    selected.add(i)
                    seen.add(r['route_id'])
                if len(seen) == limit:
                    break
    # 续审只生成新增案例，避免重新生成上一轮已有的逐帧证据。
    if only_ids is not None:
        if extra_ids:
            raise ValueError('choose only_ids or extra_ids, not both')
        selected = set(only_ids)
        unknown = selected - rows.keys()
        if unknown:
            raise ValueError(f'unknown case IDs: {sorted(unknown)}')
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    for i in sorted(selected):
        r = rows[i]
        if (r['scenario'], r['route_id']) not in allowed:
            continue
        run = data_root/r['scenario']/r['route_id']
        traj = load_route_trajectory(run)
        if traj is None:
            raise ValueError(f'missing raw meta: {run}')
        anchor = r['frame_id']
        inputs = [run/'rgb'/Path(v).name for v in r['history_rgb_paths_used']]
        if [digest(f) for f in inputs] != r['history_rgb_sha256']:
            raise ValueError(f'actual input RGB SHA256 mismatch: case {i}')
        signals = traj.signals(anchor)
        expected = r['action_evidence']['future_speeds_exact_mps']
        if signals['future_speeds'] != expected:
            raise ValueError(f'raw speed mismatch: case {i}')
        labels = label_actions(signals)
        # choice 的 gt 已投影；先核验原始纵横证据，再独立核验主要动作，不能
        # 把被优先级合法省略的速度/横向动作当作原 meta 标定错误。
        raw_answers = r.get('action_answers', r['gt'])
        label_mismatches, target_mismatches = compare_action_labels(r, labels)
        # 3秒末帧的首次越线由下一采样确认；+3.25s只用于确认，不扩张预测窗口。
        frames = [int(f.stem) for f in inputs] + list(range(anchor+1, anchor+14))
        canvas = Image.new('RGB', (2304, ((len(frames)+3)//4)*224+50), '#101010')
        draw = ImageDraw.Draw(canvas)
        draw.text((8,5), f"#{i} {r['scenario']} f{anchor} {r['context_id']} RS={r['prompt_road_structure']}", fill='white')
        draw.text((8,23), f"GT {display_action(r, r['gt'])} | PRED {display_action(r, r['parsed'])} | FIRST {len(inputs)} PANELS=INPUT, others=FUTURE EVIDENCE", fill='yellow')
        frame_rows, missing_frames = [], []
        for j, f in enumerate(frames):
            rgb, meta = run/'rgb'/f'{f:04d}.jpg', traj.metas.get(f)
            x, y = (j % 4)*576, 50+(j//4)*224
            if not rgb.exists() or meta is None:
                draw.text((x+4,y+4),f'MISSING f{f}',fill='red')
                missing_frames.append(f)
                continue
            with Image.open(rgb) as im:
                canvas.paste(im.convert('RGB').resize((576,192)),(x,y))
            speed = float(meta['speed'])
            caption = f"f{f} t={(f-anchor)*.25:+.2f}s v={speed:.3f} road/lane={meta.get('road_id')}/{meta.get('lane_id')} brake={int(bool(meta.get('brake')))}"
            draw.text((x+3,y+195),caption,fill='yellow' if j<len(inputs) else 'white')
            frame_rows.append(dict(frame=f, input=j<len(inputs), rgb=str(rgb), rgb_sha256=digest(rgb),
                meta_sha256=digest(run/'metas'/f'{f:04d}.pkl'), speed=speed,
                road_id=meta.get('road_id'), lane_id=meta.get('lane_id'), section_id=meta.get('section_id'),
                confirmation_only=f == anchor+13,
                brake=bool(meta.get('brake')), throttle=float(meta.get('throttle',0)),
                vehicle_hazard=bool(meta.get('vehicle_hazard')), light_hazard=bool(meta.get('light_hazard'))))
        sheet=output/f'case_{i:03d}.jpg'
        canvas.save(sheet,quality=92)
        evidence.append(dict(case_index=i, scenario=r['scenario'], route_id=r['route_id'],
            frame_id=anchor,context=r['context_id'],gt=r['gt'],pred=r['parsed'],correct=r['all_ok'],
            context_detail=r['context_detail'], label_mismatches=label_mismatches,
            raw_action_answers=raw_answers, target_mismatches=target_mismatches,
            raw_speed_match=True, input_sha256_match=True, frames=frame_rows,
            missing_frames=missing_frames, lateral_window_issue=signals.get('lateral_window_issue'),
            sheet=str(sheet)))
        print(f"prepared #{i}: {r['context_id']}",flush=True)
    result=dict(bundle=str(bundle),route_checks=checks,cases=evidence,
                human_review_complete=False,selection='targeted errors plus context controls; not random noise-rate sample')
    (output/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(evidence),routes=len({(r['scenario'],r['route_id']) for r in evidence}),
        contexts=dict(Counter(r['context'] for r in evidence))),ensure_ascii=False))


def main():
    """CLI路径明确，不读取或下载任何模型。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-ids',type=int,nargs='*',default=[])
    p.add_argument('--only-ids',type=int,nargs='+',default=None,
                   help='prepare only these new cases; reuse existing evidence for earlier reviews')
    args=p.parse_args()
    prepare(args.bundle,args.data_root,args.output,args.extra_ids,args.only_ids)


if __name__=='__main__':
    main()
