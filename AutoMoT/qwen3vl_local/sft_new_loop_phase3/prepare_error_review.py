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


def digest(path):
    """记录原始文件身份，不把压缩拼图当作模型实际输入。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        valid = r['gt']['INVALID_ACTION_CONTEXT'] == 'NO'
        label_mismatches = [k for k, v in r['gt'].items() if k != 'INVALID_ACTION_CONTEXT'
                            and valid and (labels is None or labels[k] != (v == 'YES'))]
        frames = [int(f.stem) for f in inputs] + list(range(anchor+1, anchor+13))
        canvas = Image.new('RGB', (2304, 4*224+50), '#101010')
        draw = ImageDraw.Draw(canvas)
        positive = lambda d: '+'.join(k for k,v in d.items() if v == 'YES') or 'NONE'
        draw.text((8,5), f"#{i} {r['scenario']} f{anchor} {r['context_id']} RS={r['prompt_road_structure']}", fill='white')
        draw.text((8,23), f"GT {positive(r['gt'])} | PRED {positive(r['parsed'])} | FIRST ROW=INPUT, other rows=FUTURE EVIDENCE", fill='yellow')
        frame_rows = []
        for j, f in enumerate(frames):
            rgb, meta = run/'rgb'/f'{f:04d}.jpg', traj.metas.get(f)
            x, y = (j % 4)*576, 50+(j//4)*224
            if not rgb.exists() or meta is None:
                draw.text((x+4,y+4),f'MISSING f{f}',fill='red')
                continue
            with Image.open(rgb) as im:
                canvas.paste(im.convert('RGB').resize((576,192)),(x,y))
            speed = float(meta['speed'])
            caption = f"f{f} t={(f-anchor)*.25:+.2f}s v={speed:.3f} road/lane={meta.get('road_id')}/{meta.get('lane_id')} brake={int(bool(meta.get('brake')))}"
            draw.text((x+3,y+195),caption,fill='yellow' if j<4 else 'white')
            frame_rows.append(dict(frame=f, input=j<4, rgb=str(rgb), rgb_sha256=digest(rgb),
                meta_sha256=digest(run/'metas'/f'{f:04d}.pkl'), speed=speed,
                road_id=meta.get('road_id'), lane_id=meta.get('lane_id'), section_id=meta.get('section_id'),
                brake=bool(meta.get('brake')), throttle=float(meta.get('throttle',0)),
                vehicle_hazard=bool(meta.get('vehicle_hazard')), light_hazard=bool(meta.get('light_hazard'))))
        sheet=output/f'case_{i:03d}.jpg'
        canvas.save(sheet,quality=92)
        evidence.append(dict(case_index=i, scenario=r['scenario'], route_id=r['route_id'],
            frame_id=anchor,context=r['context_id'],gt=r['gt'],pred=r['parsed'],correct=r['all_ok'],
            context_detail=r['context_detail'], label_mismatches=label_mismatches,
            raw_speed_match=True, input_sha256_match=True, frames=frame_rows, sheet=str(sheet)))
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
