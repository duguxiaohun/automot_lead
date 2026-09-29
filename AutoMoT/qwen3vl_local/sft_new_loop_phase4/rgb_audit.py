#!/usr/bin/env python3
"""从实际事件候选选跨场景/Town/路线的连续审计窗；生成不等于已目视。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.collection_reader import iter_routes
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import _load_meta


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(collection, data_root):
    """只作候选发现，保留原始映射来源，不输出条件成立标签。"""
    pools = defaultdict(list)
    counts = Counter()
    for source in sorted(collection.glob('*_result.json')):
        scenario = source.name.removesuffix('_result.json')
        per_cell = Counter()
        for route in iter_routes(source):
            counts['routes_scanned'] += 1
            run = data_root / scenario / route['route_id']
            if not run.is_dir() or is_abnormal_lead_route(run, scenario)[0]:
                counts['missing_or_abnormal_routes'] += 1
                continue
            # 每场景/城镇/事件最多保留四条路线；全源扫描仍记录覆盖。
            town = str(route.get('xml_town') or route['route_id'].split('_')[0])
            by_event = defaultdict(list)
            for ann in route.get('annotations', []):
                fid = int(ann['frame_id'])
                if fid < 8:
                    continue
                by_event[str(ann.get('primary_event'))].append(ann)
            for event, anns in by_event.items():
                if per_cell[town, event] >= 4:
                    continue
                # 首/中/尾覆盖接近、冲突与恢复；最终选片时不靠动作真值选答案。
                found = False
                for frac in (.25, .55, .80):
                    ann = anns[min(len(anns)-1, int(len(anns)*frac))]
                    fid = int(ann['frame_id'])
                    if any(not (run/'rgb'/f'{f:04d}.jpg').is_file() for f in range(fid-7, fid+9)):
                        continue
                    contexts, evidence = mapped_contexts(scenario, route['route_id'], fid,
                        ann.get('primary_road_structure', 'UNKNOWN'), event, ann.get('events', [event]))
                    for context in contexts:
                        pools[context].append(dict(scenario=scenario, town=town,
                            route_id=route['route_id'], frame_id=fid, rs=ann.get('primary_road_structure'),
                            context_id=context, event=event, selection_fraction=frac,
                            physical_group=physical_route_group(scenario, route['route_id']),
                            mapping_evidence=evidence))
                        found = True
                if found:
                    per_cell[town, event] += 1
        print(f'[discover] {scenario}: routes={counts["routes_scanned"]} candidates={sum(map(len,pools.values()))}', flush=True)
    return dict(pools), dict(counts)


def select(pools, per_event):
    """贪心覆盖城镇/场景/不同路线和窗位置，不能把多个Rep算独立路线。"""
    selected = []
    for context in CONTEXT_IDS:
        candidates = list(pools.get(context, []))
        towns, scenarios, groups, fractions = Counter(), Counter(), set(), Counter()
        for _ in range(per_event):
            remaining = [r for r in candidates if r['physical_group'] not in groups]
            if not remaining:
                break
            row = min(remaining, key=lambda r: (towns[r['town']]+scenarios[r['scenario']],
                towns[r['town']], scenarios[r['scenario']], fractions[r['selection_fraction']],
                sha_text(f"{r['route_id']}:{r['frame_id']}")))
            selected.append(dict(row, review_id=f'p4_{len(selected):03d}'))
            towns[row['town']] += 1
            scenarios[row['scenario']] += 1
            fractions[row['selection_fraction']] += 1
            groups.add(row['physical_group'])
    return selected


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def render(selected, data_root, output):
    """逐帧原始拼接RGB等比缩放成两页；原始哈希和meta只作审计证据。"""
    from PIL import Image, ImageDraw, ImageFont
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    records = []
    for row in selected:
        run = data_root/row['scenario']/row['route_id']
        record = dict(row, frames=[], panels=[], visual_review_status='pending')
        for page in range(2):
            sheet = Image.new('RGB', (1600, 4*293+65), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8,4), f"{row['review_id']} {row['context_id']} {row['scenario']}", fill='black', font=font)
            draw.text((8,27), f"{row['route_id']} | page {page+1}; chronological row-major; audit-only", fill='black', font=font)
            for pos in range(8):
                fid = row['frame_id']-7+page*8+pos
                rgb, mp = run/'rgb'/f'{fid:04d}.jpg', run/'metas'/f'{fid:04d}.pkl'
                m = _load_meta(mp)
                with Image.open(rgb) as source:
                    im = source.convert('RGB').resize((798,266))
                x,y = (pos%2)*800, 65+(pos//2)*293
                draw.text((x+4,y), f"f{fid} v={float(m['speed']):.2f} road/lane={m.get('road_id')}/{m.get('lane_id')}", fill='black', font=font)
                sheet.paste(im,(x,y+25))
                record['frames'].append(dict(frame_id=fid,rgb=str(rgb.relative_to(data_root)),
                    rgb_sha256=sha(rgb),meta_sha256=sha(mp),speed=float(m['speed']),
                    road_id=m.get('road_id'), lane_id=m.get('lane_id')))
            panel = output/f"{row['review_id']}_{page}.jpg"
            sheet.save(panel,quality=95)
            record['panels'].append(str(panel))
        records.append(record)
    (output/'evidence.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-dir', type=Path, default=ROOT/'keyframe_filter/collection_output')
    p.add_argument('--data-root', type=Path, default=ROOT/'lead_data')
    p.add_argument('--output-dir', type=Path, default=ROOT/'checkpoints/sft_new_loop_phase4_rgb_audit')
    p.add_argument('--per-event', type=int, default=8)
    p.add_argument('--reuse-discovery', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    discovery=a.output_dir/'discovery.json'
    if a.reuse_discovery:
        data=json.loads(discovery.read_text())
    else:
        pools,counts=discover(a.collection_dir,a.data_root)
        data=dict(pools=pools,counts=counts)
        discovery.write_text(json.dumps(data,ensure_ascii=False))
    selected=select(data['pools'],a.per_event)
    records=render(selected,a.data_root,a.output_dir)
    print(json.dumps(dict(windows=len(records), per_context=dict(Counter(r['context_id'] for r in records)),
        generated_not_reviewed=True),ensure_ascii=False))


if __name__ == '__main__':
    main()
