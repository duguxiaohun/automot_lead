"""生成本地 RGB/候选动作审计页；引用原图，不复制图片、不把机器扫描冒充人工确认。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_review(candidate_index: Path, notes_path: Path, data_root: Path, output: Path):
    """每条候选保留逐帧速度、车道身份和实际 prompt，并提供原始 RGB 帧切换。"""
    from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt

    routes = {}
    for line in notes_path.read_text().splitlines():
        note = json.loads(line)
        key = f"{note['scenario']}/{note['route_id']}"
        routes.setdefault(key, {"notes": [], "candidates": {}})["notes"].append(note)
    for line in candidate_index.open():
        row = json.loads(line)
        key = f"{row['scenario']}/{row['route_id']}"
        entry = routes.setdefault(key, {"notes": [], "candidates": {}})
        spec = make_prompt_spec(variant="all_random_order", answers=row["answers"],
            seed_key=f"review:{key}:{row['frame_id']}", context_id=row["context_id"],
            road_structure=row["prompt_road_structure"], goal_xy=row["goal_ego_xy"],
            context_detail=row.get("context_detail", ""))
        item = {k: row.get(k) for k in ("context_id", "true_rs", "prompt_road_structure",
                "event_codes", "answers", "invalid_source", "invalid_reason", "goal_ego_xy", "action_evidence",
                "mapping_evidence", "context_detail")}
        item["prompt"] = build_action_prompt(spec=spec)
        entry["candidates"].setdefault(str(row["frame_id"]), []).append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    for key, entry in routes.items():
        # 同一原图使用路径引用；用户可检查候选前后未被抽中的帧。
        images = sorted((data_root / key / "rgb").glob("*.jpg"))
        entry["frames"] = [dict(frame=int(p.stem), src=os.path.relpath(p, output.parent))
                           for p in images if p.stem.isdigit()]
        for note in entry["notes"]:
            if note.get("sheet"):
                note["sheet"] = os.path.relpath(note["sheet"], output.parent)
    payload = json.dumps(routes, ensure_ascii=False).replace("<", "\\u003c")
    page = r"""<!doctype html><meta charset="utf-8"><title>Phase3 RGB 与动作候选复核</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}select{max-width:95%;padding:8px}
img{width:100%;max-width:1500px}pre{white-space:pre-wrap;background:white;padding:12px}input{width:70%}
.bar{position:sticky;top:0;background:#f5f5f5;padding:10px}summary,button{cursor:pointer}</style>
<h1>Phase3 RGB 与动作候选复核</h1>
<p>候选动作尚未经全量人工确认。速度窗含未来信息，仅供离线审计，不进入模型 prompt。
没有候选记录的帧表示未收录，不代表动作全 NO。方向：x 前，y 负左、正右。</p>
<select id="route"></select><div class="bar"><button id="prev">上一帧</button>
<input id="frame" type="range" min="0"><button id="next">下一帧</button><b id="fid"></b></div>
<img id="rgb"><pre id="notes"></pre><div id="candidate"></div>
<script>const data=PAYLOAD;
const route=document.querySelector('#route'),slider=document.querySelector('#frame');
for(const key of Object.keys(data).sort()){const o=document.createElement('option');o.value=key;o.textContent=key;route.append(o)}
function render(){const r=data[route.value],f=r.frames[Number(slider.value)];if(!f)return;
document.querySelector('#rgb').src=f.src;document.querySelector('#fid').textContent=' frame '+f.frame;
document.querySelector('#notes').textContent=r.notes.map(n=>`${n.review_level||'审计记录'} ${n.original_rgb_frames ? JSON.stringify(n.original_rgb_frames) : n.start_frame+'-'+n.end_frame}：${n.note||n.reason}`).join('\n');
const box=document.querySelector('#candidate');box.replaceChildren();
for(const raw of r.candidates[String(f.frame)]||[]){const {prompt,...info}=raw;
const pre=document.createElement('pre');pre.textContent=JSON.stringify(info,null,2);box.append(pre);
const d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('pre');
s.textContent='查看此候选的实际文本 prompt';p.textContent=prompt;d.append(s,p);box.append(d)}
if(!box.childNodes.length)box.textContent='此帧没有候选动作记录。'}
route.onchange=()=>{slider.max=Math.max(0,data[route.value].frames.length-1);slider.value=0;render()};slider.oninput=render;
document.querySelector('#prev').onclick=()=>{slider.value=Number(slider.value)-1;render()};
document.querySelector('#next').onclick=()=>{slider.value=Number(slider.value)+1;render()};
document.onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.key==='ArrowLeft')document.querySelector('#prev').click()};route.onchange();
</script>""".replace("PAYLOAD", payload)
    output.write_text(page)
    return {"routes": len(routes), "output": str(output), "candidate_index": str(candidate_index),
            "manual_notes": str(notes_path), "action_review_status": "candidates_not_fully_verified"}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--notes", type=Path, default=Path(__file__).with_name("rgb_mapping_review_20260905.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "lead_data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(args.candidate_index.resolve(), args.notes.resolve(),
                                 args.data_root.resolve(), args.output.resolve()), ensure_ascii=False))
