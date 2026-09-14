#!/usr/bin/env python3
"""把手工逐帧笔记与两种输出的原始评测关联，核验错例覆盖并生成本地审计页。

只引用已有图片，不生成或推断视觉结论；case_index 始终使用 binary 的编号。
"""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path


def read_cases(bundle):
    """读取实际 production cases，拒绝重复编号。"""
    rows = [json.loads(line) for p in sorted((bundle / 'lora_production').glob('cases*.jsonl'))
            for line in p.read_text().splitlines() if line.strip()]
    if not rows or len({r['case_index'] for r in rows}) != len(rows):
        raise ValueError(f'empty or duplicate cases: {bundle}')
    return rows


def identity(row):
    """上下文也是题目身份的一部分，不能只按 RGB 帧号配对。"""
    return tuple(row[k] for k in ('scenario', 'route_id', 'frame_id', 'context_id',
                                  'prompt_road_structure', 'context_detail'))


def positive(answers):
    """按原始 YES 显示答案，空集标成 NONE。"""
    return '+'.join(k for k, v in answers.items() if v == 'YES') or 'NONE'


def render(notes_path, binary_bundle, choice_bundle, output):
    """拒绝缺少手工笔记的 production 错例；报告真实覆盖与缺图情况。"""
    notes = [json.loads(l) for l in notes_path.read_text().splitlines() if l.strip()]
    by_note = {n['case_index']: n for n in notes}
    binary = read_cases(binary_bundle)
    by_id = {r['case_index']: r for r in binary}
    by_identity = {identity(r): r for r in binary}
    if len(by_note) != len(notes) or len(by_identity) != len(binary):
        raise ValueError('duplicate review or binary identity')
    choice = read_cases(choice_bundle)
    paired = {}
    for r in choice:
        b = by_identity[identity(r)]
        if b['gt'] != r['gt'] or b['history_rgb_sha256'] != r['history_rgb_sha256']:
            raise ValueError('paired ground truth or image mismatch')
        if b['case_index'] in paired:
            raise ValueError('duplicate choice identity')
        paired[b['case_index']] = r
    binary_errors = {r['case_index'] for r in binary if not r['all_ok']}
    choice_errors = {i for i, r in paired.items() if not r['all_ok']}
    errors = binary_errors | choice_errors
    missing = sorted(errors - by_note.keys())
    if missing:
        raise ValueError(f'unreviewed production errors: {missing}')
    output.parent.mkdir(parents=True, exist_ok=True)
    cards, unique_frames, missing_frames = [], set(), []
    for i, n in sorted(by_note.items()):
        b = by_id[i]
        if (n['gt'] != b['gt'] or n['pred'] != b['parsed'] or
                n['input_sha256'] != b['history_rgb_sha256'] or
                (n['scenario'], n['route_id'], n['frame_id']) !=
                (b['scenario'], b['route_id'], b['frame_id'])):
            raise ValueError(f'review identity mismatch #{i}')
        sheet = notes_path.parent / n['evidence_sheet']
        if not sheet.is_file():
            raise FileNotFoundError(sheet)
        unique_frames.update((n['scenario'], n['route_id'], f) for f in n['reviewed_frames'])
        if n.get('missing_frames'):
            missing_frames.append(dict(case_index=i, frames=n['missing_frames']))
        c = paired.get(i)
        title = f"#{i} {n['scenario']} / {n['route_id']} / f{n['frame_id']}"
        results = f"GT: {positive(b['gt'])} | binary: {positive(b['parsed'])}"
        if c:
            results += f" | choice #{c['case_index']}: {positive(c['parsed'])}"
        details = n['observation'] + '\n' + n.get('source_review', '')
        tag = n['classification'] + ' / ' + n['label_decision']
        src = html.escape(os.path.relpath(sheet, output.parent), quote=True)
        cards.append(f'<article id="case-{i}"><h2>{html.escape(title)}</h2>'
                     f'<p>{html.escape(results)}</p><p>{html.escape(tag)}</p>'
                     f'<p class="note">{html.escape(details)}</p>'
                     f'<p>逐帧：{html.escape(str(n["reviewed_frames"]))}</p>'
                     f'<a href="{src}"><img loading="lazy" src="{src}" alt="{html.escape(title)}"></a>'
                     '</article>')
    summary = dict(reviewed_cases=len(notes), reviewed_runs=len({n['route_id'] for n in notes}),
                   distinct_reviewed_frame_ids=len(unique_frames), binary_errors=len(binary_errors),
                   choice_errors=len(choice_errors), union_errors=len(errors),
                   correct_controls=len(by_note.keys()-errors), missing_error_notes=missing,
                   incomplete_future_sheets=missing_frames,
                   classification_counts=dict(Counter(n['classification'] for n in notes)),
                   label_decisions=dict(Counter(n['label_decision'] for n in notes)),
                   scope='LoRA production errors only; not base/audit-mode error coverage',
                   not_a_random_dataset_noise_rate_sample=True)
    intro = f"{len(notes)} 个样本；binary 错例 {len(binary_errors)}，choice 错例 {len(choice_errors)}，并集 {len(errors)}，正确对照 {summary['correct_controls']}。"
    output.write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>Phase3 RGB 审计 20260914</title><style>'
        'body{max-width:1500px;margin:24px auto;padding:0 16px;font-family:system-ui;background:#f4f5f6;color:#222}'
        'article{background:white;margin:24px 0;padding:16px;border:1px solid #ddd;border-radius:8px}'
        'img{width:100%;height:auto}h2{font-size:17px;overflow-wrap:anywhere}.note{white-space:pre-wrap}'
        'input{padding:12px;width:90%;font-size:16px}header{position:sticky;top:0;background:#f4f5f6;padding:8px}'
        '</style><header><h1>Phase3 RGB 审计 · 2026-09-14</h1><p>' + intro + '</p>'
        '<p>首行四图为模型输入；后续帧仅供离线取证，+3.25s 只确认边界。缺帧在图片内标红。'
        '数字动作规则复核不等于 RGB 已证明上下文正确；待核结论保留在笔记中。</p>'
        '<input id="q" placeholder="搜索编号、场景、路线、归因、帧号"></header>' + ''.join(cards) +
        '<script>document.getElementById("q").addEventListener("input",function(){'
        'const q=this.value.toLowerCase();document.querySelectorAll("article").forEach(a=>'
        'a.hidden=!a.textContent.toLowerCase().includes(q));});</script></html>', encoding='utf-8')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('notes', 'binary-bundle', 'choice-bundle', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.notes, args.binary_bundle, args.choice_bundle, args.output), ensure_ascii=False))
