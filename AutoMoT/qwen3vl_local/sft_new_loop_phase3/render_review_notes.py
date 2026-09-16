#!/usr/bin/env python3
"""把已落盘的逐帧人工笔记与现有证据拼图关联，不生成或推断审计结论。"""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def render(evidence_path, notes_path, output):
    """核对笔记身份、帧号和原图指纹，只报告本次精选窗口的笔记覆盖。"""
    evidence = json.loads(evidence_path.read_text())
    cases = {r['case_index']: r for r in evidence['cases']}
    notes = [json.loads(line) for line in notes_path.read_text().splitlines() if line.strip()]
    seen, sections = set(), []
    for note in notes:
        index = note['case_index']
        if index in seen or index not in cases:
            raise ValueError(f'duplicate or unknown review case: {index}')
        seen.add(index)
        case = cases[index]
        for key in ('scenario', 'route_id', 'frame_id'):
            if note[key] != case[key]:
                raise ValueError(f'review identity mismatch: #{index}/{key}')
        if note['reviewed_frames'] != [r['frame'] for r in case['frames']]:
            raise ValueError(f'review frame mismatch: #{index}')
        if note['input_rgb_sha256'] != [r['rgb_sha256'] for r in case['frames'] if r['input']]:
            raise ValueError(f'review input fingerprint mismatch: #{index}')
        # 拼图默认就在 evidence.json 同目录，避免依赖生成时的工作目录。
        sheet = evidence_path.parent / Path(case['sheet']).name
        if not sheet.is_file():
            raise FileNotFoundError(sheet)
        href = quote(os.path.relpath(sheet.resolve(), output.parent.resolve()))
        title = f"#{index} {case['scenario']} / {case['route_id']} / f{case['frame_id']}"
        details = dict(context=case['context'], gt=case['gt'], pred=case['pred'],
                       frames=note['reviewed_frames'], missing_frames=case['missing_frames'])
        sections.append(f'<section id="case-{index}"><h2>{html.escape(title)}</h2>'
            f'<p>{html.escape(note["observation"])}</p>'
            f'<details><summary>帧号与答案</summary><pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details>'
            f'<a href="{href}"><img loading="lazy" src="{href}" alt="case {index} RGB evidence"></a></section>')
    missing = sorted(cases.keys() - seen)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Phase3 逐帧 RGB 审计笔记</title><style>'
        'body{font-family:system-ui;margin:24px;background:#151515;color:#eee}'
        'section{border-top:1px solid #777;padding:20px 0}img{width:100%;height:auto}'
        'pre{white-space:pre-wrap}p{line-height:1.7}a{color:#8cc8ff}</style>'
        f'<h1>逐帧 RGB 审计笔记：{len(notes)}/{len(cases)} 个精选窗口</h1>'
        '<p>这是定向错例与正确对照审计，不是全测试集覆盖，也不是随机标签噪声率调查。'
        '未来帧只作离线证据，不进入模型输入；+3.25 秒仅确认末端越线。'
        '人工笔记与自动标签复算分别保留，拼图本身不代表标签已人工确认。</p>'
        f'<p>未附笔记的精选窗口：{html.escape(str(missing))}</p>'
        + ''.join(sections) + '</html>', encoding='utf-8')
    return dict(selected_cases=len(cases), notes=len(notes), missing_notes=missing)


def main():
    """读取已有证据；不复制 RGB、不读取模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.evidence, args.notes, args.output)))


if __name__ == '__main__':
    main()
