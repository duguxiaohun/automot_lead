"""三条 action 入口自动更新的中途审计；仅打包轻量指标，不读取权重或原始数据。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from qwen3vl_local.action_prior.audit_bundle import pack


def _write(path, value):
    """原子发布 JSON，强制终止时读者只能看到旧版或完整新版。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _read(path):
    """已提交 JSON 损坏时直接报错，不把空数据当作成功审计。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record_window(out, step, values):
    """保存独立日志窗口，ZIP只取已提交checkpoint之前的记录。"""
    _write(Path(out) / 'training_audit/windows' / f'step_{step:08d}.json',
           dict(optimizer_step=step, metrics=values))


def publish(out, *, step, cursor, best, plan, args, train_metrics, validation, performance, reason):
    """同目录临时ZIP完成后原子替换；上次有效ZIP在新包成功前始终可用。"""
    out = Path(out)
    directory = out / 'training_audit'
    epoch = int(cursor.get('validation_epoch', cursor['epoch'] - (cursor['micro'] == 0))) + 1
    complete = bool(cursor.get('validation_full_epoch') or cursor['micro'] == 0)
    state = dict(schema='action_training_audit_v1', optimizer_step=step, epoch=epoch,
                 epoch_training_complete=complete, validation_pending=bool(cursor.get('validation_pending')),
                 checkpoint='latest.pt', checkpoint_included=False, reason=reason,
                 best_selection_score=best if math.isfinite(best) else None,
                 train_metrics=train_metrics, validation=validation, performance=performance,
                 next_cursor={key: cursor[key] for key in ('epoch', 'micro')})
    previous_path = directory / 'latest.json'
    if previous_path.is_file() and not cursor.get('validation_pending'):
        previous = _read(previous_path)
        if previous['optimizer_step'] == step and previous['epoch'] == epoch:
            if validation is None:
                state['validation'] = previous.get('validation')
            if not train_metrics:
                state['train_metrics'] = previous.get('train_metrics', {})
    # 完整验证成功时使用原待办cursor标明所属epoch，但不再宣称仍待验证。
    if validation is not None:
        state['validation_pending'] = False
    _write(directory / 'latest.json', state)
    _write(directory / 'epochs' / f'epoch_{epoch:03d}.json', state)
    history = []
    for path in sorted((directory / 'epochs').glob('epoch_*.json')):
        value = _read(path)
        if value['optimizer_step'] <= step:
            history.append(value)
    windows = []
    for path in sorted((directory / 'windows').glob('step_*.json'), reverse=True):
        if int(path.stem.split('_')[1]) <= step:
            windows.append(path)
    validation_paths = []
    for path in (out / 'validation').glob('*_step*.json'):
        match = re.search(r'_step(\d+)\.json$', path.name)
        if match and int(match.group(1)) <= step:
            validation_paths.append(path)
    validation_paths.sort(key=lambda p: int(re.search(r'_step(\d+)\.json$', p.name).group(1)), reverse=True)
    # 核心是当前进度、完整epoch曲线摘要和实际配置；详细历史按30MB上限选入。
    headlines = []
    for item in history:
        headlines.append(dict(epoch=item['epoch'], optimizer_step=item['optimizer_step'],
            training_complete=item['epoch_training_complete'], validation_pending=item['validation_pending'],
            train={k: v for k, v in item['train_metrics'].items()
                   if k in ('samples', 'loss', 'route_fm_mse', 'waypoint_fm_mse')},
            validation={k: v for k, v in (item['validation'] or {}).items()
                        if k in ('samples', 'loss', 'route_ade_m', 'waypoint_ade_m', 'route_fde_m',
                                 'waypoint_fde_m', 'event_balanced_route_ade_m', 'event_balanced_waypoint_ade_m')},
            best_selection_score=item['best_selection_score']))
    with tempfile.TemporaryDirectory(prefix='.training_audit_', dir=out) as temporary:
        staging = Path(temporary)
        _write(staging / 'metrics.json', dict(latest=state, epoch_history=headlines,
               window_records_total=len(windows), window_records_selected=min(200, len(windows)),
               validation_records_total=len(validation_paths), validation_records_selected=min(200, len(validation_paths))))
        _write(staging / 'config.json', vars(args))
        _write(staging / 'training_plan.json', plan)
        (staging / 'paper_table.md').write_text(
            '# Action 中途训练审计\n\n'
            f'已提交 optimizer step：{step}；当前 epoch：{epoch}；触发原因：{reason}。\n\n'
            '先读 metrics.json 的 latest 和 epoch_history。validation_pending=true 表示训练已保存、'
            '完整验证尚未完成，不能把历史 best 当本轮成绩。train_metrics 是本轮已呈现样本的全rank汇总。\n\n'
            'validation/ 含近期验证；audit/ 含完整epoch记录和最近200个日志窗口（loss/LR/梯度/更新幅度）。'
            'AUDIT_MANIFEST.json 说明实际收录、哈希和大小裁剪。没有权重、缓存、RGB、原始TensorBoard或聊天记录。\n\n'
            '本包用于中途审计，不是可恢复训练备份。强杀时可能落后于磁盘最新checkpoint；'
            '以包内optimizer_step为准。吞吐为本次进程会话，不含停机时间。\n', encoding='utf-8')
        for name in ('selected_priors.json', 'condition_contract.json', 'model_contract.json', 'run_manifest.json'):
            source = out / name
            if source.is_file() and not source.is_symlink():
                shutil.copyfile(source, staging / name)
        audit = staging / 'audit'
        audit.mkdir()
        for item in history:
            _write(audit / f"epoch_{item['epoch']:03d}.json", item)
        for source in windows[:200]:
            shutil.copyfile(source, audit / source.name)
        for source in validation_paths[:200]:
            target = staging / 'validation' / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
        return pack(staging, output=out / 'training_audit.zip')
