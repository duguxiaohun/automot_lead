"""Collect existing ABC metrics only, no imports of training packages or model loading."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import os, json, hashlib

root = Path.cwd()
run = Path(os.environ.get('ABC_AUDIT_RUN', 'checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636')).expanduser().resolve()
if not (root / 'qwen3vl_local').is_dir():
    raise SystemExit('请先进入当前服务器的 AutoMoT 根目录')
for v in 'abc':
    if not (run / (v + '_bev_only') / 'config.json').is_file():
        raise SystemExit('当前机器缺少 ABC 分组，请修改 ABC_AUDIT_RUN：' + str(run / (v + '_bev_only')))
out = root / 'checkpoints/audit_exports' / ('abc_metrics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True, exist_ok=False)
core, tb, missing = [], [], []
for v in 'abc':
    group = run / (v + '_bev_only')
    core += [p for p in group.glob('*.json') if p.is_file()]
    for name in ('training_audit', 'validation', 'performance', 'progress', 'test', 'eval_test'):
        folder = group / name
        count = 0
        if folder.is_dir():
            for base, dirs, files in os.walk(folder, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in ('cases', 'delivery') and not (Path(base)/d).is_symlink())
                for filename in sorted(files):
                    p = Path(base) / filename
                    if p.suffix in ('.json', '.jsonl') or filename == 'eval.log':
                        core.append(p); count += 1
        if not count:
            missing.append(v + '_bev_only/' + name + ': no matching records')
    fallback = group / 'training_audit.zip'
    if fallback.is_file():
        core.append(fallback)
    if (group / 'tb').is_dir():
        tb += [p for p in (group / 'tb').rglob('*tfevents*') if p.is_file()]
core = sorted(set(core)); tb = sorted(set(tb))
limit = 30_000_000
package = out / 'abc_metrics_supplement.zip'

def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)

def pack(include_tb):
    members = {}
    with ZipFile(package, 'w', ZIP_DEFLATED, compresslevel=9, allowZip64=True) as z:
        for p in core + (tb if include_tb else []):
            before = signature(p); h = hashlib.sha256(); size = 0
            with p.open('rb') as src, z.open(str(p.relative_to(run)), 'w', force_zip64=True) as dst:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    dst.write(block); h.update(block); size += len(block)
            if signature(p) != before or size != before[2]:
                raise RuntimeError('文件正在变化，请运行停止后再收集：' + str(p))
            members[str(p.relative_to(run))] = {'bytes': size, 'sha256': h.hexdigest()}
        manifest = {'run': str(run), 'collection_only': True, 'weights_included': False,
                    'members': members, 'missing': missing,
                    'tensorboard_files_found': len(tb), 'tensorboard_included': include_tb,
                    'omitted_tensorboard': [] if include_tb else [str(p.relative_to(run)) for p in tb]}
        z.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

try:
    print('[CPU] 收集三组已有指标与 TensorBoard；不加载权重', flush=True)
    manifest = pack(True)
    if package.stat().st_size > limit:
        print('[CPU] 超过 30MB，保留全部核心指标并省略原始 TensorBoard', flush=True)
        manifest = pack(False)
    if package.stat().st_size > limit:
        raise RuntimeError('核心指标仍超过 30MB；未发布压缩包，不截断指标')
except BaseException:
    package.unlink(missing_ok=True)
    raise
print(json.dumps({'result_directory': str(out), 'package': str(package),
                  'bytes': package.stat().st_size,
                  'sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
                  'tensorboard_included': manifest['tensorboard_included'],
                  'missing': missing}, ensure_ascii=False, indent=2))
