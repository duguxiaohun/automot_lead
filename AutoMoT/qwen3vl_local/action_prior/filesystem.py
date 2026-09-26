"""Action / Phase3 数据产物的有限 ESTALE 恢复；不重放流式扫描或模型训练。"""
from __future__ import annotations

import errno
import hashlib
from pathlib import Path
import sys
import time
import uuid

ESTALE_DELAYS = (0.5, 1.0, 2.0, 4.0)


def retry_estale(operation, description, *, delays=None):
    """每次重新按路径打开；权限、空间、EIO 等错误不转换成成功。"""
    delays = ESTALE_DELAYS if delays is None else delays
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == len(delays):
                raise
            delay = delays[attempt]
            print(f'[filesystem] ESTALE: {description}; retry {attempt + 1}/'
                  f'{len(delays)} in {delay}s', file=sys.stderr, flush=True)
            time.sleep(delay)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_file(temporary, destination, *, expected_sha256=None):
    """发布已关闭的完整文件；处理 rename 已提交但返回 ESTALE 的不确定结果。"""
    temporary, destination = Path(temporary), Path(destination)
    expected = expected_sha256 or retry_estale(lambda: _sha256(temporary), f'hash {temporary}')

    def replace_once():
        try:
            temporary.replace(destination)
        except OSError as exc:
            if exc.errno not in (errno.ESTALE, errno.ENOENT):
                raise
            try:
                matches = _sha256(destination) == expected
            except FileNotFoundError:
                matches = False
            # 已移动的源可以消失；仅目标与发布前快照相同才确认成功。
            if not matches:
                raise exc

    try:
        retry_estale(replace_once, f'publish {temporary} -> {destination}')
    except BaseException:
        print(f'[filesystem] publication failed: {temporary} -> {destination}; '
              'remaining temporary output retained; check mount health before retry',
              file=sys.stderr, flush=True)
        raise


def atomic_write_text(path, text, *, encoding='utf-8'):
    """小型元信息完整写入独立临时文件后发布；重复写入只作用于该临时文件。"""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    retry_estale(lambda: temporary.write_text(text, encoding=encoding), f'write {temporary}')
    publish_file(temporary, path)
