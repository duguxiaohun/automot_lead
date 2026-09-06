"""跨 rank 冻结文本缓存：分桶文件 + POSIX 锁 + 原子发布，不保存 RGB/GPU KV。"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import os
import tempfile
import zlib
from qwen3vl_local.action_prior.contracts import digest


class TextCache:
    """所有 rank 共用目录；同 key 首次生成在锁内二次检查，进程退出自动释放锁。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def key(self, identity, images, navigation, sample_key):
        """绑定实际四图、完整执行合同、导航和确定性问法 seed。"""
        rgb = [
            (im.size, im.mode, hashlib.sha256(im.tobytes()).hexdigest())
            for im in images
        ]
        return digest(
            dict(
                identity=identity,
                images=rgb,
                navigation=navigation,
                sample_key=sample_key,
            )
        )

    def _file(self, key):
        """限制桶数并拒绝非哈希文件名。"""
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("cache key must be SHA256")
        folder = self.path / key[:3]
        folder.mkdir(exist_ok=True)
        return folder / (key + ".json.z")

    @contextmanager
    def _lock(self, key):
        """同桶可能等待；4096 个桶避免为每帧留下单独锁文件。"""
        with (self._file(key).parent / ".lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def get(self, key):
        """原子文件可无锁读取；损坏直接报错，不能悄悄换语言条件。"""
        path = self._file(key)
        try:
            blob = path.read_bytes()
        except FileNotFoundError:
            return None
        return json.loads(zlib.decompress(blob))

    def put(self, key, value):
        """写临时文件后原子发布；保留原始回答但压缩重复长 prompt。"""
        value = dict(value)
        value["calls"] = [
            dict(
                **{k: v for k, v in c.items() if k not in ("prompt", "history")},
                prompt_sha256=hashlib.sha256(c["prompt"].encode()).hexdigest(),
                history_responses=[h[1] for h in c["history"]],
            )
            for c in value["calls"]
        ]
        path = self._file(key)
        fd, tmp = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(zlib.compress(json.dumps(value).encode(), level=3))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def get_or_compute(self, key, compute):
        """跨进程 miss 只计算一次；命中也继续由调用方重建 base KV。"""
        value = self.get(key)
        if value is not None:
            return value, True
        with self._lock(key):
            value = self.get(key)
            if value is not None:
                return value, True
            value = compute()
            self.put(key, value)
            return value, False
