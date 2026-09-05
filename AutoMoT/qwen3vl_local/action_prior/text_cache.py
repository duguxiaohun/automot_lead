"""冻结模型的可复用文本结果；不缓存 GPU KV，不保存 RGB。"""

import hashlib
import json
import sqlite3
import zlib
from qwen3vl_local.action_prior.contracts import digest


class TextCache:
    """每 rank 独立 SQLite，避免共享训练进程争写或 NFS 多写者锁。"""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )

    def key(self, identity, images, navigation, sample_key):
        """直接绑定四张解码 RGB 的字节与尺寸，换图不会误用旧先验。"""
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

    def get(self, key):
        """命中只返回文本/计数，KV 每次仍由 base 新 prefill。"""
        result = self.db.execute(
            "SELECT payload FROM results WHERE key=?", (key,)
        ).fetchone()
        return json.loads(zlib.decompress(result[0])) if result else None

    def put(self, key, value):
        """裁掉重复长 prompt，保留原始回答和 prompt 指纹供溯源。"""
        value = dict(value)
        value["calls"] = [
            dict(
                phase=c["phase"],
                variant=c["variant"],
                keys=c["keys"],
                response=c["response"],
                prompt_sha256=hashlib.sha256(c["prompt"].encode()).hexdigest(),
                history_responses=[h[1] for h in c["history"]],
            )
            for c in value["calls"]
        ]
        blob = zlib.compress(json.dumps(value).encode(), level=3)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO results VALUES (?, ?)", (key, blob))
