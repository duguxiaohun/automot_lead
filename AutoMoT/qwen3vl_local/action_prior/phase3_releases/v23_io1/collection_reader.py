"""逐 route 解码大型 collection JSON；内存只保留当前 route 与一个读取块。"""
import json
import re


def iter_routes(path, chunk_size=4 * 1024 * 1024):
    """复用标准 JSON 解码器，避免对 16GB 标注逐字节执行 Python 循环。"""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with open(path, encoding="utf-8") as handle:
        while True:
            block = handle.read(chunk_size)
            buffer += block
            if not started:
                match = re.search(r'"routes"\s*:\s*\[', buffer)
                if match:
                    buffer = buffer[match.end():]
                    started = True
                elif not block:
                    raise ValueError(f"missing routes array: {path}")
                else:
                    continue
            while True:
                buffer = buffer.lstrip(" \r\n\t,")
                if buffer.startswith("]"):
                    return
                try:
                    route, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not block:
                        raise ValueError(f"truncated route JSON: {path}")
                    break
                if not isinstance(route, dict):
                    raise ValueError(f"route must be an object: {path}")
                yield route
                buffer = buffer[end:]
