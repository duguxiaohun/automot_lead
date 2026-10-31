"""跨图片的有界纯文本前缀缓存；仅用于固定权重、单进程串行推理。

AUTOMOT_TEXT_PREFIX=1 启用新输入合同，=0 保留旧合同。训练仍全量反传。
缓存只在内存中存在，包含完整 hybrid 状态，绝不保存图片相关状态。
"""
from collections import OrderedDict
import os

import torch

CONTRACT = 'text_before_images_v1'


def enabled():
    """严格读取开关，防止拼写错误静默改变输入合同。"""
    value = os.environ.get('AUTOMOT_TEXT_PREFIX', '0')
    if value not in ('0', '1'):
        raise ValueError('AUTOMOT_TEXT_PREFIX must be 0 or 1')
    return value == '1'


def identity():
    """布局属于权重合同；缓存容量、命中与否不改变输入。"""
    return CONTRACT if enabled() else 'images_before_text_legacy'


def cache_enabled():
    """同一新布局可以禁用缓存做完整前向对照，不改变提示词指纹。"""
    value = os.environ.get('AUTOMOT_TEXT_PREFIX_CACHE', '1')
    if value not in ('0', '1'):
        raise ValueError('AUTOMOT_TEXT_PREFIX_CACHE must be 0 or 1')
    return enabled() and value == '1'


def bind_hash(digest):
    """保留旧指纹字节，仅新布局增加身份。"""
    import hashlib
    return hashlib.sha256((CONTRACT + ':' + digest).encode()).hexdigest() if enabled() else digest


def content(images, text):
    """同一入口供训练和推理构造图文顺序，不改任何问题或标定。"""
    visual = [dict(type='image', image=image) for image in images]
    words = [dict(type='text', text=text)]
    return words + visual if enabled() else visual + words


def _native(model):
    """只接受本地 Qwen3.5，避免套用到不兼容的 cache 实现。"""
    raw = model.module if hasattr(model, 'module') else model
    raw = raw.get_base_model() if hasattr(raw, 'get_base_model') else raw
    if getattr(raw.config, 'model_type', None) != 'qwen3_5':
        raise ValueError('Text prefix cache requires native Qwen3.5')
    return raw


def _weight_identity(model):
    """版本计数绑定 optimizer/load_state_dict、adapter、设备与精度。"""
    adapters = tuple((name, str(getattr(m, 'active_adapters', None)),
                      str(getattr(m, 'disable_adapters', None)),
                      str(getattr(m, 'merged_adapters', None)))
                     for name, m in model.named_modules() if hasattr(m, 'active_adapters'))
    tensors = []
    for name, p in (*model.named_parameters(), *model.named_buffers()):
        try:
            version = p._version
        except RuntimeError as exc:
            raise ValueError('Unversioned weights: cannot safely retain prefix cache') from exc
        tensors.append((name, id(p), p.data_ptr(), version, str(p.dtype), str(p.device)))
    autocast = tuple((device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
                     for device in ('cpu', 'cuda'))
    return id(model), adapters, tuple(tensors), autocast


def _cache_bytes(cache):
    """统计本地 hybrid cache 的全部可变状态，不重复计数同一 tensor。"""
    tensors = {}
    for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
        if not hasattr(cache, name):
            raise ValueError('Incomplete hybrid prefix cache: ' + name)
        for value in getattr(cache, name):
            if value is not None:
                tensors[id(value)] = value
    return sum(t.numel() * t.element_size() for t in tensors.values())


def cache_limits():
    """Per-model/rank upper bounds, allocated lazily; never silently override a budget."""
    result={}
    for key,env,default in (('max_entries','AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES',2),
                            ('max_bytes','AUTOMOT_TEXT_PREFIX_CACHE_MIB',1024),
                            ('max_tokens','AUTOMOT_TEXT_PREFIX_CACHE_TOKENS',8192)):
        raw=os.environ.get(env,str(default))
        if not raw.isascii() or not raw.isdecimal() or int(raw)<=0:
            raise ValueError(env+' must be a positive integer')
        result[key]=int(raw)*(1024**2 if key=='max_bytes' else 1)
    return result


class TextPrefixCache:
    """每张卡/模型实例独立的 LRU，换图片保留前缀，换权重立即失效。"""

    def __init__(self, max_entries=None, max_bytes=None, max_tokens=None):
        """容量同时按条目、实际字节和 token 限制。"""
        defaults=cache_limits()
        max_entries=defaults['max_entries'] if max_entries is None else max_entries
        max_bytes=defaults['max_bytes'] if max_bytes is None else max_bytes
        max_tokens=defaults['max_tokens'] if max_tokens is None else max_tokens
        if any(type(v) is not int for v in (max_entries,max_bytes,max_tokens)) or min(max_entries, max_bytes, max_tokens) <= 0:
            raise ValueError('Positive cache limits required')
        self.max_entries, self.max_bytes, self.max_tokens = max_entries, max_bytes, max_tokens
        self.entries = OrderedDict()
        self.weights = None
        self.stats = dict(requests=0, hits=0, misses=0, evictions=0, invalidations=0,
                          reused_text_tokens=0, computed_text_prefix_tokens=0,
                          full_hits=0, partial_hits=0, promotions=0)
        self.last = {}

    def clear(self):
        """模型外部非标准权重修改后必须显式清空；不落盘。"""
        self.entries.clear()
        self.weights = None

    def report(self):
        """只返回统计；不泄漏输入文本、图像或缓存内容。"""
        return dict(self.stats, entries=len(self.entries), bytes=sum(v[1] for v in self.entries.values()),
                    max_entries=self.max_entries, max_bytes=self.max_bytes, max_tokens=self.max_tokens,
                    scope='process_local_text_only_complete_hybrid_state', last=dict(self.last))

    @torch.no_grad()
    def prefill(self, model, inputs):
        """用当前完整序列计算 M-RoPE，再从纯文本 cache 分叉当前图片。"""
        if model.training:
            self.clear()
            raise ValueError('Persistent text prefix cache is inference-only')
        native = _native(model)
        from . import cache_policy
        if hasattr(native,cache_policy.ATTR):
            import json
            from types import SimpleNamespace
            cache_policy.validate(json.loads(getattr(native,cache_policy.ATTR)),SimpleNamespace(_text_prefix_cache=self))
        ids = inputs['input_ids']
        mask = inputs.get('attention_mask', torch.ones_like(ids))
        if ids.ndim != 2 or ids.shape[0] != 1 or not bool((mask == 1).all()):
            raise ValueError('Text prefix cache requires unpadded batch size one')
        if any(k in inputs for k in ('labels', 'inputs_embeds', 'past_key_values', 'position_ids',
                                     'cache_position', 'pixel_values_videos', 'video_grid_thw')):
            raise ValueError('Unsupported text-prefix input keys')
        starts = torch.where(ids[0] == native.config.vision_start_token_id)[0]
        if not len(starts) or int(starts[0]) == 0:
            raise ValueError('Expected a nonempty text prefix before the first image')
        n = min(int(starts[0]), self.max_tokens)
        if bool(((ids[:, :n] == native.config.image_token_id) |
                 (ids[:, :n] == native.config.video_token_id)).any()):
            raise ValueError('Visual tokens in reusable text prefix')
        mm = inputs.get('mm_token_type_ids', (ids == native.config.image_token_id).long())
        if bool(mm[:, :n].any()):
            raise ValueError('Multimodal tokens in reusable text prefix')
        weights = _weight_identity(model)
        if self.weights != weights:
            if self.weights is not None:
                self.stats['invalidations'] += 1
            self.entries.clear()
            self.weights = weights
        # Always recompute positions/delta for the NEW images, including changed grids.
        positions, delta = native.model.get_rope_index(ids,
            image_grid_thw=inputs.get('image_grid_thw'), attention_mask=mask, mm_token_type_ids=mm)
        if not torch.equal(positions[:, :, :n], torch.arange(n, device=ids.device)[None, None].expand(3, 1, n)):
            raise ValueError('Reusable text prefix has nonstandard rotary positions')
        from qwen3vl_local.engine import _clone_cache
        key = tuple(ids[0, :n].tolist())
        candidates = [k for k in self.entries if len(k) <= n and key[:len(k)] == k]
        # A shorter retained prefix introduces an extra arithmetic partition.
        # Canonically rebuild the new prefix so answers do not depend on cache history.
        promoting = bool(candidates) and key not in self.entries
        old_key = key if key in self.entries else None
        reused = len(old_key) if old_key else 0
        hit = old_key is not None
        full_hit = reused == int(starts[0])
        self.stats['requests'] += 1
        self.stats['hits' if hit else 'misses'] += 1
        if hit:
            prefix, size = self.entries.pop(old_key)
            self.entries[old_key] = prefix, size
            self.stats['reused_text_tokens'] += reused
            self.stats['full_hits' if full_hit else 'partial_hits'] += 1
        if reused < n:
            # Build the canonical complete (or token-capped) text prefix in one call.
            kwargs = dict(past_key_values=_clone_cache(prefix)) if hit else {}
            out = model(input_ids=ids[:, reused:n], attention_mask=mask[:, :n],
                position_ids=positions[:, :, reused:n],
                cache_position=torch.arange(reused, n, device=ids.device),
                logits_to_keep=1, use_cache=True, return_dict=True, **kwargs)
            if hit and prefix.get_seq_length() != reused:
                raise ValueError('Text promotion mutated retained prefix')
            prefix = out.past_key_values
            size = _cache_bytes(prefix)
            del out
            self.stats['computed_text_prefix_tokens'] += n - reused
            self.stats['promotions'] += int(promoting)
            # Oversized entries do not evict otherwise useful retained entries.
            if size <= self.max_bytes:
                while self.entries and (len(self.entries) >= self.max_entries or
                                       sum(v[1] for v in self.entries.values()) + size > self.max_bytes):
                    self.entries.popitem(last=False)
                    self.stats['evictions'] += 1
                self.entries[key] = prefix, size
        suffix = dict(inputs)
        for name in ('input_ids', 'mm_token_type_ids', 'token_type_ids'):
            if name in suffix:
                suffix[name] = suffix[name][:, n:]
        # Full attention mask includes cached tokens; pixels/grids remain current.
        suffix.update(attention_mask=mask, position_ids=positions[:, :, n:],
                      cache_position=torch.arange(n, ids.shape[1], device=ids.device),
                      past_key_values=_clone_cache(prefix))
        native.model.rope_deltas = delta
        output = model(**suffix, use_cache=True, return_dict=True)
        output.rope_deltas = delta
        if prefix.get_seq_length() != n:
            raise ValueError('Image branch mutated retained text prefix')
        self.last = dict(hit=hit, full_hit=full_hit, reused_tokens=reused,
                         computed_tokens=n-reused, complete_text_tokens=int(starts[0]),
                         prefix_tokens=n, suffix_tokens=ids.shape[1]-n,
                         stored=size <= self.max_bytes, prefix_bytes=size)
        return output


def bank(bundle):
    """缓存挂在 bundle 上，不跨 rank、不跨独立权重共享。"""
    cache = getattr(bundle, '_text_prefix_cache', None)
    if cache is None:
        cache = TextPrefixCache()
        bundle._text_prefix_cache = cache
    return cache


def prefill(bundle, inputs):
    """旧模式保持完整前向，新模式复用纯文本。"""
    from .cache_policy import check
    check(bundle)
    if cache_enabled():
        return bank(bundle).prefill(bundle.model, inputs)
    return bundle.model(**inputs, use_cache=True, return_dict=True)


def report(bundle):
    """供原生 metrics 使用，避免把未执行的缓存路径记为启用成功。"""
    cache = getattr(bundle, '_text_prefix_cache', None)
    return dict(input_layout=identity(), cache_enabled=cache_enabled(),
                execution='serial_requests_no_frame_reordering', cache=cache.report() if cache else None)


@torch.no_grad()
def benchmark(bundle, requests, *, max_new_tokens=160, repeats=2):
    """同分段重算/复用严格配对；完整前向差异单列，流计时包含 clone 与失效检查。

    requests 为已经 processor 编码、已移至设备的独立输入；图片处理不在计时内。
    每轮清空缓存后按原顺序回放，报告真实 miss，不能只计理想命中。
    """
    import time
    from qwen3vl_local.sft_new_loop_phase12.cache import _decode
    if bundle.model.training or not requests or repeats < 1 or max_new_tokens < 1:
        raise ValueError('Expected eval model, requests and positive benchmark budgets')

    def sync():
        """计时前后同步当前设备。"""
        if bundle.device.type == 'cuda':
            torch.cuda.synchronize(bundle.device)

    source=getattr(bundle,'_text_prefix_cache',None) or TextPrefixCache()
    limits={k:getattr(source,k) for k in ('max_entries','max_bytes','max_tokens')}
    def run(shared, *, fresh=False):
        """完整回放一次流，缓存不跨独立计时试验。"""
        cache = TextPrefixCache(**limits)
        answers, times = [], []
        sync(); started = time.perf_counter()
        for item in requests:
            if fresh:cache = TextPrefixCache(**limits)
            sync(); begin = time.perf_counter()
            out = (cache.prefill(bundle.model, item) if shared else
                   bundle.model(**item, use_cache=True, return_dict=True))
            answers.append(_decode(bundle, item, out, max_new_tokens))
            del out
            sync(); times.append(time.perf_counter() - begin)
        sync()
        return dict(seconds=time.perf_counter()-started, request_seconds=times,
                    answers=answers, cache=cache.report() if shared else None)

    def assert_same(a, b):
        """不能仅比较解析结果；包含每一个输出 token、EOS 和截断状态。"""
        for number, (left, right) in enumerate(zip(a['answers'], b['answers'])):
            if left != right:
                raise ValueError(f'Text prefix changes generated tokens/stop/text at request {number}')

    cold = run(True); reference = run(False)
    recomputed = run(True,fresh=True); assert_same(cold,recomputed)
    from qwen3vl_local.sft_new_loop_phase12 import cache as paired,cache_structure
    boundary=min(int(torch.where(requests[0]['input_ids'][0]==_native(bundle.model).config.vision_start_token_id)[0][0]),cold['cache']['max_tokens'])
    other=next((x for x in requests[1:] if torch.equal(x['input_ids'][:,:boundary],
        requests[0]['input_ids'][:,:boundary]) and
        not torch.equal(x['pixel_values'],requests[0]['pixel_values'])),None)
    structure=cache_structure.measure(bundle,requests[0],paired,other_item=other)
    cache_structure.verify_reuse(structure,require_cross=True)
    if not cold['cache']['full_hits']:raise ValueError('Production stream exercised no complete text cache hits')
    targets=[];alternates=[];keys=set()
    for item in requests:
        n=int(torch.where(item['input_ids'][0]==_native(bundle.model).config.vision_start_token_id)[0][0])
        key=tuple(item['input_ids'][0,:n].tolist())
        if key in keys:continue
        match=next((x for x in requests if torch.equal(x['input_ids'][:,:n],item['input_ids'][:,:n])
            and not torch.equal(x['pixel_values'],item['pixel_values'])),None)
        if match is None:raise ValueError('Each target question needs alternate real images')
        targets.append(item);alternates.append(match);keys.add(key)
        if len(targets)==2:break
    resident=cache_structure.measure_stream(bundle,targets,alternates,paired,max_new_tokens=max_new_tokens)
    differences=[i for i,(a,b) in enumerate(zip(cold['answers'],reference['answers'])) if a!=b]
    trials = []
    for repeat in range(repeats):
        trial = {}
        for shared in ((False, True) if repeat % 2 == 0 else (True, False)):
            result = run(shared); assert_same(cold if shared else reference, result)
            trial['cached' if shared else 'full'] = result
        trials.append(trial)
    cached = sum(t['cached']['seconds'] for t in trials)
    full = sum(t['full']['seconds'] for t in trials)
    return dict(protocol='text_prefix_ordered_stream_v2', cold=cold, reference=reference, trials=trials,
                recomputed_split=recomputed,text_reuse_structure=structure,same_partition_equivalence=True,resident_text_stream=resident,
                full_output_difference_requests=differences,
                saving_fraction=1-cached/full, complete_output_equivalence=not differences,
                scope='preencoded ordered stream; includes misses, cloning, prefill and decode; full control may decode different tokens/length; excludes processor',
                model_dtype=str(next(bundle.model.parameters()).dtype), device=str(bundle.device),
                max_new_tokens=max_new_tokens, quality_certified=False)
