"""Phase3 新输入布局的索引视图：原索引逐字节复制，只新建 manifest。

原标签、split、顺序、历史来源和旧 manifest 均不修改。两服务器各自在本机生成。
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import text_prefix


def sha(path):
    """流式绑定索引内容。"""
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def hashes():
    """当前原生 Phase3 所有输入模式的指纹。"""
    from qwen3vl_local.sft_new_loop_phase3.prompts import action_prompt_sha256
    return {output:{mode:action_prompt_sha256(audit=False,history_rgb_mode=mode,action_output_mode=output)
                    for mode in ('4rgb','2rgb_endpoints')} for output in ('binary','choice')}


def verify(index, manifest):
    """新视图必须保留可回读的父索引/manifest；不能伪造或改写旧回执。"""
    view=manifest.get('text_prefix_view')
    if not view:return
    if view.get('contract')!=text_prefix.CONTRACT:raise ValueError('Unknown input-layout view')
    source=Path(view['source_index']);source_manifest=source.with_name('manifest.json')
    if (sha(source)!=view['index_sha256'] or sha(index)!=view['index_sha256'] or
            sha(source_manifest)!=view['source_manifest_sha256']):
        raise ValueError('Text-prefix source/index/manifest changed')
    parent=json.loads(source_manifest.read_text())
    restored=copy.deepcopy(manifest);restored.pop('text_prefix_view')
    restored['prompt_contract']=view['parent_prompt_contract']
    if restored!=parent:raise ValueError('Text-prefix view changed more than input-layout metadata')
    expected={out:{mode:hashlib.sha256((text_prefix.CONTRACT+':'+digest).encode()).hexdigest()
                   for mode,digest in modes.items()}
              for out,modes in parent['prompt_contract']['production_prompt_sha256'].items()}
    if manifest['prompt_contract']['production_prompt_sha256']!=expected:
        raise ValueError('Text-prefix view is not a pure input-layout derivation')


def prepare(index, output):
    """返回新视图路径；已有正确视图只读校验并原路返回。"""
    if not text_prefix.enabled():raise ValueError('Text-prefix data view requires AUTOMOT_TEXT_PREFIX=1')
    index=Path(index).resolve(strict=True);output=Path(output).resolve()
    manifest_path=index.with_name('manifest.json');parent=json.loads(manifest_path.read_text())
    expected=hashes()
    if parent.get('text_prefix_view'):
        verify(index,parent)
        if parent['prompt_contract']['production_prompt_sha256']!=expected:
            raise ValueError('Existing text-prefix view belongs to another prompt')
        return index
    contract=parent['prompt_contract']
    derived={out:{mode:text_prefix.bind_hash(digest) for mode,digest in modes.items()}
             for out,modes in contract['production_prompt_sha256'].items()}
    if derived!=expected:
        raise ValueError('Source prompt differs beyond layout; cannot relabel its manifest')
    from qwen3vl_local.sft_new_loop_phase3.prompts import PROMPT_NAME
    if contract.get('prompt_name')!=PROMPT_NAME:raise ValueError('Source prompt_name mismatch')
    before=sha(index);manifest_sha=sha(manifest_path)
    output.mkdir(parents=True,exist_ok=False)
    target=output/'frame_index.jsonl';shutil.copyfile(index,target)
    result=copy.deepcopy(parent)
    result['prompt_contract']['production_prompt_sha256']=expected
    result['text_prefix_view']=dict(contract=text_prefix.CONTRACT,source_index=str(index),
        index_sha256=before,source_manifest_sha256=manifest_sha,parent_prompt_contract=contract,
        labels_and_splits='byte_identical',purpose='new training input layout only')
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    verify(target,result)
    return target
