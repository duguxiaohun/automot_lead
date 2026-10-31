"""Paired native online streams; CPU-safe import, no labels or CARLA certification.

JSONL rows: {"images": [...], "context": {...}}. Images are already selected
in chronological order for the adapter's RGB mode. No raw prompt override.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import time
import torch
from PIL import Image
from . import online, text_prefix
from .question_schedule import QuestionSchedule
from qwen3vl_local.sft_new_loop_phase12.cache import _decode, _sync

CONTEXT_KEYS = {1: set(), 2: {'domain'}, 12: {'domain'},
    3: {'context_id','road_structure','context_detail','goal_xy','current_speed_mps'}}
PARITY_FIELDS = ('text','token_ids','stop_token_id','reached_token_limit','generated_tokens')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def read_requests(path, phase):
    path=Path(path).resolve()
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(rows,phase)
    for row in rows:
        row['images']=[str((path.parent/name).resolve()) for name in row['images']]
        for name in row['images']:
            if not Path(name).is_file():raise FileNotFoundError(name)
    return rows


def validate_rows(rows,phase):
    if phase not in CONTEXT_KEYS or not 2<=len(rows)<=4096:
        raise ValueError('Expected phase 1/2/3/12 and 2..4096 ordered input groups')
    for row in rows:
        if set(row)!={'images','context'} or not isinstance(row['context'],dict):
            raise ValueError('Use only images/context; labels and raw prompts are forbidden')
        if set(row['context'])-CONTEXT_KEYS[phase]:raise ValueError('Unsupported online context fields')
        if (not isinstance(row['images'],list) or len(row['images']) not in (2,4)
                or any(not isinstance(name,str) or not name for name in row['images'])):
            raise ValueError('Expected 2 or 4 selected image paths')
        if phase in (2,12) and 'domain' not in row['context']:raise ValueError('Missing question domain')
        if phase==3 and not {'context_id','road_structure'}<=set(row['context']):
            raise ValueError('Phase3 requires current context_id and road_structure')
        digest(row)  # reject NaN/Infinity before starting a stream


def quantiles(values):
    """Nearest-rank quantiles, with no trimming of cold starts or switches."""
    values=sorted(values)
    return dict(count=len(values),total_seconds=sum(values),
        p50_seconds=values[math.ceil(.5*len(values))-1] if values else None,
        p95_seconds=values[math.ceil(.95*len(values))-1] if values else None)


def summarize(run,deadline_ms=None):
    frames=run['frames'];queries=[q for frame in frames for q in frame['queries']]
    return dict(frames=quantiles([f['seconds'] for f in frames]),
        model=quantiles([q['model_seconds'] for q in queries]),
        image_io_decode_seconds=sum(f['image_io_decode_seconds'] for f in frames),
        rendering_seconds=sum(f['rendering_seconds'] for f in frames),
        encoding_seconds=sum(q['encoding_seconds'] for q in queries),
        input_verification_seconds=sum(q['input_verification_seconds'] for q in queries),
        cache_classes={kind:quantiles([q['model_seconds'] for q in queries if q['cache_class']==kind])
                       for kind in ('full_hit','partial_hit','miss','disabled')},
        frame_groups={kind:quantiles([f['seconds'] for f in frames if
                     (f['index']==0 if kind=='first' else f[kind])])
                     for kind in ('first','window_changed','context_changed')},
        deadline_ms=deadline_ms,
        deadline_exceeded=sum(f['seconds']*1000>deadline_ms for f in frames) if deadline_ms is not None else None,
        generation_calls=len(queries),generated_tokens=sum(q['answer']['generated_tokens'] for q in queries))


def assert_same(left,right,*,require_outputs=True):
    differences=[]
    if len(left['frames'])!=len(right['frames']):raise ValueError('Input stream length changed')
    for i,(a,b) in enumerate(zip(left['frames'],right['frames'])):
        for field in ('schedule','images_sha256','context_sha256'):
            if a[field]!=b[field]:raise ValueError(f'Paired stream input/policy mismatch: frame {i}, {field}')
        if len(a['queries'])!=len(b['queries']):raise ValueError('Generation-call count differs')
        for number,(x,y) in enumerate(zip(a['queries'],b['queries'])):
            if x['input_sha256']!=y['input_sha256']:raise ValueError(f'Encoded input differs: frame {i}, query {number}')
            if any(x['answer'][k]!=y['answer'][k] for k in PARITY_FIELDS):
                differences.append(dict(frame=i,query=number))
                if require_outputs:raise ValueError(f'Output token/EOS/text mismatch: frame {i}, query {number}')
    return differences


def _input_sha(inputs):
    value=hashlib.sha256()
    for name,tensor in sorted(inputs.items()):
        if not isinstance(tensor,torch.Tensor):raise ValueError('Expected tensor processor outputs')
        value.update(str((name,str(tensor.dtype),tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def _prepared(bundle,phase,images,context,schedule):
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import infer
        cfg=infer.bound_online_contract(bundle)
        request=infer.prepare_request(bundle,images,cfg['mode'],context['domain'],schedule=schedule)
        return request['messages'],request['schedule']
    request=online.prepare_request(bundle,phase,images,schedule=schedule,**context)
    return [request['messages']],request['schedule']


@torch.no_grad()
def benchmark(bundle,phase,rows,*,window_frames=32,slots=1,seed='benchmark',rank=0,
              max_new_tokens=160,repeats=2,deadline_ms=None,on_trial=None,on_run=None):
    """Each arm replays identical original frames from an empty cache/schedule.

    Text cache and question schedule are private to this benchmark; production
    bundle banks/counters are never warmed. Timing includes audit fingerprinting,
    which is also separately reported. It is not CARLA/controller wall latency.
    """
    validate_rows(rows,phase)
    if not text_prefix.cache_enabled() or bundle.model.training:raise ValueError('Requires enabled text cache and eval model')
    if type(repeats) is not int or repeats<2 or repeats%2 or not 0<max_new_tokens<=512:
        raise ValueError('Use an even number of repeats >=2 and 1..512 generated tokens')
    if deadline_ms is not None and (not math.isfinite(deadline_ms) or deadline_ms<=0):raise ValueError('Invalid deadline')
    if phase==12 and slots!=1:raise ValueError('Shared two-task stream already uses two text entries')
    QuestionSchedule(window_frames,slots,seed,rank)
    if phase!=12:online.bound_contract(bundle,phase)
    else:
        if not hasattr(bundle,'fusion_config'):raise ValueError('Load the verified shared online bundle first')
    weights=text_prefix._weight_identity(bundle.model)
    source=getattr(bundle,'_text_prefix_cache',None) or text_prefix.TextPrefixCache()
    limits={k:getattr(source,k) for k in ('max_entries','max_bytes','max_tokens')}
    new_bank=lambda:text_prefix.TextPrefixCache(**limits)

    def run(cached,label):
        if text_prefix._weight_identity(bundle.model)!=weights:raise ValueError('Weights changed between trials')
        bank=new_bank();schedule=QuestionSchedule(window_frames,slots,seed,rank)
        frames=[];previous=None;structures=[]
        _sync(bundle.device);start=time.perf_counter()
        for number,row in enumerate(rows):
            _sync(bundle.device);begin=time.perf_counter();images=[];image_sha=[];queries=[];ticket=None
            try:
                for name in row['images']:
                    data=Path(name).read_bytes();image_sha.append(hashlib.sha256(data).hexdigest())
                    with Image.open(io.BytesIO(data)) as source:images.append(source.convert('RGB'))
                io_seconds=time.perf_counter()-begin;render_start=time.perf_counter()
                messages,ticket=_prepared(bundle,phase,images,row['context'],schedule)
                rendering_seconds=time.perf_counter()-render_start;queries=[]
                for message in messages:
                    encoded_start=time.perf_counter()
                    chat=bundle.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
                    cpu=bundle.processor(text=[chat],images=images,return_tensors='pt',padding=phase!=12)
                    if cpu['input_ids'].shape[1]>8192:raise ValueError('Prompt exceeds native 8192-token context')
                    encoding_seconds=time.perf_counter()-encoded_start
                    check_start=time.perf_counter();input_sha=_input_sha(cpu)
                    verification_seconds=time.perf_counter()-check_start
                    transfer_start=time.perf_counter()
                    inputs={k:v.to(bundle.device) for k,v in cpu.items()}
                    _sync(bundle.device);encoding_seconds+=time.perf_counter()-transfer_start
                    if cached=='fresh':
                        bank=new_bank()
                    model_start=time.perf_counter()
                    out=bank.prefill(bundle.model,inputs) if cached else bundle.model(**inputs,use_cache=True,return_dict=True)
                    answer=_decode(bundle,inputs,out,max_new_tokens);del out
                    _sync(bundle.device);model_seconds=time.perf_counter()-model_start
                    state=dict(bank.last) if cached else None
                    kind=('full_hit' if state['full_hit'] else 'partial_hit' if state['hit'] else 'miss') if state else 'disabled'
                    queries.append(dict(input_sha256=input_sha,input_tokens=inputs['input_ids'].shape[1],
                        encoding_seconds=encoding_seconds,input_verification_seconds=verification_seconds,
                        model_seconds=model_seconds,answer=answer,cache=state,cache_class=kind))
                    del inputs,cpu
                current=digest(row['context'])
                for image in images:image.close()
                images=[]
                frames.append(dict(index=number,schedule=ticket,images_sha256=image_sha,context_sha256=current,
                    window_changed=number>0 and ticket['window']!=frames[-1]['schedule']['window'],
                    context_changed=number>0 and current!=previous,image_io_decode_seconds=io_seconds,
                    rendering_seconds=rendering_seconds,seconds=time.perf_counter()-begin,queries=queries))
                previous=current
            except BaseException as exc:
                if on_run:on_run(label,dict(status='failed',error=f'{type(exc).__name__}: {exc}',
                    frames=frames,failed_frame=number,partial_queries=queries,
                    partial_images_sha256=image_sha,partial_schedule=ticket,cache=bank.report() if cached else None))
                raise
            finally:
                for image in images:image.close()
        _sync(bundle.device)
        result=dict(seconds=time.perf_counter()-start,frames=frames,cache=bank.report() if cached else None,
                    text_reuse_structure=structures)
        if text_prefix._weight_identity(bundle.model)!=weights:raise ValueError('Weights changed during a trial')
        result['summary']=summarize(result,deadline_ms)
        return result

    def recorded_run(name,cached):
        result=run(cached,name)
        # Persist both raw arms BEFORE equality checks, including a failing pair.
        # This callback runs outside all measurement clocks.
        if on_run:on_run(name,result)
        return result
    cold=recorded_run('cold_cached',True);reference=recorded_run('cold_full',False)
    differences=assert_same(cold,reference,require_outputs=False)
    if not cold['cache']['full_hits']:raise ValueError('Production stream exercised no complete text cache hits')
    recomputed=recorded_run('recomputed_split','fresh');assert_same(cold,recomputed)
    resident=resident_probe(bundle,phase,rows,window_frames=window_frames,slots=slots,seed=seed,rank=rank,
                            max_new_tokens=max_new_tokens,on_evidence=(lambda r:on_run('resident_text_stream',r)) if on_run else None)
    trials=[]
    for repeat in range(repeats):
        order=('full','cached') if repeat%2==0 else ('cached','full')
        trial=dict(order=list(order))
        for name in order:
            trial[name]=recorded_run(f'trial_{repeat:02d}_{name}',name=='cached');assert_same(cold if name=='cached' else reference,trial[name])
        trials.append(trial)
        if on_trial:on_trial(repeat,trial)
    cache_s=sum(t['cached']['seconds'] for t in trials);full_s=sum(t['full']['seconds'] for t in trials)
    cache_model=sum(t['cached']['summary']['model']['total_seconds'] for t in trials)
    full_model=sum(t['full']['summary']['model']['total_seconds'] for t in trials)
    return dict(protocol='native_online_stream_v3',phase=phase,cold_cached=cold,cold_full=reference,trials=trials,
        summary={name:summarize(dict(frames=[f for t in trials for f in t[name]['frames']]),deadline_ms)
                 for name in ('cached','full')},
        saving_fraction=1-cache_s/full_s,model_saving_fraction=1-cache_model/full_model,
        complete_output_equivalence=not differences,full_output_differences=differences,
        recomputed_split=recomputed,same_partition_equivalence=True,resident_text_stream=resident,
        quality_certified=False,carla_certified=False,
        policy=dict(window_frames=window_frames,slots=slots,seed=seed,rank=rank,max_new_tokens=max_new_tokens,
                    generation_calls_per_frame=2 if phase==12 else 1,prior_engine_rechecks=False,
                    equal_forward_call_count_required=False,
                    forward_call_note='cold cache builds and decode invoke additional forwards; equality is required for generation requests'),
        scope='native phase inputs + image IO/decode + processor/transfer + model + audit hashing; excludes model load, PriorEngine rechecks, batch/reuse_rgb and CARLA controller',
        timing=dict(quantile='nearest_rank',cache_reset='every arm starts empty',
                    cold_means='text cache cold, not OS disk or GPU hardware cold',
                    primary='all requests in alternating replay trials, including cache builds and switches',
                    clone_and_invalidation_included=True,hashing_included=True,
                    output_work_note='full control may decode different tokens/length; savings are not pure equal-output KV savings'),
        device=str(bundle.device),dtype=str(next(bundle.model.parameters()).dtype))



def resident_probe(bundle,phase,rows,*,window_frames,slots,seed,rank,max_new_tokens,on_evidence=None):
    """Hold native questions/context fixed only in this diagnostic; use two real RGB groups."""
    from qwen3vl_local.sft_new_loop_phase12 import cache as paired,cache_structure
    def sha_images(row):return [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['images']]
    first=rows[0];first_sha=sha_images(first)
    other=next((row for row in rows[1:] if sha_images(row)!=first_sha),None)
    if other is None:raise ValueError('Resident probe requires another real image group')
    images=[];alternate=[]
    try:
        for path in first['images']:
            with Image.open(path) as im:images.append(im.convert('RGB'))
        for path in other['images']:
            with Image.open(path) as im:alternate.append(im.convert('RGB'))
        schedule=QuestionSchedule(window_frames,slots,seed,rank)
        messages,_=_prepared(bundle,phase,images,first['context'],schedule)
        if phase!=12 and slots==2:
            extra,_=_prepared(bundle,phase,images,first['context'],schedule);messages+=extra
        original=[];changed=[]
        for message in messages:
            replacement=[];index=0
            for turn in message:
                content=turn['content']
                if isinstance(content,list):
                    parts=[]
                    for part in content:
                        if part.get('type')=='image':
                            parts.append(dict(part,image=alternate[index]));index+=1
                        else:parts.append(dict(part))
                    content=parts
                replacement.append(dict(turn,content=content))
            if index!=len(alternate):raise ValueError('Native message/image count mismatch')
            for msg,ims,target in ((message,images,original),(replacement,alternate,changed)):
                chat=bundle.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                encoded=bundle.processor(text=[chat],images=ims,return_tensors='pt',padding=False)
                target.append({k:v.to(bundle.device) for k,v in encoded.items()})
        # Two schedule slots may render the same permutation (especially two-key P2).
        # Count actual distinct complete prefixes; never claim two resident entries for one.
        selected=[];seen=set()
        for index,item in enumerate(original):
            n=int(torch.where(item['input_ids'][0]==text_prefix._native(bundle.model).config.vision_start_token_id)[0][0])
            key=tuple(item['input_ids'][0,:n].tolist())
            if key not in seen:selected.append(index);seen.add(key)
        if phase==12 and len(selected)!=2:raise ValueError('Shared stream requires two distinct task prefixes')
        def save(report):
            report['scheduled_question_calls']=len(messages)
            if on_evidence:on_evidence(report)
        return cache_structure.measure_stream(bundle,[original[i] for i in selected],[changed[i] for i in selected],
            paired,max_new_tokens=max_new_tokens,on_evidence=save)
    finally:
        for image in images+alternate:image.close()


def check_plan(args,rows):
    """Validate native rendering and frame counts on CPU before model loading."""
    from types import SimpleNamespace
    from .online_contract import bind_prompt_contract
    if not text_prefix.enabled():raise ValueError('Set AUTOMOT_TEXT_PREFIX=1')
    if args.phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import CONTRACT, prompts, infer
        cfg=json.loads((args.adapter_dir/'fusion_config.json').read_text())
        if cfg['contract']!=CONTRACT or cfg['prompt']!=prompts.identity(cfg['mode']):
            raise ValueError('Shared prompt/history contract mismatch')
        bundle=infer.bind_online_contract(SimpleNamespace(model=None),cfg)
    else:
        cfg=json.loads((args.adapter_dir/f'sft_new_loop_phase{args.phase}_adapter_config.json').read_text())
        bundle=SimpleNamespace(model=None);bind_prompt_contract(bundle,args.phase,cfg)
    if args.phase==12 and args.slots!=1:raise ValueError('Shared stream requires slots=1')
    scheduler=QuestionSchedule(args.window_frames,args.slots,args.seed,args.rank)
    for row in rows:_prepared(bundle,args.phase,[None]*len(row['images']),row['context'],scheduler)
    for name in dict.fromkeys(name for row in rows for name in row['images']):
        # Reject broken images before selecting a GPU or loading several GB of weights.
        with Image.open(name) as image:image.verify()
        with Image.open(name) as image:image.load()
    return cfg


def file_bindings(args,package):
    from .text_prefix_benchmark import sha
    return dict(requests_sha256=sha(args.requests),
        adapter_sha256={p.name:sha(p) for p in args.adapter_dir.iterdir() if p.is_file() and
            (p.suffix in ('.safetensors','.bin') or p.name.endswith('config.json') or p.name.startswith('qwen35_'))},
        source_sha256={str(p.relative_to(package)):sha(p) for p in package.rglob('*.py') if p.is_file()})


def verify_file_bindings(args,package,expected):
    current=file_bindings(args,package)
    if current!=expected:raise ValueError('Requests/adapter/source files changed during benchmark')


def worker(args):
    from qwen3vl_local.sft_new_loop_phase1.run_integrity import write, environment
    output=args.output
    if (output/'report.json').exists():raise FileExistsError(output/'report.json')
    report=dict(status='running',quality_certified=False,carla_certified=False,optimizer_updates=0)
    write(output/'report.json',report)
    try:
        package=Path(__file__).resolve().parents[1]
        bindings=file_bindings(args,package)
        report.update(bindings)
        rows=read_requests(args.requests,args.phase);cfg=check_plan(args,rows)
        verify_file_bindings(args,package,bindings)
        report.update(requests_file=str(args.requests),requests=rows,
            model_dir=str(args.model_dir),adapter_dir=str(args.adapter_dir),adapter_config=cfg,
            source_files_verified_after_load=False,source_files_verified_after_measurement=False)
        if args.phase==12:
            from qwen3vl_local.sft_new_loop_phase12.infer import load_online_bundle
            bundle=load_online_bundle(args.model_dir,args.adapter_dir,device=args.device,
                                      dtype=torch.bfloat16)
        else:
            bundle=online.load_online_bundle(args.model_dir,args.adapter_dir,args.phase,device=args.device,
                                            dtype=torch.bfloat16)
        report.update(environment=environment(),base_asset_sha256=getattr(bundle.model.config,'local_asset_sha256',None))
        verify_file_bindings(args,package,bindings);report['source_files_verified_after_load']=True
        write(output/'report.json',report)
        def completed(number,trial):
            write(output/f'trial_{number:02d}.json',trial)
            print(f'[online-benchmark] paired trial {number+1}/{args.repeats} passed',flush=True)
        report['benchmark']=benchmark(bundle,args.phase,rows,window_frames=args.window_frames,slots=args.slots,
            seed=args.seed,rank=args.rank,max_new_tokens=args.max_new_tokens,repeats=args.repeats,
            deadline_ms=args.deadline_ms,on_trial=completed,
            on_run=lambda name,value:write(output/(name+'.json'),value))
        verify_file_bindings(args,package,bindings);report['source_files_verified_after_measurement']=True
        report['status']='text_reuse_checked_full_control_recorded_quality_pending'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:write(output/'report.json',report)


def main():
    import argparse,os,subprocess,sys
    from datetime import datetime
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',type=int,choices=(1,2,3,12),required=True)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--adapter-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--window-frames',type=int,default=32)
    parser.add_argument('--slots',type=int,choices=(1,2),default=1)
    parser.add_argument('--seed',default='benchmark');parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--max-new-tokens',type=int,default=160)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--deadline-ms',type=float)
    parser.add_argument('--timeout',type=float,default=7200,help='Worker timeout in seconds')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0',
                        help='Execution device; both use the BF16 production contract. CPU is for small-model checks.')
    parser.add_argument('--check-only',action='store_true',help='CPU-only request/prompt validation; no model load')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    for name in ('requests','model_dir','adapter_dir','output'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if args.repeats<2 or args.repeats%2 or not 0<args.max_new_tokens<=512:parser.error('Use even repeats >=2 and 1..512 tokens')
    if args.deadline_ms is not None and (not math.isfinite(args.deadline_ms) or args.deadline_ms<=0):parser.error('Invalid deadline')
    if not math.isfinite(args.timeout) or args.timeout<=0:parser.error('Invalid timeout')
    if args.worker:
        if args.output is None:parser.error('worker requires output')
        return worker(args)
    rows=read_requests(args.requests,args.phase);check_plan(args,rows)
    if args.check_only:
        print(json.dumps(dict(status='request_prompt_plan_valid',groups=len(rows),weights_validated=False,
                              inference_executed=False,quality_certified=False)))
        return
    selected=''
    if args.device!='cpu':
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        idle=[r.split(',')[0].strip() for r in raw.splitlines() if float(r.split(',')[1])<256 and float(r.split(',')[2])<=5]
        selected=os.environ.get('GPU_IDS') or (idle[0] if idle else '')
        if selected not in idle:raise ValueError('One idle GPU required; GPU_IDS may pin one idle physical ID')
    args.output=args.output or Path('checkpoints/online_prefix_benchmark')/f'run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m',__spec__.name,'--worker']
    for name in ('phase','requests','model_dir','adapter_dir','output','window_frames','slots','seed','rank','max_new_tokens','repeats','device','deadline_ms'):
        value=getattr(args,name)
        if value is not None:command.extend(['--'+name.replace('_','-'),str(value)])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':selected,'GPU_IDS':selected,'PYTHONUNBUFFERED':'1'}
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(name,None)
    data_stage.run(command,args.output/'worker.json',timeout=args.timeout,env=env)
    print('[ONLINE_PREFIX_REPORT] '+str(args.output/'report.json'),flush=True)


if __name__=='__main__':main()
