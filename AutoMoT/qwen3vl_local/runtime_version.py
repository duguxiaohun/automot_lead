"""Local content-addressed execution trees. Never rewrites experiment receipts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA='qwen_local_execution_v1'
SOURCE_SUFFIXES={'.py','.sh','.json','.yaml','.yml','.toml','.ini','.cfg','.jinja','.txt','.md','.css','.html'}
CODE_ROOTS=('qwen3vl_local','Automot','leaderboard','scenario_runner')
EXCLUDED={'.git','__pycache__','.pytest_cache','.venv','node_modules'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield


def inventory(app):
    """Copy source/config bytes, share data/weights via explicit asset links."""
    app=Path(app).resolve();files={};links={}
    for name in CODE_ROOTS:
        root=app/name
        if not root.is_dir():continue
        for folder,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in EXCLUDED)
            for d in list(dirs):
                p=Path(folder)/d
                if p.is_symlink():
                    # Source directory links would allow lazy imports to escape isolation.
                    raise ValueError('linked_source_directory_requires_real_directory: '+str(p))
            for filename in sorted(names):
                p=Path(folder)/filename;rel=str(p.relative_to(app))
                if p.suffix in ('.pyc','.pyo','.log','.lock'):continue
                if p.suffix in SOURCE_SUFFIXES or p.name in ('Makefile','Dockerfile'):
                    files[rel]=sha(p)
                else:links[rel]=str(p.resolve())
    # Scripts at application root may be imported/called by older launchers.
    for p in sorted(app.iterdir()):
        if p.name in CODE_ROOTS or p.name in EXCLUDED:continue
        if p.is_file() and p.suffix in SOURCE_SUFFIXES:files[p.name]=sha(p)
        else:links[p.name]=str(p.resolve())
    # Preserve existing relative ../lead map discovery without copying its data.
    siblings={p.name:str(p.resolve()) for p in app.parent.iterdir()
              if p.name in ('lead','CARLA_0915')}
    return dict(schema=SCHEMA,origin=str(app),files=files,assets=links,siblings=siblings)


def verify(directory):
    directory=Path(directory).resolve();record=json.loads((directory/'manifest.json').read_text())
    if record['schema']!=SCHEMA or directory.name!=digest(record):raise ValueError('invalid_execution_version_identity')
    tree=directory/'app'
    for name,value in record['files'].items():
        p=tree/name
        if p.is_symlink() or sha(p)!=value:raise ValueError('execution_snapshot_changed: '+name)
    # Detect additions as well as mutations: added .py changes lazy imports/fingerprints.
    actual=set()
    for name in CODE_ROOTS:
        for folder,dirs,names in os.walk(tree/name,followlinks=False):
            dirs[:]=[d for d in dirs if d not in EXCLUDED]
            for d in dirs:
                if (Path(folder)/d).is_symlink():raise ValueError('execution_source_link_added')
            for filename in names:
                p=Path(folder)/filename;rel=str(p.relative_to(tree))
                if p.suffix in SOURCE_SUFFIXES or p.name in ('Makefile','Dockerfile'):actual.add(rel)
    expected={p for p in record['files'] if Path(p).parts[0] in CODE_ROOTS}
    if actual!=expected:raise ValueError('execution_snapshot_members_changed')
    for name,target in record['assets'].items():
        p=tree/name
        if not p.is_symlink() or os.readlink(p)!=target:raise ValueError('execution_asset_link_changed: '+name)
    for name,target in record['siblings'].items():
        if not (directory/name).is_symlink() or os.readlink(directory/name)!=target:raise ValueError('execution_sibling_link_changed')
    return record


def prepare(app,cache=None):
    app=Path(app).resolve();cache=Path(cache or app/'checkpoints/runtime_versions').resolve()
    cache.mkdir(parents=True,exist_ok=True)
    with lock(cache/'.build.lock'):
        for attempt in range(3):
            record=inventory(app);version=digest(record);target=cache/version
            if target.exists():verify(target);return target
            pending=Path(tempfile.mkdtemp(prefix='.preparing-',dir=cache));tree=pending/'app';tree.mkdir()
            try:
                for name,expected in record['files'].items():
                    p=tree/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(app/name,p)
                    if sha(p)!=expected:raise RuntimeError('source_changed_during_snapshot')
                    p.chmod(0o555 if p.suffix=='.sh' else 0o444)
                for name,link in record['assets'].items():
                    p=tree/name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(link)
                for name,link in record['siblings'].items():(pending/name).symlink_to(link)
                if inventory(app)!=record:raise RuntimeError('source_changed_during_snapshot')
                write(pending/'manifest.json',record);pending.rename(target);verify(target)
                return target
            except RuntimeError:
                if attempt==2:raise
                print('[version] sources changed during capture; retrying before work starts',file=sys.stderr,flush=True)
            finally:
                if pending.exists():shutil.rmtree(pending)
    raise RuntimeError('snapshot_not_created')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--version',type=Path,help='reuse a verified local execution directory for the same run')
    p.add_argument('--cache',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args(argv);command=args.command
    if command and command[0]=='--':command=command[1:]
    app=args.app.resolve()
    # New ABC runs persist their version. Later pack/evaluate retries stay on it.
    if not args.version and command and command[0]=='qwen3vl_local/action_expert_ablation/geometry_abc/run.sh':
        run_dir=None
        for i,value in enumerate(command):
            if value=='--run-dir' and i+1<len(command):run_dir=Path(command[i+1])
            elif value.startswith('--run-dir='):run_dir=Path(value.split('=',1)[1])
        if run_dir is not None:
            if not run_dir.is_absolute():run_dir=app/run_dir
            proof=run_dir/'execution_version.json'
            if proof.is_file():
                saved=json.loads(proof.read_text());candidate=Path(saved['directory']).resolve()
                if not candidate.is_relative_to(app/'checkpoints/runtime_versions'):
                    raise ValueError('saved_execution_version_outside_local_cache')
                args.version=candidate
    directory=args.version.resolve() if args.version else prepare(app,args.cache)
    record=verify(directory);tree=directory/'app'
    if args.prepare_only:print(directory);return
    if not command:p.error('supply a relative qwen3vl_local script or -m module after --')
    if command[0]=='-m':
        if len(command)<2 or not command[1].startswith('qwen3vl_local.'):p.error('only qwen3vl_local modules supported')
        cmd=[sys.executable,'-B',*command]
    else:
        path=Path(command[0])
        if path.is_absolute():
            try:path=path.relative_to(app)
            except ValueError:p.error('script must belong to this application')
        if '..' in path.parts or not path.parts or path.parts[0]!='qwen3vl_local':p.error('script must be inside qwen3vl_local')
        target=tree/path
        if not target.is_file():p.error('missing script: '+str(path))
        cmd=(['bash',str(target)] if path.suffix=='.sh' else [sys.executable,'-B',str(target)])+command[1:]
    inherited=os.environ.get('PYTHONPATH','').split(os.pathsep)
    paths=[x for x in inherited if x and Path(x).resolve()!=app]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(tree),*paths]),PYTHONDONTWRITEBYTECODE='1',
             QWEN_EXECUTION_VERSION=str(directory),QWEN_EXECUTION_ORIGIN=record['origin'],PYTHON=sys.executable)
    # Legacy shells invoke `python` by name; inherit the same selected environment.
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    logdir=Path(record['origin'])/'checkpoints/runtime_executions';logdir.mkdir(parents=True,exist_ok=True)
    log=logdir/f'{time.time_ns()}_{os.getpid()}.json'
    receipt=dict(schema=SCHEMA,version=directory.name,snapshot=str(directory),command=cmd,pid=os.getpid(),status='running',
                 assets_shared=True,python=sys.executable)
    write(log,receipt)
    print(f'[version] {directory.name}\n[version] receipt={log}\n[version] source={tree}',flush=True)
    code=1
    try:
        process=subprocess.Popen(cmd,cwd=tree,env=env,start_new_session=True)
        previous={}
        def forward(signum,frame):
            if process.poll() is None:
                try:os.killpg(process.pid,signum)
                except ProcessLookupError:pass
        for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,forward)
        try:code=process.wait()
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
        if code<0:code=128-code
        try:verify(directory)
        except Exception:
            code=1
            raise
    finally:
        receipt.update(status='completed' if code==0 else 'nonzero_exit',exit_code=code);write(log,receipt)
    raise SystemExit(code)


if __name__=='__main__':main()
