from pathlib import Path
import tempfile,subprocess,sys,json,hashlib,zipfile,struct,random,os,shutil
TOOL=Path(__file__).resolve().parents[1]/'collect_bev_comparison.py'
def enc(x):return json.dumps(x,ensure_ascii=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f.read_bytes()) for f in p.rglob('*') if f.is_file()}
checks=[]
with tempfile.TemporaryDirectory(prefix='bev_collector_validation_') as tmp:
 app=Path(tmp);run=app/'checkpoints/action_expert_ablation/bev_only/run_20260928_211328';run.mkdir(parents=True)
 def write(name,obj):
  p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(enc(obj));return p
 data=app/'checkpoints/old_data';data.mkdir();idx=data/'test.jsonl';idx.write_bytes(b'{"scenario":"Example","run_id":"Town01_1","anchor":12}\n')
 bev=app/'checkpoints/bev/model.pth';bev.parent.mkdir();bev.write_bytes(b'not deserializable tensor weights')
 write('config.json',{'data_dir':'/other/server/AutoMoT/checkpoints/old_data','lead_bev_ckpt':'checkpoints/bev/model.pth','event_balance_source_identity':{'action_dataset_hashes':{'test':sha(idx.read_bytes())}}})
 write('condition_contract.json',{'identity_payload':{'bev':sha(bev.read_bytes())}});write('training_plan.json',{'epochs':7,'actual_step_limit':12719})
 write('eval_test/metrics.json',{'metrics':{'samples':513,'waypoint_ade_m':0.3},'split':'test','ema':True,'checkpoint_step':12719})
 write('training_audit/epochs/epoch_007.json',{'epoch':7,'train_metrics':{'loss':.0141}})
 for i in range(321):write(f'training_audit/windows/step_{i:08d}.json',{'optimizer_step':i,'metrics':{'loss':.1/(i+1)}})
 write('validation/epoch_007.json',{'samples':66478,'loss':.0177});write('performance/session.json',{'samples':813792,'training_seconds':14000})
 (run/'best.pt').write_bytes(b'Do not deserialize');(run/'latest.pt').symlink_to('best.pt')
 rng=random.Random(7);original=[]
 for i in range(513):
  coords={k:[[[struct.unpack('<f',struct.pack('<f',rng.uniform(-20,20)))[0] for _ in range(2)] for _ in range(10 if 'route' in k else 8)]] for k in ('pred_route','pred_waypoints','gt_route','gt_waypoints')}
  if i==0:coords['pred_route'][0][0][0]=1.00000000001 # float64 fallback must be exact.
  if i==1:coords['pred_waypoints'][0][0][0]=-0.0
  record={'sample':{'scenario':'Example','run_id':'Town01_1','anchor':i,'event_balance_bucket':'UE1'},'condition':{'qwen_prefills':0},'metrics':{'loss':.01,'waypoint_ade_m':.3},**coords}
  p=write(f'eval_test/cases/rank0_case{i:06d}.json',record);original.append((p,record))
 before=tree(run)
 def execute(tool,output,extra=()):
  result=subprocess.run([sys.executable,'-B',str(tool),'--run',str(run),'--output',str(app/output),*extra],cwd=app,text=True,capture_output=True)
  if result.returncode:raise AssertionError(result.stderr)
  package=next((app/output).glob('*/bev_comparison_review.zip'));return package,json.loads((package.parent/'receipt.json').read_text())
 p,r=execute(TOOL,'out');assert p.stat().st_size+(p.parent/'receipt.json').stat().st_size<=30_000_000
 assert r['coordinates_included'];assert r['case_sets']['old_bev']['complete'];assert tree(run)==before
 with zipfile.ZipFile(p) as z:
  manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
  for e in manifest['members']:b=z.read(e['path']);assert len(b)==e['bytes'] and sha(b)==e['sha256']
  rows=[json.loads(line) for line in z.read('old_bev/cases.jsonl').splitlines()];binary=z.read('old_bev/case_coordinates.bin');assert len(rows)==513
  for row,(source,expected) in zip(rows,original):
   assert row['source']['sha256']==sha(source.read_bytes())
   assert row['record']['sample']==expected['sample']
   for key,c in row['coordinates'].items():
    raw=binary[c['offset_bytes']:c['offset_bytes']+c['bytes']];values=struct.unpack('<'+str(c['count'])+('f' if c['dtype']=='<f4' else 'd'),raw)
    flat=[v for a in expected[key] for b in a for v in b];assert list(values)==flat
  assert rows[0]['coordinates']['pred_route']['dtype']=='<f8'
  assert len(z.read('old_bev/training_audit.jsonl').splitlines())==322
  m=json.loads(z.read('COLLECTION_MANIFEST.json'));assert m['external_resources']['current_test_index']['matches_training_record'];assert m['external_resources']['current_bev_file']['matches_training_record']
 checks+=['513 complete case identities, metrics and coordinate round trips','float64 fallback and negative-zero coordinate storage','all 321 windows retained, no old 200-window truncation','all archive member hashes and hard limit verified','run inputs byte-identical, no weight deserialization','relocated current-machine data index accepted only with recorded hash comparison']
 # Force the size fallback on the same actual collector logic; every metric row must remain.
 with zipfile.ZipFile(p) as z:
  coord_info=z.getinfo('old_bev/case_coordinates.bin');compressed_coordinates=coord_info.compress_size
 cap=p.stat().st_size-compressed_coordinates//2+64_000
 altered=app/'cap_collector.py';altered.write_text(TOOL.read_text().replace('LIMIT = 30_000_000','LIMIT = '+str(cap)))
 fp,fr=execute(altered,'fallback');assert not fr['coordinates_included']
 with zipfile.ZipFile(fp) as z:
  assert len(z.read('old_bev/cases.jsonl').splitlines())==513
  assert 'old_bev/case_coordinates.bin' not in z.namelist()
 checks.append('forced size fallback removes coordinates only; retains every case metric and identity')
 # Explicit same-machine ABC suite supplements its full training curves.
 abc=app/'abc_suite'
 for v in 'abc':
  d=abc/(v+'_bev_only');(d/'training_audit/epochs').mkdir(parents=True)
  (d/'config.json').write_bytes(enc({'variant':v}))
  (d/'metrics.json').write_bytes(enc({'all':{'cases':58173,'waypoint_ADE':.3}}))
  (d/'training_audit/epochs/epoch_007.json').write_bytes(enc({'epoch':7,'train_metrics':{'loss':.02}}))
 ap,ar=execute(TOOL,'withabc',['--abc-run',str(abc)])
 with zipfile.ZipFile(ap) as z:
  for v in 'abc':assert len(z.read(v+'_bev_only/training_audit.jsonl').splitlines())==1
 checks.append('explicit three-group ABC suite includes each full epoch record')
 # Optional current sources can drift without invalidating saved training evidence.
 source=app/'qwen3vl_local/action_prior/training_core.py';source.parent.mkdir(parents=True)
 source.write_text('# optional reference before synchronization\n')
 injected=app/'drifting_source.py'
 hook="\n_real_signature=signature\n_drift=0\ndef signature(p):\n global _drift\n value=_real_signature(p)\n if p.name=='training_core.py':\n  _drift+=1\n  return (*value,_drift)\n return value\n"
 injected.write_text(TOOL.read_text().replace("if __name__=='__main__':main()",hook+"if __name__=='__main__':main()"))
 dp,dr=execute(injected,'source_drift')
 with zipfile.ZipFile(dp) as z:
  m=json.loads(z.read('COLLECTION_MANIFEST.json'));entry=m['source_comparison']['qwen3vl_local/action_prior/training_core.py']
  assert entry['status']=='omitted_unstable_or_unavailable' and entry['attempts']==3
  assert entry['matches'] is None
  assert 'current_source_reference/qwen3vl_local/action_prior/training_core.py' not in z.namelist()
  assert len(z.read('old_bev/cases.jsonl').splitlines())==513
 assert any('training_core.py' in w for w in dr['warnings'])
 checks.append('persistent optional source drift omits reference after three attempts and publishes complete saved metrics')
 sp,sr=execute(TOOL,'skip_sources',['--no-current-source-reference'])
 with zipfile.ZipFile(sp) as z:
  m=json.loads(z.read('COLLECTION_MANIFEST.json'));assert all(v['status']=='omitted_by_request' for v in m['source_comparison'].values())
  assert not any(n.startswith('current_source_reference/') for n in z.namelist())
 checks.append('explicit source-reference skip retains original contracts and all case metrics')
 # A later update cannot change the already captured reference or invalidate publication.
 snapshot=source.read_bytes();later=app/'later_source.py'
 later.write_text(TOOL.read_text().replace("        put('collector.py',Path(__file__).read_bytes())", "        (app/'qwen3vl_local/action_prior/training_core.py').write_text('# changed after snapshot\\n')\n        put('collector.py',Path(__file__).read_bytes())"))
 lp,lr=execute(later,'later_source')
 with zipfile.ZipFile(lp) as z:assert z.read('current_source_reference/qwen3vl_local/action_prior/training_core.py')==snapshot
 checks.append('post-snapshot source update does not invalidate captured reference or run evidence')
 # Core results retain the original strict behavior.
 core=app/'drifting_core.py';core.write_text(TOOL.read_text().replace("if __name__=='__main__':main()",hook.replace("p.name=='training_core.py'","p.name=='config.json'")+"if __name__=='__main__':main()"))
 failed=subprocess.run([sys.executable,'-B',str(core),'--run',str(run),'--output',str(app/'core_drift')],cwd=app,capture_output=True,text=True)
 assert failed.returncode and 'input changed while reading' in failed.stderr
 assert not list((app/'core_drift').glob('*/bev_comparison_review.zip'))
 checks.append('core run configuration drift still rejects publication')
 # Real filesystem metadata-only change is retried before a stable snapshot is accepted.
 import importlib.util,stat
 spec=importlib.util.spec_from_file_location('collector_under_test',TOOL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original_read=Path.read_bytes;changed=[False]
 def ctime_once(path):
  value=original_read(path)
  if path==source and not changed[0]:
   changed[0]=True;path.chmod(stat.S_IMODE(path.stat().st_mode)^0o100)
  return value
 Path.read_bytes=ctime_once
 try:captured,info=module.optional_source_snapshot(source)
 finally:Path.read_bytes=original_read
 assert info['status']=='captured_snapshot' and info['attempts']==2 and captured==source.read_bytes()
 assert str(source) not in module.READ_SIGNATURES
 checks.append('actual ctime-only file change recovers via two identical stable reads without entering core receipt registry')
 # Missing optional cases is reported, never fabricated.
 shutil.rmtree(run/'eval_test/cases');mp,mr=execute(TOOL,'missing');assert any('no saved rank' in x for x in mr['missing']);checks.append('missing saved cases explicitly reported without inference')
 # Too-small core must not publish an oversized/partial result.
 tiny=app/'tiny.py';tiny.write_text(TOOL.read_text().replace('LIMIT = 30_000_000','LIMIT = 100'))
 failure=subprocess.run([sys.executable,'-B',str(tiny),'--run',str(run),'--output',str(app/'tiny')],cwd=app,capture_output=True,text=True)
 assert failure.returncode and not list((app/'tiny').glob('*/bev_comparison_review.zip'));checks.append('oversized core rejects publication rather than silently dropping metrics')
 # Explicit ABC suite is required to exist; never choose latest across servers.
 fail=subprocess.run([sys.executable,'-B',str(TOOL),'--run',str(run),'--abc-run',str(app/'absent'),'--output',str(app/'absentout')],cwd=app,capture_output=True,text=True)
 assert fail.returncode and 'ABC group missing' in fail.stderr;checks.append('missing explicitly requested ABC run rejected')
 print(json.dumps({'status':'passed','checks':checks,'fixture_cases':513,'synthetic_capacity_not_remote_acceptance':True,'script_sha256':sha(TOOL.read_bytes()),'initial_fixture_zip_bytes':p.stat().st_size},ensure_ascii=False,indent=2)+'\n')
