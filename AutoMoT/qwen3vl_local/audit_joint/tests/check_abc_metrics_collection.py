"""Standard-library regression for the CPU-only ABC metrics supplement collector."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import redirect_stdout
from unittest.mock import patch
import hashlib
import importlib.util
import io
import json
import random
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('abc_collect', Path(__file__).resolve().parents[1]/'collect_abc_metrics.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root/'runs'
        for v in 'abc':
            p = self.run/(v+'_bev_only')
            p.mkdir(parents=True)
            (p/'config.json').write_text('{"batch":1}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self, name='out', limit=c.LIMIT):
        with redirect_stdout(io.StringIO()):
            return c.collect(self.run, self.root/name, limit)

    def contents(self, result):
        with ZipFile(result['package']) as z:
            self.assertIsNone(z.testzip())
            m = json.loads(z.read('MANIFEST.json'))
            payload = z.read('evidence.bin')
        self.assertEqual(hashlib.sha256(payload).hexdigest(), m['storage']['sha256'])
        self.assertEqual(len(payload), m['storage']['bytes'])
        result = {}
        for name, r in m['sources'].items():
            b = payload[r['offset']:r['offset']+r['bytes']]
            self.assertEqual(len(b),r['bytes'])
            self.assertEqual(hashlib.sha256(b).hexdigest(),r['sha256'])
            result[name] = b
        return m,result

    def test_full_history_exact_bytes_and_dedup(self):
        original = b'{"route_fm_mse":0.123456789012345678901,"waypoint_fm_mse":1e-42}\n'
        for v in 'abc':
            p=self.run/(v+'_bev_only')
            (p/'training_audit/windows').mkdir(parents=True)
            for i in range(321):
                (p/'training_audit/windows'/f'step_{i:08d}.json').write_bytes(original)
            (p/'test/cases').mkdir(parents=True)
            (p/'test/cases/ignored.json').write_text('{}')
            (p/'best.pt').write_bytes(b'not a model')
            with ZipFile(p/'training_audit.zip','w') as z:
                z.writestr('audit/step.json',original)
                z.writestr('unique.json',b'{"only_in_archive": true}')
        before={str(p):c.sha_file(p) for p in self.run.rglob('*') if p.is_file()}
        result=self.collect();m,restored=self.contents(result)
        self.assertEqual(len(restored),3*(1+321+2))
        for name,b in restored.items():
            if '/windows/' in name or name.endswith('::audit/step.json'):
                self.assertEqual(b,original)
        self.assertLess(m['core_unique_bytes'],500)
        self.assertFalse(any('best.pt' in n or '/cases/' in n for n in restored))
        self.assertEqual(before,{str(p):c.sha_file(p) for p in self.run.rglob('*') if p.is_file()})

    def test_archive_fallback_and_weight_exclusion(self):
        p=self.run/'a_bev_only/training_audit.zip'
        with ZipFile(p,'w') as z:
            z.writestr('audit/epoch_007.json','{"route_fm_mse":0.125}')
            z.writestr('best.pt',b'not a model')
        m,files=self.contents(self.collect())
        self.assertIn('a_bev_only/training_audit.zip::audit/epoch_007.json',files)
        self.assertFalse(any(n.endswith('best.pt') for n in files))
        self.assertEqual(m['archives']['a_bev_only/training_audit.zip']['sha256'],c.sha_file(p))
        self.assertEqual(m['archives']['a_bev_only/training_audit.zip']['omitted'][0]['member'],'best.pt')

    def test_tb_included_and_fallback(self):
        p=self.run/'a_bev_only/tb';p.mkdir()
        original=random.Random(42).randbytes(100000)
        (p/'events.out.tfevents.test').write_bytes(original)
        m,files=self.contents(self.collect('with_tb'))
        self.assertTrue(m['tensorboard_included'])
        self.assertEqual(files['a_bev_only/tb/events.out.tfevents.test'],original)
        m,files=self.contents(self.collect('without_tb',8000))
        self.assertFalse(m['tensorboard_included'])
        self.assertEqual(len(files),3)
        self.assertEqual(len(m['omitted_tensorboard']),1)

    def test_core_over_limit_keeps_inventory(self):
        with self.assertRaisesRegex(RuntimeError,'体积清单'):
            self.collect(limit=1)
        self.assertFalse((self.root/'out/abc_metrics_supplement.zip').exists())
        self.assertTrue((self.root/'out/collection_inventory.json').is_file())

    def test_missing_group_and_overlapping_output(self):
        with self.assertRaises(ValueError):c.collect(self.run,self.run/'output')
        (self.run/'c_bev_only/config.json').unlink()
        with self.assertRaises(ValueError):self.collect()

    def test_live_file_rejected(self):
        original=c.signature
        target=self.run/'a_bev_only/config.json'
        reads=[0]
        def changed(p):
            value=original(p)
            if p==target:
                reads[0]+=1
                if reads[0]>1:value=value[:-1]+(value[-1]+1,)
            return value
        with patch.object(c,'signature',changed):
            with self.assertRaisesRegex(RuntimeError,'文件正在变化'):self.collect()

    def test_duplicate_archive_names_rejected(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with ZipFile(self.run/'a_bev_only/training_audit.zip','w') as z:
                z.writestr('metrics.json','{}');z.writestr('metrics.json','{}')
        with self.assertRaisesRegex(ValueError,'duplicate ZIP'):self.collect()

    def test_two_server_layouts(self):
        for layout in ['home/cruser1/lda/AutoMoT','mnt/workspace/cee3sgh/AutoMoT']:
            app=self.root/layout;(app/'qwen3vl_local').mkdir(parents=True)
            # Absolute input here deliberately belongs to this fixture, not another server.
            with patch.dict('os.environ',{'ABC_AUDIT_RUN':str(self.run)}),patch.object(Path,'cwd',return_value=app):
                with redirect_stdout(io.StringIO()):c.main([])
            result=list((app/'checkpoints/audit_exports').glob('*/*/*.zip'))
            self.assertEqual(len(result),3)
            self.assertEqual({p.name for p in result},{v+'_metrics_supplement.zip' for v in 'abc'})


    def test_split_independent_caps_and_exact_source_partition(self):
        expected = {}
        for v in 'abc':
            p = self.run/(v+'_bev_only')/'metrics.json'
            b = json.dumps({'route_fm_mse':0.123, 'diagnostic':random.Random(ord(v)).randbytes(4000).hex()}).encode()
            p.write_bytes(b);expected[v] = b
        with self.assertRaises(RuntimeError):
            self.collect('combined_too_large',limit=8000)
        with redirect_stdout(io.StringIO()):
            receipt=c.collect_split(self.run,self.root/'separate',limit=8000)
        self.assertEqual(receipt['status'],'completed')
        self.assertEqual(set(receipt['groups']),set('abc'))
        for v,result in receipt['groups'].items():
            self.assertLessEqual(result['bytes'],8000)
            m,files=self.contents(result)
            self.assertEqual(m['variants'],[v])
            self.assertTrue(all(name.startswith(v+'_bev_only/') for name in files))
            self.assertEqual(files[v+'_bev_only/metrics.json'],expected[v])
            self.assertEqual(Path(result['package']).name,v+'_metrics_supplement.zip')

    def test_one_oversized_group_does_not_block_remaining(self):
        p=self.run/'a_bev_only/metrics.json'
        p.write_text(json.dumps({'diagnostic':random.Random(90).randbytes(30000).hex()}))
        with redirect_stdout(io.StringIO()):
            receipt=c.collect_split(self.run,self.root/'partial',limit=8000)
        self.assertEqual(receipt['status'],'partial')
        self.assertEqual(set(receipt['groups']),set('bc'))
        self.assertEqual(set(receipt['errors']),{'a'})
        self.assertTrue((self.root/'partial/a/collection_inventory.json').exists())
        self.assertFalse((self.root/'partial/a/a_metrics_supplement.zip').exists())
        for result in receipt['groups'].values():self.contents(result)
        self.assertEqual(json.loads((self.root/'partial/delivery.json').read_text()),receipt)

    def test_missing_group_and_single_group_retry(self):
        (self.run/'b_bev_only/config.json').unlink()
        with redirect_stdout(io.StringIO()):
            receipt=c.collect_split(self.run,self.root/'missing')
        self.assertEqual(set(receipt['groups']),set('ac'))
        self.assertEqual(set(receipt['errors']),{'b'})
        with redirect_stdout(io.StringIO()):
            result=c.collect_split(self.run,self.root/'only_c',variants='c')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(set(result['groups']),{'c'})

    def test_invalid_group_list(self):
        for variants in ('', 'aa', 'd'):
            with self.assertRaises(ValueError):c.collect_split(self.run,self.root/'invalid',variants=variants)

    def test_real_30mb_previous_core_failure_regression(self):
        # Large repeated contracts/metadata: old per-file ZIP plus opaque nested
        # ZIPs exceeds the real 30MB cap; v2 preserves every original member.
        shared=random.Random(123).randbytes(3500000).hex()
        expected={}
        for v in 'abc':
            p=self.run/(v+'_bev_only');(p/'training_audit/windows').mkdir(parents=True)
            with ZipFile(p/'training_audit.zip','w',ZIP_DEFLATED) as z:
                for i in (1,2):
                    b=('{"optimizer_step":%d,"route_fm_mse":0.25,"binding":"%s"}'%(i,shared)).encode()
                    file=p/'training_audit/windows'/f'step_{i}.json';file.write_bytes(b)
                    z.writestr('audit/'+file.name,b)
                    expected[str(file.relative_to(self.run))]=b
        old=self.root/'old_core.zip'
        with ZipFile(old,'w',ZIP_DEFLATED,compresslevel=9) as z:
            for p in self.run.rglob('*'):
                if p.is_file():z.write(p,str(p.relative_to(self.run)))
        self.assertGreater(old.stat().st_size,c.LIMIT)
        result=self.collect();self.assertLess(result['bytes'],c.LIMIT)
        m,restored=self.contents(result)
        for name,b in expected.items():self.assertEqual(restored[name],b)
        self.assertEqual(len(restored),15)
        print('real-cap regression: old ZIP=%d; new ZIP=%d; all 15 sources restored'%(old.stat().st_size,result['bytes']))


if __name__=='__main__':unittest.main()
