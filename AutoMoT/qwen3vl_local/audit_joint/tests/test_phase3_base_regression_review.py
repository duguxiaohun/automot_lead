import unittest
import json
import hashlib
import subprocess
import sys
import pytest
from qwen3vl_local.audit_joint.phase3_base_regression_review import compare

class RegressionReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(key=['val','s','r',i,'E'],gt=gt,pred=pred,correct=gt==pred,features=[feature]) for i,(gt,pred,feature) in enumerate([
            ('RESUME','STOP','nearzero_resume'),('STOP','STOP','wait_then_gain'),('STOP','STOP','wait_then_gain'),('DECELERATE','KEEP','short_drop')])]
        self.pred=[dict(key=r['key'],raw_output=r['pred'],reference_sha256='sha') for r in self.rows]
    def test_self_control(self):
        result=compare(self.rows,self.pred,'sha')
        self.assertEqual(result['status'],'no_regression_observed');self.assertFalse(result['promotion_authorized'])
        self.assertEqual(result['metrics']['ALL']['delta_correct'],0)
    def test_total_improves_but_existing_slice_degrades(self):
        self.pred[0]['raw_output']='RESUME';self.pred[3]['raw_output']='DECELERATE';self.pred[1]['raw_output']='RESUME'
        result=compare(self.rows,self.pred,'sha')
        self.assertEqual(result['metrics']['ALL']['delta_correct'],1)
        self.assertIn('semantic/wait_then_gain',result['regression_slices'])
        self.assertEqual(result['status'],'regression_detected')
    def test_subset_rejected(self):
        with self.assertRaises(ValueError):compare(self.rows,self.pred[:1],'sha')
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):compare(self.rows,self.pred+[self.pred[0]],'sha')
    def test_reference_binding_rejected(self):
        with self.assertRaises(ValueError):compare(self.rows,self.pred,'other')
    def test_changed_label_rejected(self):
        self.pred[0]['gt']='STOP'
        with self.assertRaises(ValueError):compare(self.rows,self.pred,'sha')
    def test_invalid_format_is_failure(self):
        self.pred[1]['raw_output']='STOP because danger'
        result=compare(self.rows,self.pred,'sha')
        self.assertEqual(result['metrics']['ALL']['after']['format_valid'],3)
        self.assertIn('format_valid',result['regression_slices'])
    def test_test_rows_never_enter_tuning(self):
        extra=dict(self.rows[0],key=['test','s','r',0,'E'])
        result=compare(self.rows+[extra],self.pred,'sha')
        self.assertEqual(result['metrics']['ALL']['before']['n'],4)

if __name__=='__main__':unittest.main()

class AdditionalRegressionTests(unittest.TestCase):
    def test_event_precision_decline_cannot_cancel_across_events(self):
        rows=[];predictions=[]
        for i,(event,gt,old,new) in enumerate([
            ('LEAD_BRAKE','STOP','STOP','STOP'),
            ('LEAD_BRAKE','RESUME','KEEP','STOP'),
            ('SIGNAL_FAILURE','STOP','STOP','STOP'),
            ('SIGNAL_FAILURE','RESUME','STOP','KEEP')]):
            key=['val','s','r',i,event]
            rows.append(dict(key=key,gt=gt,pred=old,correct=gt==old))
            predictions.append(dict(key=key,raw_output=new,reference_sha256='s'))
        result=compare(rows,predictions,'s')
        self.assertEqual(result['metrics']['ALL']['delta_correct'],0)
        self.assertEqual(result['previously_correct_lost'],[])
        self.assertEqual(result['metrics']['ALL']['before']['actions'],result['metrics']['ALL']['after']['actions'])
        self.assertIn('event/LEAD_BRAKE/action_metric/STOP/precision',result['regression_slices'])
        self.assertIn('event/LEAD_BRAKE/action_metric/STOP/f1',result['regression_slices'])
        self.assertIn('event/LEAD_BRAKE/macro_f1',result['regression_slices'])
        self.assertTrue(result['review_required'])
    def test_equal_score_swap_is_not_a_clean_pass(self):
        rows=[dict(key=['val','s','r',i,'E'],gt='STOP',pred=p,correct=p=='STOP')
              for i,p in enumerate(['STOP','KEEP'])]
        predictions=[dict(key=r['key'],raw_output=p,reference_sha256='s')
                     for r,p in zip(rows,['KEEP','STOP'])]
        result=compare(rows,predictions,'s')
        self.assertEqual(result['status'],'tradeoff_review_required')
        self.assertTrue(result['review_required'])
        self.assertEqual(result['previously_correct_lost'],[rows[0]['key']])

    def test_action_precision_decline_is_reported(self):
        rows=[dict(key=['val','s','r',i,'E'],gt=g,pred=p,correct=g==p)
              for i,(g,p) in enumerate([('RESUME','RESUME'),('STOP','KEEP'),('KEEP','STOP')])]
        predictions=[dict(key=r['key'],raw_output=p,reference_sha256='s')
                     for r,p in zip(rows,['RESUME','RESUME','KEEP'])]
        result=compare(rows,predictions,'s')
        self.assertIn('action_metric/RESUME/precision',result['regression_slices'])
        self.assertEqual(result['metrics']['ALL']['delta_correct'],1)


@pytest.mark.parametrize('encoding',['json','jsonl'])
@pytest.mark.parametrize('changed',[False,True])
def test_cli_predictions_formats_and_exit_status(tmp_path,encoding,changed):
    reference=tmp_path/'reference.json'
    rows=[dict(key=['val','s','r',0,'E'],gt='STOP',pred='STOP',correct=True)]
    reference.write_text(json.dumps(rows))
    predictions=[dict(key=rows[0]['key'],raw_output='KEEP' if changed else 'STOP',
                      reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())]
    source=tmp_path/('predictions.'+encoding)
    source.write_text(json.dumps(predictions,indent=2) if encoding=='json' else '\n'.join(json.dumps(r) for r in predictions)+'\n')
    output=tmp_path/'review.json'
    result=subprocess.run([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_base_regression_review',
        '--reference',str(reference),'--predictions',str(source),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==(2 if changed else 0),result.stderr
    assert json.loads(output.read_text())['review_required'] is changed


@pytest.mark.parametrize('text',['[1]','null','{"key": "not a cohort"}\n[]'])
def test_predictions_reject_non_object_rows(tmp_path,text):
    from qwen3vl_local.audit_joint.phase3_base_regression_review import read_predictions
    path=tmp_path/'bad.json';path.write_text(text)
    with pytest.raises(ValueError):read_predictions(path)
