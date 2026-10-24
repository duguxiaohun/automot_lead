import random
import unittest
from copy import deepcopy
from qwen3vl_local.audit_joint.phase3_base_label_lab import decision,project
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as native

class LabelLabTests(unittest.TestCase):
    def test_stop_boundary(self):
        s=[8,8,7,6,5,3,.3,0,0]
        self.assertEqual(decision(s,variant='native')['action'],'DECELERATE')
        self.assertEqual(decision(s,variant='stop_confirm_175')['action'],'STOP')
    def test_pair_outside_new_boundary(self):
        s=[8,8,7,6,5,3,2,.3,0]
        self.assertEqual(decision(s,variant='stop_confirm_175')['action'],'DECELERATE')
    def test_single_drop_and_recovery(self):
        s=[6,6,4,6,6,6,6,6,6]
        self.assertEqual(decision(s,variant='native')['action'],'DECELERATE')
        self.assertEqual(decision(s,variant='drop_confirm_two')['action'],'NONE')
    def test_two_drops(self):
        self.assertEqual(decision([6,6,4,4,6,6,6,6,6],variant='drop_confirm_two')['action'],'DECELERATE')
    def test_last_drop_requires_confirmation(self):
        self.assertEqual(decision([6]*8+[4],variant='drop_confirm_two')['action'],'NONE')
    def test_monotone_deceleration_can_be_lost_at_window_end(self):
        # RGB audit case 6: this is sustained slowing, not an isolated speed dip.
        s=[7.727,7.726,7.508,7.463,7.240,6.991,6.721,6.401,6.009]
        self.assertTrue(all(y<=x for x,y in zip(s,s[1:])))
        self.assertEqual(decision(s)['first_drop_s'],2.0)
        self.assertEqual(decision(s)['action'],'DECELERATE')
        self.assertEqual(decision(s,variant='drop_confirm_two')['action'],'NONE')
    def test_control_free_nearzero_jitter_can_expose_lateral_label(self):
        # RGB audit case 28: +0.5s is still essentially stationary. Preserve this
        # known candidate failure rather than silently treating it as valid onset.
        s=[.0000325,.0000356,.0000368,.0638,.8946,1.9949,3.6865,5.2379,5.9049]
        e=dict(future_speeds_exact_mps=s,brake=True,throttle=0,
               lane_change_direction='LEFT',lateral_observation_complete=True)
        self.assertEqual(project(e,'STATIC_BLOCKAGE','native')['action'],'STOP')
        self.assertEqual(project(e,'STATIC_BLOCKAGE','pullaway_without_controls')['action'],'LANE_CHANGE_LEFT')
    def test_late_wait_does_not_become_pullaway(self):
        for v in ('native','stop_confirm_175','drop_confirm_two','pullaway_without_controls'):
            self.assertEqual(decision([0,0,0,0,0,1,2,3,4],brake=True,throttle=0,variant=v)['action'],'STOP')
    def test_control_free_only_changes_gate(self):
        s=[0,.1,.4,1.5,2,3,4,5,6]
        self.assertEqual(decision(s,brake=True,throttle=0)['action'],'STOP')
        self.assertEqual(decision(s,brake=True,throttle=0,variant='pullaway_without_controls')['action'],'RESUME')
    def test_missing_and_invalid_never_keep(self):
        for s in ([1]*8,[1]*8+[float('nan')],[1]*8+[-1]):
            for v in ('native','stop_confirm_175','drop_confirm_two','pullaway_without_controls'):
                self.assertFalse(decision(s,variant=v)['eligible'])
    def test_drop_confirmation_can_create_mixed_window(self):
        s=[6,4,8,8,6,6,6,6,6]
        self.assertTrue(decision(s)['eligible'])
        self.assertFalse(decision(s,variant='drop_confirm_two')['eligible'])
    def test_domain_and_precedence(self):
        e=dict(future_speeds_exact_mps=[8,8,7,6,5,3,.3,0,0],brake=False,throttle=1,lane_change_direction='LEFT',lateral_observation_complete=True)
        self.assertEqual(project(e,'STATIC_BLOCKAGE','native')['action'],'LANE_CHANGE_LEFT')
        self.assertEqual(project(e,'STATIC_BLOCKAGE','stop_confirm_175')['action'],'STOP')
        self.assertEqual(project(e,'LEAD_BRAKE','native')['action'],'DECELERATE')
    def test_inputs_not_mutated(self):
        e=dict(future_speeds_exact_mps=[6]*9,brake=False,throttle=1,lane_change_direction='',lateral_observation_complete=True);before=deepcopy(e)
        for v in ('native','stop_confirm_175','drop_confirm_two','pullaway_without_controls'):project(e,'STATIC_BLOCKAGE',v)
        self.assertEqual(before,e)
    def test_tails_ignored(self):
        for v in ('native','stop_confirm_175','drop_confirm_two','pullaway_without_controls'):
            self.assertEqual(decision([6]*9,variant=v)['action'],decision([6]*9+[0,0],variant=v)['action'])
    def test_unaffected_variants_match_native_random_windows(self):
        # Without near-zero values, changed STOP horizon/control gate cannot matter.
        rng=random.Random(2410)
        for _ in range(1000):
            s=[rng.uniform(1,12) for _ in range(9)];old=decision(s)
            for v in ('stop_confirm_175','pullaway_without_controls'):
                new=decision(s,variant=v)
                self.assertEqual((old['eligible'],old['action'],old['reason']),(new['eligible'],new['action'],new['reason']))
    def test_unknown_hypothesis(self):
        with self.assertRaises(ValueError):decision([1]*9,variant='other')

if __name__=='__main__':unittest.main()
