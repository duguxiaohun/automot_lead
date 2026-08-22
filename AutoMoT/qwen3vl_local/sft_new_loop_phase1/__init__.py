"""Phase1 + Phase2 fused visible-fact YES/NO loop.

This package asks the finalized Phase1 visible-fact questions and the latest
Phase2 ROAD_STRUCTURE questions in one Qwen turn.  It keeps the original
route-disjoint split and focus-balance idea: every supervised answer key is
sampled YES:NO = 1:1, and the four Phase1 keys and four Phase2 keys contribute
the same number of focused training/eval cases.
"""

DATASET_NAME = "sft_new_loop_phase1_phase1_phase2_visible_facts"
