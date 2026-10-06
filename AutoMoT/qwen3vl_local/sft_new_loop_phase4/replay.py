"""在连续真实RGB上按预测状态构题，逐次派生先验并输出完整回放轨迹。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from pathlib import Path
from .identity import ROOT,write_json,digest
from .route_prompts import prompt
from .model import load_for_inference,generate
from .evaluate import replay
from .dataset import EPISODE_FIELDS
from .observation import validate_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT.parents[1]/'checkpoints/Qwen3.5-4B')
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--offline-geometry-safety',action='store_true',
                   help='compute clearances from source-bound current planner envelopes and causal bboxes')
    a=p.parse_args()
    import torch
    sequence=json.loads(a.sequence.read_text())
    bundle=load_for_inference(a.model_dir,a.adapter,torch.device(a.device))
    for obs in sequence['observations']:
        if obs['frame_id'] != obs['observation']['frame_id']:
            raise ValueError('sequence and RGB timestamps disagree')
        validate_observation(bundle.observation_contract,obs['observation'],len(obs['images']))
    def predictor(ep,key,obs):
        row=dict(episode={k:v for k,v in ep.to_dict().items() if k in EPISODE_FIELDS},edge=key,
                 observation=obs['observation'],images=obs['images'],image_sha256=obs['image_sha256'],
                 prompt_sha256=digest(prompt(ep,key,obs['observation'])))
        answer,_=generate(bundle,row,a.data_root)
        return 'UNKNOWN' if answer=='MALFORMED' else answer
    from .replay_safety import ReplaySafetyAdapter
    safety=ReplaySafetyAdapter(a.data_root) if a.offline_geometry_safety else None
    result=replay(sequence['initial'],sequence['observations'],predictor,safety_adapter=safety)
    write_json(a.output,result)

if __name__=='__main__':
    main()
