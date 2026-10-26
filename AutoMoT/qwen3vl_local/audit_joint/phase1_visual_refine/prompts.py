"""Candidate wording after RGB review; never promotes itself over frozen v5."""
from __future__ import annotations
import hashlib
import json
from . import CONTRACT
from qwen3vl_local.sft_new_loop_phase1 import prompts as native
NAME='sft_new_loop_phase1_visual_refine_v1_candidate'
_ORIGINAL_BUILD=native.build_phase1_prompt
_ORIGINAL_HASH=native.phase1_prompt_sha256
SYSTEM_PROMPT=native.SYSTEM_PROMPT+' Return only the requested KEY: YES or KEY: NO lines, without Markdown or explanation. If audit evidence is explicitly requested, put it only after all answer lines.'
RULES='''[CURRENT_VISIBLE_STRUCTURE_CHECK]
For every RS question, check the same current road structure, including alternatives that are NOT asked in this subset. Do not choose YES merely because an alternative answer key is absent.
An oncoming vehicle in its own lane, a painted centre line, or curbside parking beside an open ego lane is not by itself an opposing-lane constraint. RS2 needs visible narrowing, a shared passage, encroachment, or a current need to borrow/yield for opposing space. Retain RS2 when an obstruction actually removes ego's usable passage; ego need not already have crossed the centre line.
Keep RS1 for a visibly open ordinary surface-road corridor when no stronger CURRENT structure is evidenced. A bend, distant side opening, weather, moving actor, or mere opposing-lane visibility does not establish a junction or RS2. Do not default to RS1 when the governing structure is unobservable.
Keep RS4 when identifiable traffic-signal hardware governs the LOCAL junction, including failed/dark hardware; absence of a working signal does not automatically imply RS5. A bare pole, streetlamp, reflection, or distant signal on another road is insufficient.
Keep RS5 when a local road connection and priority/stop/yield/gap-acceptance context are visible. Do not discard a real STOP-controlled intersection merely because ego continues straight. A speed-limit sign or a driveway alone is insufficient.
Keep the existing HIGHWAY controlled-access-topology requirement: width, rails, fog and fast-looking traffic alone do not prove it. Do not erase clear ramp/merge/separated controlled-corridor evidence.
Older RGB is evidence only while its target and context remain relevant to the newest frame. Do not transfer visibility from an initialization/weather discontinuity to an unobservable newest situation.
[/CURRENT_VISIBLE_STRUCTURE_CHECK]'''

def build(*,spec=None,audit=False,history_rgb_mode='4rgb'):
    text=_ORIGINAL_BUILD(spec=spec,audit=audit,history_rgb_mode=history_rgb_mode)
    text=text.replace(native.PROMPT_NAME,NAME,1)
    tail=('Output each requested answer key exactly once as KEY: YES or KEY: NO. No bullets, bold text, preamble, reasoning, or extra answer keys. '
          +('Then emit only the requested EVIDENCE lines.' if audit else 'Stop immediately after the last requested answer.'))
    return RULES+'\n'+text+'\n[FINAL_FORMAT_CHECK]\n'+tail+'\n[/FINAL_FORMAT_CHECK]'

def fingerprint(*,audit=False,history_rgb_mode='4rgb'):
    payload={'contract':CONTRACT,'name':NAME,'parent_prompt_sha256':_ORIGINAL_HASH(audit=audit,history_rgb_mode=history_rgb_mode),
             'shared_rules':RULES,'system_prompt':SYSTEM_PROMPT,'audit':audit,'rgb_mode':history_rgb_mode,
             'rendered_default':build(audit=audit,history_rgb_mode=history_rgb_mode)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
