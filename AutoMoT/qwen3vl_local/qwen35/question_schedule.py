"""Online-only, deterministic per-rank question windows; never a training sampler."""
import os
from dataclasses import dataclass


@dataclass
class QuestionSchedule:
    window_frames: int = 32
    slots: int = 1
    seed: str = 'online'
    rank: int = 0
    requests: int = 0

    def __post_init__(self):
        if type(self.window_frames) is not int or self.window_frames < 1:
            raise ValueError('window_frames must be a positive integer')
        if self.slots not in (1, 2) or self.window_frames < self.slots:
            raise ValueError('Use one or two question slots within each window')
        if type(self.rank) is not int or self.rank < 0 or self.requests < 0:
            raise ValueError('Invalid rank/request counter')

    def next(self):
        number=self.requests;window=number//self.window_frames
        slot=(number%self.window_frames)%self.slots
        self.requests+=1
        return dict(protocol='rank_question_windows_v1', request=number, rank=self.rank,
            window=window, slot=slot, window_frames=self.window_frames, slots=self.slots,
            seed=f'{self.seed}:rank={self.rank}:window={window}:slot={slot}')


def for_bundle(bundle, namespace, seed='online'):
    """Changing explicit policy starts a new schedule; caches still match exact tokens."""
    policy=(int(os.environ.get('AUTOMOT_QUESTION_WINDOW', '32')),
            int(os.environ.get('AUTOMOT_QUESTION_SLOTS', '1')), str(seed),
            int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))))
    schedules=getattr(bundle, '_question_schedules', None)
    if schedules is None:
        schedules={};bundle._question_schedules=schedules
    key=(namespace,policy)
    if key not in schedules:schedules[key]=QuestionSchedule(*policy)
    return schedules[key]
