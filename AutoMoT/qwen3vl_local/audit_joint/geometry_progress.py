"""Optional launch-time progress observer; geometry source/data identities stay intact.

Used by geometry_v1/run.sh for new train/evaluate processes. Cannot attach to an
already running job. ETA covers optimizer steps only, excluding validation/save/test.
"""
import functools
import json
import os
from pathlib import Path
import time


class Progress:
    def __init__(self, output, epochs):
        self.output = Path(output)
        self.epochs = epochs
        self.epoch = 0
        self.completed = 0
        self.validation_total = 0
        self.predictions = 0
        self.prediction_total = 0
        self.prediction_stage = 'validation'
        self.last_emit = 0.
        self.io_warned = False

    def emit(self, stage, **fields):
        if int(os.environ.get('RANK', '0')) != 0:
            return
        row = dict(stage=stage, epoch=self.epoch, epochs=self.epochs,
                   time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)
        print('[geometry progress] ' + json.dumps(row, ensure_ascii=False), flush=True)
        # A failed progress write must not turn a successful optimizer step into
        # a distributed training failure. The original run owns output creation.
        if self.output.is_dir():
            try:
                temporary = self.output/'progress.json.tmp'
                temporary.write_text(json.dumps(row, indent=2)+'\n')
                temporary.replace(self.output/'progress.json')
            except OSError as exc:
                if not self.io_warned:
                    print('[geometry progress] file write failed: '+str(exc), flush=True)
                    self.io_warned = True

    def epoch_call(self, original, bundle, work, optimizer, scheduler, pack_record, **kwargs):
        self.epoch += 1
        steps = len(work)//kwargs.get('world', 1)
        self.emit('train', step=0, epoch_steps=steps, global_step=self.completed,
                  total_steps=steps*self.epochs, eta_seconds=None)
        start = time.monotonic()
        count = 0
        old_step = scheduler.step
        # The native loop calls scheduler.step only AFTER optimizer.step succeeds.
        # Restore the exact instance state before checkpoint serialization.
        had_instance_step = 'step' in scheduler.__dict__
        instance_step = scheduler.__dict__.get('step')

        @functools.wraps(old_step)
        def step(*args, **kw):
            nonlocal count
            value = old_step(*args, **kw)
            count += 1
            now = time.monotonic()
            if count == 1 or count % 10 == 0 or count == steps or now-self.last_emit >= 30:
                elapsed = now-start
                seconds_per_step = elapsed/count
                self.emit('train', step=count, epoch_steps=steps,
                          global_step=self.completed+count, total_steps=steps*self.epochs,
                          seconds_per_step=round(seconds_per_step, 3),
                          epoch_remaining_seconds=round(seconds_per_step*(steps-count), 1),
                          optimizer_remaining_seconds=round(seconds_per_step*(steps*self.epochs-self.completed-count), 1),
                          eta_excludes='validation, checkpoint saving, test; estimate changes with speed',
                          eta_provisional=count < 100, learning_rate=optimizer.param_groups[0]['lr'],
                          progress_scope='rank0 successful optimizer and scheduler steps')
                self.last_emit = now
            return value

        scheduler.step = step
        try:
            result = original(bundle, work, optimizer, scheduler, pack_record, **kwargs)
        finally:
            if had_instance_step:
                scheduler.step = instance_step
            else:
                del scheduler.step
        self.completed += result['steps']
        self.emit(result['status'], global_step=self.completed, step=result['steps'],
                  epoch_steps=steps, loss_rank0=result['loss'], elapsed_seconds=result['elapsed_seconds'])
        return result


def install(train):
    """Install observers in this new process only; retain all native operations."""
    state = {}

    def wrap_operation(name):
        original = getattr(train, name)

        @functools.wraps(original)
        def operation(args):
            if int(os.environ.get('RANK', '0')) != 0:
                return original(args)
            progress = Progress(args.output, args.epochs)
            state['progress'] = progress
            progress.prediction_stage = 'test' if name == 'evaluate' else 'validation'
            progress.emit('startup')
            try:
                result = original(args)
            except BaseException as exc:
                progress.emit('failed', error=f'{type(exc).__name__}: {exc}')
                raise
            else:
                progress.emit('evaluation_complete' if name == 'evaluate' else 'training_complete',
                              global_step=progress.completed)
                return result
            finally:
                state.pop('progress', None)
        setattr(train, name, operation)

    original_epoch = train.train_epoch
    def epoch(*args, **kwargs):
        p = state.get('progress')
        return p.epoch_call(original_epoch, *args, **kwargs) if p else original_epoch(*args, **kwargs)
    train.train_epoch = epoch

    original_members = train.training_members
    def members(*args, **kwargs):
        result = original_members(*args, **kwargs)
        if 'progress' in state:
            state['progress'].validation_total = len(result)
        return result
    train.training_members = members

    original_save = train.save_checkpoint
    def save(*args, **kwargs):
        p = state.get('progress')
        if p: p.emit('checkpoint_saving', global_step=p.completed)
        result = original_save(*args, **kwargs)
        if p:
            p.predictions = 0
            p.prediction_total = p.validation_total
            p.emit('checkpoint_saved', global_step=p.completed)
        return result
    train.save_checkpoint = save

    original_evaluation = train.planning.evaluation
    def evaluation(*args, **kwargs):
        rows = original_evaluation(*args, **kwargs)
        p = state.get('progress')
        if p: p.prediction_total = len(rows)
        return rows
    train.planning.evaluation = evaluation

    original_predict = train.objective.predict
    def predict(*args, **kwargs):
        p = state.get('progress')
        if p and p.predictions == 0:
            p.emit(p.prediction_stage, processed=0, total=p.prediction_total)
        result = original_predict(*args, **kwargs)
        if p:
            p.predictions += 1
            if p.predictions == 1 or p.predictions % 10 == 0 or p.predictions == p.prediction_total:
                p.emit(p.prediction_stage, processed=p.predictions, total=p.prediction_total)
        return result
    train.objective.predict = predict
    wrap_operation('run')
    wrap_operation('evaluate')


if __name__ == '__main__':
    from .geometry_v1 import train
    from .geometry_v1.__main__ import main
    install(train)
    main()
