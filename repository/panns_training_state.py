import math
import random


class EpochSampler:
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Empty training panel')
        self.size = size
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.cursor = size
        self.epoch = 0

    def next_batch(self):
        if self.cursor == self.size:
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        batch = self.order[self.cursor:self.cursor+4]
        self.cursor += len(batch)
        return batch

    def state_dict(self):
        return dict(order=self.order.copy(), cursor=self.cursor, epoch=self.epoch, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if sorted(state['order']) != list(range(self.size)):
            raise ValueError('Sampler permutation differs from training panel')
        if not 0 <= state['cursor'] <= self.size or state['epoch'] < 0:
            raise ValueError('Invalid sampler cursor or epoch')
        self.order, self.cursor, self.epoch = state['order'].copy(), state['cursor'], state['epoch']
        self.rng.setstate(state['rng'])


def accumulate_diagnostics(total, record):
    for name, current in record['gradient_norms'].items():
        value = total.setdefault('gradient_norms', {}).setdefault(name, dict(steps=0, clipped=0, norm_sum=0., norm_max=0.))
        value['steps'] += 1
        value['clipped'] += int(current > 1.)
        value['norm_sum'] += current
        value['norm_max'] = max(value['norm_max'], current)
        value['clipping_fraction'] = value['clipped']/value['steps']
        value['mean_preclip_norm'] = value['norm_sum']/value['steps']
    if record.get('attention'):
        current = record['attention']
        value = total.setdefault('attention', dict(segments=0, entropy_sum=0., peak_sum=0.))
        for key in ('segments', 'entropy_sum', 'peak_sum'):
            value[key] += current[key]
        value['mean_entropy_nats'] = value['entropy_sum']/value['segments']
        value['mean_peak_weight'] = value['peak_sum']/value['segments']


def record_validation(state, metrics, epoch):
    if epoch != state['last_validated_epoch'] + 1:
        raise ValueError('Selection validation must commit each complete epoch exactly once')
    values = [metrics[name] for name in ('selection_presence_bce', 'voice_bce', 'music_bce')]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Nonfinite or negative selection loss')
    loss = metrics['selection_presence_bce']
    improved = state['best_loss'] is None or loss < state['best_loss']
    if improved:
        state['best_loss'], state['best_epoch'] = loss, epoch
    anchor = state['patience_anchor']
    if anchor is None or (loss < anchor and anchor-loss >= state['policy']['min_delta']):
        state['patience_anchor'], state['bad_epochs'] = loss, 0
    else:
        state['bad_epochs'] += 1
    state['last_validated_epoch'] = epoch
    state['history'].append(dict(metrics, epoch=epoch))
    return improved


def validate_stopping_state(state, policy, sampler, step, steps_per_pass):
    expected_epoch = (step-1)//steps_per_pass+1 if step else 0
    if state.get('policy') != policy or sampler.epoch != expected_epoch:
        raise ValueError('Checkpoint stopping policy or sampler epoch changed')
    expected_cursor = sampler.size if step and step % steps_per_pass == 0 else (step % steps_per_pass)*4
    if sampler.cursor != (expected_cursor if step else sampler.size):
        raise ValueError('Checkpoint sampler cursor differs from completed optimizer steps')
    last = state.get('last_validated_epoch')
    if type(last) is not int or last not in ({sampler.epoch-1, sampler.epoch} if sampler.cursor == sampler.size and step else {sampler.epoch-1} if step else {0}):
        raise ValueError('Checkpoint contains missing or premature epoch validation')
    history = state.get('history', [])
    if [record.get('epoch') for record in history] != list(range(last+1)):
        raise ValueError('Checkpoint validation history is incomplete')
    replay = dict(policy=policy, best_loss=None, best_epoch=0, patience_anchor=None,
                  bad_epochs=0, last_validated_epoch=-1, history=[])
    for record in history:
        record_validation(replay, {key:value for key,value in record.items() if key != 'epoch'}, record['epoch'])
    if replay != state:
        raise ValueError('Checkpoint patience or selected epoch differs from its validation history')


def validate_completion(done, run_id, checkpoint_sha256, budget):
    policy = budget['early_stopping']
    integers = ('steps', 'epochs_completed', 'best_epoch', 'bad_epochs', 'last_validated_epoch')
    if (done.get('complete') is not True or done.get('run_identity') != run_id
            or done.get('checkpoint_sha256') != checkpoint_sha256
            or done.get('early_stopping') != policy or done.get('monitor') != policy['monitor']
            or any(type(done.get(key)) is not int for key in integers)):
        raise ValueError('Completed training identity or early-stopping contract mismatch')
    epochs = done['epochs_completed']
    if (not 1 <= epochs <= policy['max_epochs'] or done['last_validated_epoch'] != epochs
            or done['steps'] != epochs * budget['steps_per_pass'] or done['steps'] > budget['training_steps']
            or not 0 <= done['best_epoch'] <= epochs or not 0 <= done['bad_epochs'] <= epochs
            or any(type(done.get(key)) not in (int, float) or not math.isfinite(done[key]) or done[key] < 0
                   for key in ('best_loss', 'patience_anchor'))
            or done['best_loss'] > done['patience_anchor']):
        raise ValueError('Completion requires a fully validated epoch and finite selected loss')
    reason = 'early_stopping' if done['bad_epochs'] >= policy['patience'] else 'max_epochs' if epochs == policy['max_epochs'] else None
    if reason is None or done.get('stop_reason') != reason:
        raise ValueError('Interrupted or nonterminal training cannot be marked complete')
