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
