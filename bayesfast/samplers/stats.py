from collections import namedtuple, OrderedDict


__all__ = ['StepStats', 'SamplerStats']


stats_items = ('logp', 'energy', 'tree_depth', 'tree_size', 'mean_tree_accept',
               'step_size', 'step_size_bar', 'warmup', 'energy_change', 
               'max_energy_change', 'diverging')


StepStats = namedtuple('StepStats', stats_items)


class SamplerStats:
    
    def __init__(self, logp=None):
        for si in stats_items:
            setattr(self, '_' + si, ['__init__'])
        if logp is not None:
            try:
                self._logp[0] = float(logp)
            except:
                raise ValueError('logp should be a float.')
    
    def update(self, step_stats):
        if not isinstance(step_stats, StepStats):
            raise ValueError('step_stats should be a StepStats.')
        for si in stats_items:
            getattr(self, '_' + si).append(getattr(step_stats, si))
    
    def get(self, since_iter=0):
        return OrderedDict(
            zip(stats_items, [getattr(self, '_' + si)[since_iter:] for si in 
            stats_items]))
        
    @property
    def n_iter(self):
        return len(self._logp) - 1
    