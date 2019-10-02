import numpy as np
from scipy.special import expit
import multiprocessing as mp
import warnings

from ..samplers.pymc3.nuts import NUTS
from ..utils.warnings import SamplingProgess

__all__ = ['get_sig_beta', 'TemperedDensity', 'AIS']


# TODO: add support for logp_and_grad, probably use Density class
# TODO: add support for lambda functions with pool
# TODO: improve progress output

def get_sig_beta(n, delta=4):
    n = int(n)
    if not n >= 2:
        raise ValueError('n should be at least 2, instead of {}.'.format(n))
    _beta = expit(delta * (2 * np.arange(1, n + 1) / n - 1))
    beta = (_beta - _beta[0]) / (_beta[-1] - _beta[0])
    return beta


class TemperedDensity:
    
    def __init__(self, logp, grad, logp_0, grad_0, beta, start=0):
        self._logp = logp
        self._logp_0 = logp_0
        self._grad = grad
        self._grad_0 = grad_0
        if isinstance(beta, int):
            self._n = beta
            self._beta = get_sig_beta(self._n)
        else:
            try:
                beta = np.asarray(beta)
                beta = beta.reshape(-1)
                assert beta.size >= 2
            except:
                raise ValueError('beta should be an array with len >= 2.')
            self._beta = beta
        self.set(start)
    
    def _get_b(self, i):
        if i is not None:
            self.set(i)
        return self._beta[self._i]
    
    def logp_t(self, x, i=None):
        b = self._get_b(i)
        return b * self._logp(x) + (1 - b) * self._logp_0(x)
    
    def grad_t(self, x, i=None):
        b = self._get_b(i)
        return b * self._grad(x) + (1 - b) * self._grad_0(x)
    
    def logp_and_grad_t(self, x, i=None):
        b = self._get_b(i)
        return (b * self._logp(x) + (1 - b) * self._logp_0(x), 
                b * self._grad(x) + (1 - b) * self._grad_0(x))
    
    @property
    def i(self):
        return self._i
    
    @property
    def n(self):
        return self._n
    
    @property
    def beta(self):
        return self._beta.copy()
    
    def next(self):
        if self._i < self._n - 1:
            self._i += 1
        else:
            warnings.warn('we are already at the last beta.', RuntimeWarning)
    
    def previous(self):
        if self._i > 0:
            self._i -= 1
        else:
            warnings.warn('we are already at the first beta.', RuntimeWarning)
    
    def set(self, index):
        index = int(index)
        if not 0 <= index < self._n:
            start = 0
            warnings.warn(
                'index is out of range. Use 0 for now.', RuntimeWarning)
        self._i = int(index)
    
    def set_first(self):
        self._i = 0
    
    def set_last(self):
        self._i = self._n - 1

class _AIS:
    
    def __init__(self, logp_t, x_0, reverse, n_warmup, nuts_kwargs, verbose, 
                 n_update):
        self.logp_t = logp_t
        self.x_0 = x_0
        self.reverse = reverse
        self.n_warmup = n_warmup
        self.nuts_kwargs = nuts_kwargs
        self.verbose = verbose
        self.n_update = n_update
    
    def worker(self, i):
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        import bayesfast.utils.warnings as bfwarnings
        import warnings
        warnings.simplefilter('always', SamplingProgess)
        warnings.showwarning = bfwarnings.showwarning_chain(i)
        warnings.formatwarning = bfwarnings.formatwarning_chain(i)
        if self.reverse:
            self.logp_t.set_last()
        else:
            self.logp_t.set_first()
        nuts = NUTS(logp_and_grad=self.logp_t.logp_and_grad_t, x_0=self.x_0[i], 
                    **self.nuts_kwargs)
        logz = 0
        tt = nuts.run(self.n_warmup, self.n_warmup, verbose=False, 
                      return_copy=False)
        logz -= tt._stats._logp[-1]
        if self.reverse:
            self.logp_t.previous()
        else:
            self.logp_t.next()
        logz += self.logp_t.logp_t(tt._samples[-1])
        for j in range(1, self.logp_t._n - 1):
            if self.verbose and not j % self.n_update:
                warnings.warn(
                    'sampling proceeding [{} / {}].'.format(j, self.logp_t._n), 
                    SamplingProgess)
            tt = nuts.run(1, 1, verbose=False, return_copy=False)
            logz -= tt._stats._logp[-1]
            if self.reverse:
                self.logp_t.previous()
            else:
                self.logp_t.next()
            logz += self.logp_t.logp_t(tt._samples[-1])
        if self.verbose:
            warnings.warn(
                'sampling finished [{} / {}].'.format(self.logp_t._n, 
                self.logp_t._n), SamplingProgess)
        if self.reverse:
            return -logz
        else:
            return logz
    
    __call__ = worker

        
def AIS(logp_t, x_0, pool=None, m_pool=None, reverse=False, n_warmup=500, 
        nuts_kwargs={}, verbose=True, n_update=None):
    if not isinstance(logp_t, TemperedDensity):
        raise ValueError('logp_t should be a TemperedDensity.')
    x_0 = np.asarray(x_0)
    if not x_0.ndim == 2:
        raise ValueError('x_0 should be a 2-d array.')
    nuts_kwargs = nuts_kwargs.copy()
    if not 'metric' in nuts_kwargs:
        nuts_kwargs['metric'] = np.diag(np.cov(x_0, rowvar=False))
    m, n = x_0.shape
    _new_pool = False
    if pool is None:
        n_pool = min(m, m_pool) if (m_pool is not None) else m
        pool = mp.Pool(n_pool)
        _new_pool = True
    elif pool is False:
        pass
    else:
        if not hasattr(pool, 'map'):
            raise ValueError('pool does not have attribute "map".')
    if n_update is None:
        n_update = logp_t._n // 5
    worker = _AIS(logp_t, x_0, reverse, n_warmup, nuts_kwargs, verbose, 
                  n_update)
    if pool:
        map_result = pool.map(worker, np.arange(m))
    else:
        map_result = list(map(worker, np.arange(m)))
    if _new_pool:
        pool.close()
        pool.join()
    if x_0.shape[0] == 1:
        # we cannot estimate the error with only one simulation
        return np.mean(map_result), None
    else:
        return np.mean(map_result), np.std(map_result) / m**0.5
