import numpy as np
from collections import namedtuple
from ..trace import Trace
from .step_size import DualAverageAdaptation
from .metrics import *
from ...utils.random_state import check_state
import warnings
from .integration import CpuLeapfrogIntegrator
from copy import deepcopy
from ..stats import StepStats
from ...utils.warnings import SamplingProgess
import time


class SamplingError(RuntimeError):
    pass


HMCStepData = namedtuple("HMCStepData", 
                         "end, accept_stat, divergence_info, stats")


DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


class BaseHMC:
    """Superclass to implement Hamiltonian/hybrid monte carlo."""

    def __init__(self, logp_and_grad, trace=None, step_size=0.25, 
                 adapt_step_size=True, metric=None, adapt_metric=True, 
                 random_state=None, Emax=1000, target_accept=0.8, gamma=0.05, 
                 k=0.75, t0=10, x_0=None):
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        scaling : array_like, ndim = {1,2}
            Scaling for momentum distribution. 1d arrays interpreted matrix
            diagonal.
        step_scale : float, default=0.25
            Size of steps to take, automatically scaled down by 1/n**(1/4)
        model : pymc3 Model instance
        potential : Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        """
        
        self._logp_and_grad = logp_and_grad
        if isinstance(trace, Trace):
            self._trace = trace
        else:
            x_0 = np.asarray(x_0).reshape(-1)
            try:
                logp_0, _ = logp_and_grad(x_0)
                assert np.isfinite(logp_0)
            except:
                raise ValueError('failed to get finite logp at x0.')
            if isinstance(step_size, DualAverageAdaptation):
                pass
            else:
                if step_size is None:
                    step_size = 1.
                step_size = DualAverageAdaptation(
                    step_size / x_0.shape[0]**0.25, target_accept, gamma, k, t0, 
                    bool(adapt_step_size))
            if isinstance(metric, QuadMetric):
                pass
            else:
                if metric is None:
                    metric = np.ones_like(x_0)
                metric = np.atleast_1d(metric)
                if metric.shape[-1] != x_0.shape[0]:
                    raise ValueError('dim of metric is incompatible with x_0.')
                if metric.ndim == 1:
                    if adapt_metric:
                        metric = QuadMetricDiagAdapt(x_0.shape[0], x_0, metric, 
                                                     10)
                    else:
                        metric = QuadMetricDiag(metric)
                elif metric.ndim == 2:
                    if adapt_metric:
                        warnings.warn(
                            'You give a full rank metric array and set '
                            'adapt_metric as True, but we haven\'t implemented '
                            'adaptive full rank metric yet, so an adaptive '
                            'diagonal metric will be used.', RuntimeWarning)
                        metric = QuadMetricDiagAdapt(x_0.shape[0], x_0, 
                                                     np.diag(metric), 10)
                    else:
                        metric = QuadMetricFull(metric)
                else:
                    raise ValueError('metric should be a QuadMetric, or a 1-d '
                                     'or 2-d array.')
            random_state = check_state(random_state)
            self._trace = Trace(step_size, metric, random_state, Emax, -1, x_0,
                                logp_0)
        self.integrator = CpuLeapfrogIntegrator(self._trace.metric, 
                                                logp_and_grad)

    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")

    def astep(self):
        """Perform a single HMC iteration."""
        q0 = self._trace.samples[-1]
        p0 = self._trace.metric.random(self._trace.random_state)
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            self._trace.metric.raise_ok()
            raise SamplingError(
                "Bad initial energy, please check the Hamiltonian at p = {}, "
                "q = {}.".format(p0, q0))
            
        step_size = self._trace.step_size.current(self.warmup)
        hmc_step = self._hamiltonian_step(start, p0, step_size)
        self._trace.step_size.update(hmc_step.accept_stat, self.warmup)
        self._trace.metric.update(hmc_step.end.q, self.warmup)
        step_stats = StepStats(**hmc_step.stats, 
                               **self._trace.step_size.sizes(), 
                               warmup=self.warmup, 
                               diverging=bool(hmc_step.divergence_info))
        self._trace.update(hmc_step.end.q, step_stats)
    
    def run(self, n_iter=3000, n_warmup=1000, verbose=True, n_update=None,
            return_copy=True):
        n_iter = int(n_iter)
        n_warmup = int(n_warmup)
        if not n_iter >= 0:
            raise ValueError('n_iter cannot be negative.')
        if n_warmup > n_iter:
            warnings.warn('n_warmup is larger than n_iter. Setting n_warmup = '
                          'n_iter for now.', RuntimeWarning)
            n_warmup = n_iter
        if self._trace.n_iter > self._trace.n_warmup and n_warmup > 0:
            warnings.warn('self.trace indicates that warmup has completed, so '
                          'n_warmup will be set to 0.', RuntimeWarning)
            n_warmup = 0
        i_iter = self._trace.i_iter
        self._trace._n_iter += n_iter
        self._trace._n_warmup += n_warmup
        n_iter = self._trace._n_iter
        n_warmup = self._trace._n_warmup
        if verbose:
            n_run = n_iter - i_iter
            if n_update is None:
                n_update = n_run // 10
            else:
                n_update = int(n_update)
                if n_update <= 0:
                    warnings.warn('invalid n_update value. Using n_run // 10 '
                                  'for now.', RuntimeWarning)
                    n_update = n_run // 10
            t_s = time.time()
            t_i = time.time()
        for i in range(i_iter, n_iter):
            if verbose:
                if i > i_iter and not i % n_update:
                    t_d = time.time() - t_i
                    t_i = time.time()
                    n_div = np.sum(self._trace._stats._diverging[-n_update:])
                    msg_0 = ('sampling proceeding [ {} / {} ], last {} '
                             'samples used {:.2f} seconds'.format(i, 
                             n_iter, n_update, t_d))
                    if n_div / n_update > 0.05:
                        msg_1 = (', while divergence encountered in {} '
                               'sample(s).'.format(n_div))
                    else:
                        msg_1 = '.'
                    if self.warmup:
                        msg_2 = ' (warmup)'
                    else:
                        msg_2 = ''
                    warnings.warn(msg_0 + msg_1 + msg_2, SamplingProgess)
            self.warmup = bool(i < n_warmup)
            self.astep()
        if verbose:
            t_f = time.time()
            warnings.warn('sampling finished   [ {} / {} ], obtained {} '
                          'samples in {:.2f} seconds.'.format(
                          n_iter, n_iter, n_run, t_f - t_s), SamplingProgess)
        return self.trace if return_copy else self._trace
    
    @property
    def trace(self):
        return deepcopy(self._trace)
    