import numpy as np
from collections import namedtuple, OrderedDict
import copy
from .module import *
from .modules.poly import *
from .transforms.constraint import *


# TODO: rewrite DensityLite


class VariableDict(OrderedDict):
    
    def __init__(self, label):
        super().__init__()
        if label == 'fun' or label == 'jac':
            self.label = label
        else:
            raise ValueError('label should be "fun" or "jac", instead of '
                             '"{}".'.format(label))


class Pipeline:
    
    def __init__(self, module_list=None, input_names=['__var__'], 
                 input_dims=None, surrogate=None, input_ranges=None, 
                 lower_bounds=False, upper_bounds=False):
        self.module_list = module_list
        self.input_names = input_names
        self.input_dims = input_dims
        self.surrogate = surrogate
        self.input_ranges = input_ranges
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
    
    @property
    def module_list(self):
        return copy.copy(self._module_list)
    
    @module_list.setter
    def module_list(self, ml):
        self._module_list = []
        if ml is not None:
            if hasattr(ml, '__iter__'):
                try:
                    for mm in ml:
                        self.add_step(mm)
                except:
                    self._module_list = []
                    raise
            else:
                raise ValueError('module_list should be iterable, or None if '
                                 'you want to reset it.')
        
    def add_step(self, pipeline_step, index=None):
        if index is None:
            index = self.nstep
        else:
            index = np.clip(int(index), -self.nstep, self.nstep)
            if self.nstep:
                index %= self.nstep
        if isinstance(pipeline_step, Module):
            self._module_list.insert(index, pipeline_step)
        else:
            try:
                for (ii, mm) in enumerate(pipeline_step):
                    assert isinstance(mm, Module)
                    self._module_list.insert(index + ii, mm)
            except:
                raise ValueError('pipeline_step should be a Module, or a list '
                                 'of Module(s).')
    
    def remove_step(self, start=-1, length=1):
        try:
            start = int(start) % self.nstep
            length = int(length)
            assert length > 0
            assert 0 <= start < self.nstep
            assert 0 <= (start + length) < self.nstep
        except:
            raise ValueError(
                'index out of range; cannot remove {} steps of {}, starting '
                'from step #{}.'.format(length, self.nstep, start))
        for _ in range(length):
            del self._module_list[start]
    
    def change_step(self, pipeline_step, start=-1, length=None):
        if isinstance(pipeline_step, Module):
            pipeline_step = [pipeline_step]
            length = 1 if length is None else int(length)
        elif hasattr(pipeline_step, '__iter__') and all(
            isinstance(ps, Module) for ps in pipeline_step):
            pipeline_step = list(pipeline_step)
            length = len(pipeline_step) if length is None else int(length)
        else:
            raise ValueError('"pipeline_step" should be a Module, or a list of '
                             'Module(s).')
        start = int(start) % self.nstep
        if not (0 <= start and (start + length) < self.nstep):
            raise ValueError(
                'index out of range; cannot change {} steps of {}, starting '
                'from step #{}.'.format(length, self.nstep, start))
        self._module_list[start:(start + length)] = pipeline_step
    
    @property
    def surrogate(self):
        return self._surrogate
    
    @surrogate.setter
    def surrogate(self, surro):
        if isinstance(surro, Surrogate):
            self._surrogate = surro
        elif surro is None:
            self._surrogate = None
        else:
            raise ValueError('surrogate should be a Surrogate, or None if you '
                             'want to reset it.')
    
    def _xx_stop_check(self, xx, stop):
        xx = np.asarray(xx)
        if xx.ndim != 1:
            raise NotImplementedError('xx should be 1-d for now.')
        elif xx.shape[-1] != self._input_size:
            raise ValueError('shape of xx should be ({},) instead of '
                             '{}.'.format(self._input_size, xx.shape))
        if stop is None:
            stop = self.nstep - 1
        else:
            try:
                stop = int(stop)
                stop = stop % self.nstep
            except:
                raise ValueError(
                    'stop should be an int or None, instead of {}'.format(stop))
        return xx, stop
    
    def fun(self, xx, surrogate=False, stop=None, original=True):
        xx, stop = self._xx_stop_check(xx, stop)
        if not original:
            xx = self.to_original(xx, False)
        fun_dict = VariableDict('fun')
        for ii, nn in enumerate(self._input_names):
            fun_dict[nn] = xx[self._input_cum[ii]:self._input_cum[ii + 1]]
        if surrogate:
            start, length = self._surrogate.indices
        for ii in range(stop + 1):
            try:
                if surrogate:
                    if ii < start or ii >= start + length:
                        _module = self._module_list[ii]
                    elif ii == start:
                        _module = self._surrogate
                    else:
                        continue
                else:
                    _module = self._module_list[ii]
                _input = [fun_dict[nn] for nn in _module._input_names]
                _output = _module.fun(*_input)
                if len(_module._output_names) == 1:
                    nn = _module._output_names[0]
                    fun_dict[nn] = _output
                else:
                    for jj, nn in enumerate(_module._output_names):
                        fun_dict[nn] = _output[jj]
                for nn in _module._delete_names:
                    del fun_dict[nn]
            except:
                raise RuntimeError(
                    'pipeline fun evaluation failed at step {}.'.format(ii))
        return fun_dict
    
    def jac(self, xx, surrogate=False, stop=None, original=True):
        return self.fun_and_jac(xx, surrogate, stop, original)[1]
    
    def fun_and_jac(self, xx, surrogate=False, stop=None, original=True):
        xx, stop = self._xx_stop_check(xx, stop)
        fun_dict = VariableDict('fun')
        jac_dict = VariableDict('jac')
        if original:
            _eye = np.eye(self._input_size)
        else:
            _eye = np.diag(self.to_original_grad(xx, False))
            xx = self.to_original(xx, False)
        for ii, nn in enumerate(self._input_names):
            fun_dict[nn] = xx[self._input_cum[ii]:self._input_cum[ii + 1]]
            jac_dict[nn] = _eye[
                self._input_cum[ii]:self._input_cum[ii + 1], :]
        if surrogate:
            start, length = self._surrogate.indices
        for ii in range(stop + 1):
            try:
                if surrogate:
                    if ii < start or ii >= start + length:
                        _module = self._module_list[ii]
                    elif ii == start:
                        _module = self._surrogate
                    else:
                        continue
                else:
                    _module = self._module_list[ii]
                _input = [fun_dict[nn] for nn in _module._input_names]
                _input_jac = np.concatenate([jac_dict[nn] for nn in 
                                             _module._input_names], axis=0)
                _output, _output_jac = _module.fun_and_jac(*_input)
                if len(_module._output_names) == 1:
                    nn = _module._output_names[0]
                    fun_dict[nn] = _output
                    jac_dict[nn] = np.dot(_output_jac, _input_jac)
                else:
                    for jj, nn in enumerate(_module._output_names):
                        fun_dict[nn] = _output[jj]
                        jac_dict[nn] = np.dot(_output_jac[jj], _input_jac)
                for nn in _module._delete_names:
                    del fun_dict[nn], jac_dict[nn]
            except:
                raise RuntimeError(
                    'pipeline fun_and_jac evaluation failed at step '
                    '{}.'.format(ii))
        return fun_dict, jac_dict
    
    __call__ = fun
    
    '''def fun_transformed(self, xx, surrogate=False, stop=None):
        return self.fun(xx, surrogate, stop, False)
    
    def jac_transformed(self, xx, surrogate=False, stop=None):
        return self.jac(xx, surrogate, stop, False)
    
    def fun_and_jac_transformed(self, xx, surrogate=False, stop=None):
        return self.fun_and_jac(xx, surrogate, stop, False)'''
    
    @property
    def input_names(self):
        return copy.copy(self._input_names)
    
    @input_names.setter
    def input_names(self, names):
        if isinstance(names, str):
            names = [names]
        else:
            try:
                names = list(names)
                assert all(isinstance(nn, str) for nn in names)
                assert len(names) > 0
            except:
                raise ValueError('input_names should be a list of str(s), '
                                 'instead of {}.'.format(names))
            if len(names) != len(set(names)):
                raise ValueError('input_names should be a list of unique '
                                 'name(s), instead of {}.'.format(names))
        self._input_names = names
    
    @property
    def input_dims(self):
        return copy.copy(self._input_dims)
    
    @input_dims.setter
    def input_dims(self, dims):
        try:
            dims = np.asarray(dims, dtype=np.int).reshape(-1)
            assert np.all(dims > 0)
            assert len(dims) > 0
        except:
            raise ValueError(
                'input_dims should be an array of positive int(s), instead of '
                '{}.'.format(dims))
        self._input_dims = dims
        self._input_cum = np.cumsum(np.insert(dims, 0, 0))
        self._input_size = np.sum(dims)
    
    @property
    def input_size(self):
        return self._input_size
    
    @property
    def input_ranges(self):
        return self._input_ranges
    
    @input_ranges.setter
    def input_ranges(self, ranges):
        if ranges is None:
            self._input_ranges = np.array(
                (np.full(self.input_size, 0.), 
                 np.full(self.input_size, 1.))).T.copy()
        else:
            try:
                ranges = np.atleast_2d(ranges).copy()
                assert ranges.ndim == 2 and ranges.shape[1] == 2
            except:
                raise ValueError('Invalid value for ranges.')
            self._input_ranges = ranges
    
    @property
    def lower_bounds(self):
        return self._lower_bounds
    
    @lower_bounds.setter
    def lower_bounds(self, bounds):
        if isinstance(bounds, bool):
            self._lower_bounds = np.full(self.input_size, bounds, np.uint8)
        else:
            try:
                bounds = np.asarray(bounds).astype(bool).astype(np.uint8)
                assert bounds.shape == (self.input_size,)
            except:
                raise ValueError(
                    'bounds should be a bool, or consist of bool(s) with shape '
                    '(self.input_size,), instead of {}.'.format(bounds))
            self._lower_bounds = bounds
    
    @property
    def upper_bounds(self):
        return self._upper_bounds
    
    @upper_bounds.setter
    def upper_bounds(self, bounds):
        if isinstance(bounds, bool):
            self._upper_bounds = np.full(self.input_size, bounds, np.uint8)
        else:
            try:
                bounds = np.asarray(bounds).astype(bool).astype(np.uint8)
                assert bounds.shape == (self.input_size,)
            except:
                raise ValueError(
                    'bounds should be a bool, or consist of bool(s) with shape '
                    '(self.input_size,), instead of {}.'.format(bounds))
            self._upper_bounds = bounds
    
    @property
    def nstep(self):
        return len(self._module_list)
    
    def from_original(self, xx, out=False):
        _return = False
        if out is None:
            out = xx
        if out is False:
            out = np.empty_like(xx)
            _return = True
        if not isinstance(out, np.ndarray):
            raise ValueError('invalid value for out.')
        if xx.shape != out.shape:
            raise ValueError('xx and out have different shapes.')
        if xx.ndim == 1:
            _from_original_f(xx, self._input_ranges, out, self._lower_bounds,
                             self._upper_bounds, xx.shape[0])
        elif xx.ndim == 2:
            _from_original_f2(xx, self._input_ranges, out, self._lower_bounds,
                              self._upper_bounds, xx.shape[1], xx.shape[0])
        else:
            raise ValueError('xx should be 1-d or 2-d.')
        if _return:
            return out
    
    def to_original(self, xx, out=False):
        _return = False
        if out is None:
            out = xx
        if out is False:
            out = np.empty_like(xx)
            _return = True
        if not isinstance(out, np.ndarray):
            raise ValueError('invalid value for out.')
        if xx.shape != out.shape:
            raise ValueError('xx and out have different shapes.')
        if xx.ndim == 1:
            _to_original_f(xx, self._input_ranges, out, self._lower_bounds,
                           self._upper_bounds, xx.shape[0])
        elif xx.ndim == 2:
            _to_original_f2(xx, self._input_ranges, out, self._lower_bounds,
                            self._upper_bounds, xx.shape[1], xx.shape[0])
        else:
            raise ValueError('xx should be 1-d or 2-d.')
        if _return:
            return out
    
    def to_original_grad(self, xx, out=False):
        _return = False
        if out is None:
            out = xx
        if out is False:
            out = np.empty_like(xx)
            _return = True
        if not isinstance(out, np.ndarray):
            raise ValueError('invalid value for out.')
        if xx.shape != out.shape:
            raise ValueError('xx and out have different shapes.')
        if xx.ndim == 1:
            _to_original_j(xx, self._input_ranges, out, self._lower_bounds,
                           self._upper_bounds, xx.shape[0])
        elif xx.ndim == 2:
            raise NotImplementedError
        else:
            raise ValueError('xx should be 1-d.')
        if _return:
            return out
    
    def to_original_grad2(self, xx, out=False):
        _return = False
        if out is None:
            out = xx
        if out is False:
            out = np.empty_like(xx)
            _return = True
        if not isinstance(out, np.ndarray):
            raise ValueError('invalid value for out.')
        if xx.shape != out.shape:
            raise ValueError('xx and out have different shapes.')
        if xx.ndim == 1:
            _to_original_jj(xx, self._input_ranges, out, self._lower_bounds,
                            self._upper_bounds, xx.shape[0])
        elif xx.ndim == 2:
            raise NotImplementedError
        else:
            raise ValueError('xx should be 1-d.')
        if _return:
            return out
    
    def summary(self):
        raise NotImplementedError


class _DensityBase:
    pass
        

class Density(Pipeline, _DensityBase):
    
    def __init__(self, density_name='__var__', *args, **kwargs):
        self.density_name = density_name
        super().__init__(*args, **kwargs)
        self._use_bound = False
        self._alpha = None
        self._gamma = None
    
    @property
    def density_name(self):
        return self._density_name
    
    @density_name.setter
    def density_name(self, name):
        if isinstance(name, str):
            self._density_name = name
        else:
            raise ValueError(
                'density_name should be a str, instead of {}'.format(name))
    
    def logp(self, xx, surrogate=False, original=True):
        _logp = float(
            self.fun(xx, surrogate, original=original)[self.density_name])
        if self._use_bound:
            if original:
                xx_bound = xx
            else:
                xx_bound = self.to_original(xx)
            beta2 = np.dot(np.dot(xx_bound - self._mu, self._hess), xx_bound - 
                           self._mu)
            if beta2 > self._alpha**2:
                _logp += -self._gamma * (beta2 - self._alpha**2)
        if not original:
            _logp += np.sum(np.log(np.abs(self.to_original_grad(xx, False))))
        return _logp
    
    def grad(self, xx, surrogate=False, original=True):
        _grad = self.jac(xx, surrogate, original=original)[self.density_name][0]
        if self._use_bound:
            beta2 = np.dot(np.dot(xx - self._mu, self._hess), xx - self._mu)
            if beta2 > self._alpha**2:
                _grad += -self._gamma * np.dot(xx - self._mu, self._hess) * 2
        if not original:
            _grad += (self.to_original_grad2(xx, False) / 
                      self.to_original_grad(xx, False))
        return _grad
    
    def logp_and_grad(self, xx, surrogate=False, original=True):
        _fun_and_jac_return = self.fun_and_jac(xx, surrogate, original=original)
        _logp = float(_fun_and_jac_return[0][self.density_name])
        _grad = _fun_and_jac_return[1][self.density_name][0]
        if self._use_bound:
            beta2 = np.dot(np.dot(xx - self._mu, self._hess), xx - self._mu)
            if beta2 > self._alpha**2:
                _logp += -self._gamma * (beta2 - self._alpha**2)
                _grad += -self._gamma * np.dot(xx - self._mu, self._hess) * 2
        if not original:
            _logp += np.sum(np.log(np.abs(self.to_original_grad(xx, False))))
            _grad += (self.to_original_grad2(xx, False) / 
                      self.to_original_grad(xx, False))
        return _logp, _grad
    
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, a):
        try:
            a = float(a)
            assert a > 0
        except:
            raise ValueError('alpha should be a positive float.')
        self._alpha = a
    
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, g):
        try:
            g = float(g)
            assert g > 0
        except:
            raise ValueError('gamma should be a positive float.')
        self._gamma = g
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def hess(self):
        return self._hess
    
    @property
    def use_bound(self):
        return self._use_bound
    
    @use_bound.setter
    def use_bound(self, ub):
        self._use_bound = bool(ub)
    
    def set_bound(self, xx, alpha=None, alpha_p=0.99, gamma=None):
        self._use_bound = True
        if not (xx.ndim == 2 and xx.shape[-1] == self._input_size):
            raise ValueError(
                'xx should be a 2-d array, with shape (# of points, {}), '
                'instead of {}.'.format(self._input_size, xx.shape))
        self._mu = np.mean(xx, axis=0)
        self._hess = np.linalg.inv(np.cov(xx, rowvar=False))
        if alpha is None:
            alpha_p = float(alpha_p)
            if alpha_p <= 0:
                raise ValueError('alpha_p should be a positive float.')
            _betas = np.einsum('ij,jk,ik->i', xx - self._mu, self._hess, 
                               xx - self._mu)
            self.alpha = np.percentile(_betas, alpha_p)
        else:
            self.alpha = alpha
        if gamma is None:
            if self._gamma is None:
                self._gamma = 0.05
        else:
            self.gamma = gamma
        self._f
    
    __call__ = logp
    

def DensityLite(logp=None, grad=None, logp_and_grad=None, dim=None, 
                logp_args=(), logp_kwargs={}, grad_args=(), 
                grad_kwargs={}, logp_and_grad_args=(), 
                logp_and_grad_kwargs={}, input_ranges=None, 
                lower_bounds=False, upper_bounds=False):
    if callable(logp):
        _fun = lambda xx, *args, **kwargs: logp(xx, *args, **kwargs)
    else:
        _fun = None
    if callable(grad):
        _jac = lambda xx, *args, **kwargs: [grad(xx, *args, **kwargs)]
    else:
        _jac = None
    if callable(logp_and_grad):
        def _fun_and_jac(xx, *args, **kwargs):
            ff, jj = logp_and_grad(xx, *args, **kwargs)
            return ff, [jj]
    else:
        _fun_and_jac = None
    logp_module = Module(fun=_fun, jac=_jac, fun_and_jac=_fun_and_jac, 
                         input_names=['__var__'], output_names=['__var__'], 
                         label='logp', concatenate_input=True, 
                         fun_args=logp_args, fun_kwargs=logp_kwargs, 
                         jac_args=grad_args, jac_kwargs=grad_kwargs, 
                         fun_and_jac_args=logp_and_grad_args, 
                         fun_and_jac_kwargs=logp_and_grad_kwargs)
    logp_density = Density(density_name='__var__', module_list=[logp_module], 
                           input_names=['__var__'], input_dims=[dim], 
                           input_ranges=input_ranges, lower_bounds=lower_bounds, 
                           upper_bounds=upper_bounds)
    return logp_density
