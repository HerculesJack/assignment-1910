import copy
from ..module import *
from ..density import *
from ._poly import *
from scipy.linalg import lstsq

__all__ = ['PolyConfig', 'PolyModel']


class PolyConfig:
    
    def __init__(self, order, input_mask, output_mask, coef=None):
        if order in ('linear', 'quad', 'cubic_2', 'cubic_3'):
            self._order = order
        else:
            raise ValueError(
                'order should be one of ("linear", "quad", "cubic_2", '
                '"cubic_3"), instead of "{}".'.format(order))
        self._input_mask = np.sort(np.unique(np.asarray(input_mask, 
                                                        dtype=np.int)))
        self._output_mask = np.sort(np.unique(np.asarray(output_mask, 
                                                         dtype=np.int)))
        self.coef = coef
    
    @property
    def order(self):
        return self._order
    
    @property
    def input_mask(self):
        return np.copy(self._input_mask)
    
    @property
    def output_mask(self):
        return np.copy(self._output_mask)
    
    @property
    def input_size(self):
        return self._input_mask.size
    
    @property
    def output_size(self):
        return self._output_mask.size
    
    @property
    def _A_shape(self):
        if self._order == 'linear':
            return (self.output_size, self.input_size + 1)
        elif self._order == 'quad':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic_2':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic_3':
            return (self.output_size, self.input_size, self.input_size, 
                    self.input_size)
        else:
            raise RuntimeError(
                'unexpected value of self.order "{}".'.format(self._order))
    
    @property
    def _a_shape(self):
        if self._order == 'linear':
            return (self.input_size + 1,)
        elif self._order == 'quad':
            return (self.input_size * (self.input_size + 1) // 2,)
        elif self._order == 'cubic_2':
            return (self.input_size * self.input_size,)
        elif self._order == 'cubic_3':
            return (self.input_size * (self.input_size - 1) * 
                    (self.input_size - 2) // 6,)
        else:
            raise RuntimeError(
                'unexpected value of self.order "{}".'.format(self._order))
    
    @property
    def coef(self):
        return self._coef
    
    @coef.setter
    def coef(self, A):
        if A is not None:
            if A.shape != self._A_shape:
                raise ValueError(
                    'shape of the coef matrix {} does not match the expected '
                    'shape {}.'.format(A.shape, self._A_shape))
            self._coef = np.copy(A)
        else:
            self._coef = None
            
    def set(self, a, ii):
        ii = int(ii)
        if a.shape != self._a_shape:
            raise ValueError('shape of a {} does not match the expected shape '
                             '{}.'.format(a.shape, self._a_shape))
        if not 0 <= ii <= self.output_size:
            raise ValueError('ii = {} out of range for self.output_size = '
                             '{}.'.format(ii, self.output_size))
        if self._order == 'linear':
            coefii = a
        else:
            coefii = np.empty(self._A_shape[1:])
            if self._order == 'quad':
                _set_quad(a, coefii, self.input_size)
            elif self._order == 'cubic_2':
                _set_cubic_2(a, coefii, self.input_size)
            elif self._order == 'cubic_3':
                _set_cubic_3(a, coefii, self.input_size)
            else:
                raise RuntimeError(
                    'unexpected value of self.order "{}".'.format(self._order))
        if self._coef is None:
            self._coef = np.zeros(self._A_shape)
        self._coef[ii] = coefii


class PolyModel(Surrogate):
    
    def __init__(self, configs, indices, input_size, output_size, 
                 input_names=['__var__'], output_names=['__var__'], 
                 delete_names=None, label=None):
        super().__init__(indices, input_names, output_names, delete_names, 
                         label, True)
        if isinstance(configs, PolyConfig):
            self._configs = [configs]
        elif hasattr(configs, '__iter__'):
            self._configs = []
            for conf in configs:
                if isinstance(conf, PolyConfig):
                    self._configs.append(conf)
                else:
                    raise ValueError(
                        'not all the element(s) in configs are PolyConfig(s).')
        else:
            raise ValueError(
                'configs should be a PolyConfig, or a list of PolyConfig(s).')
        try:
            self._input_size = int(input_size)
        except:
            raise ValueError('input_size should be an int.')
        try:
            self._output_size = int(output_size)
        except:
            raise ValueError('output_size should be an int.')
        self._build_recipe()
        self._use_bound = False
        self._alpha = None
        
    @property
    def configs(self):
        return copy.copy(self._configs)
    
    @property
    def n_config(self):
        return len(self._config)
    
    @property
    def input_size(self):
        return self._input_size
    
    @property
    def output_size(self):
        return self._output_size
    
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, a):
        try:
            a = float(a)
            assert a > 0
            self._alpha = a
        except:
            raise ValueError('alpha should be a positive float.')
    
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
    
    def set_bound(self, xx, alpha=None, alpha_p=99):
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
                               xx - self._mu)**0.5
            self.alpha = np.percentile(_betas, alpha_p)
        else:
            self.alpha = alpha
        self._f_mu = self._fun(self._mu)
    
    @property
    def recipe(self):
        return np.copy(self._recipe)
    
    def _build_recipe(self):
        rr = np.full((self._output_size, 4), -1)
        for ii, conf in enumerate(self._configs):
            if conf.order == 'linear':
                if np.any(rr[conf._output_mask, 0] >= 0):
                    raise ValueError(
                        'multiple linear PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 0] = ii
            elif conf.order == 'quad':
                if np.any(rr[conf._output_mask, 1] >= 0):
                    raise ValueError(
                        'multiple quad PolyConfig(s) share at least one common '
                        'output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 1] = ii
            elif conf.order == 'cubic_2':
                if np.any(rr[conf._output_mask, 2] >= 0):
                    raise ValueError(
                        'multiple cubic_2 PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 2] = ii
            elif conf.order == 'cubic_3':
                if np.any(rr[conf._output_mask, 3] >= 0):
                    raise ValueError(
                        'multiple cubic_3 PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 3] = ii
            else:
                raise RuntimeError('unexpected value of conf.order for '
                                   'PolyConfig #{}.'.format(ii))
        if np.any(np.all(rr < 0, axis=1)):
            raise ValueError(
                'no PolyConfig has output for variable(s) {}.'.format(
                np.argwhere(np.any(np.all(rr < 0, axis=1))).flatten()))
        self._recipe = rr
    
    @classmethod
    def _linear(cls, config, xx_in, target):
        if target == 'fun':
            return np.dot(config.coef[:, 1:], xx_in) + config.coef[:, 0]
        elif target == 'jac':
            return config.coef[:, 1:]
        elif target == 'fun_and_jac':
            ff = np.dot(config.coef[:, 1:], xx_in) + config.coef[:, 0]
            jj = config.coef[:, 1:]
            return ff, jj
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))
    
    @classmethod
    def _quad(cls, config, xx_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _quad_f(xx_in, config.coef, out_f, config.output_size, 
                    config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _quad_j(xx_in, config.coef, out_j, config.output_size, 
                    config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _quad_f(xx_in, config.coef, out_f, config.output_size, 
                    config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _quad_j(xx_in, config.coef, out_j, config.output_size, 
                    config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))
    
    @classmethod
    def _cubic_2(cls, config, xx_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _cubic_2_f(xx_in, config.coef, out_f, config.output_size, 
                       config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_2_j(xx_in, config.coef, out_j, config.output_size, 
                       config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _cubic_2_f(xx_in, config.coef, out_f, config.output_size, 
                       config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_2_j(xx_in, config.coef, out_j, config.output_size, 
                       config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))
            
    @classmethod
    def _cubic_3(cls, config, xx_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _cubic_3_f(xx_in, config.coef, out_f, config.output_size, 
                       config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_3_j(xx_in, config.coef, out_j, config.output_size, 
                       config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _cubic_3_f(xx_in, config.coef, out_f, config.output_size, 
                       config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_3_j(xx_in, config.coef, out_j, config.output_size, 
                       config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))
    
    @classmethod
    def _eval_one(cls, config, xx, target='fun'):
        xx_in = np.ascontiguousarray(xx[config.input_mask])
        if config.order == 'linear':
            return cls._linear(config, xx_in, target)
        elif config.order == 'quad':
            return cls._quad(config, xx_in, target)
        elif config.order == 'cubic_2':
            return cls._cubic_2(config, xx_in, target)
        elif config.order == 'cubic_3':
            return cls._cubic_3(config, xx_in, target)
        else:
            raise RuntimeError('unexpected value of config.order.')
    
    def _fun(self, xx):
        if not xx.shape == (self._input_size,):
            raise ValueError('shape of xx should be {}, instead of '
                             '{}.'.format((self._input_size,), xx.shape))
        if self._use_bound and np.dot(np.dot(xx - self._mu, self._hess), 
                                      xx - self._mu)**0.5 > self._alpha:
            return self._fj_bound(xx, 'fun')
        else:
            ff = np.zeros(self._output_size)
            for conf in self._configs:
                ff[conf._output_mask] += self._eval_one(conf, xx, 'fun')
            return ff
    
    def _jac(self, xx):
        if not xx.shape == (self._input_size,):
            raise ValueError('shape of xx should be {}, instead of '
                             '{}.'.format((self._input_size,), xx.shape))
        if self._use_bound and np.dot(np.dot(xx - self._mu, self._hess), 
                                      xx - self._mu)**0.5 > self._alpha:
            return self._fj_bound(xx, 'jac')            
        else:
            jj = np.zeros((self._output_size, self._input_size))
            for conf in self._configs:
                jj[conf._output_mask[:, np.newaxis], 
                   conf._input_mask] += self._eval_one(conf, xx, 'jac')
            return jj
    
    def _fun_and_jac(self, xx):
        if not xx.shape == (self._input_size,):
            raise ValueError('shape of xx should be {}, instead of '
                             '{}.'.format((self._input_size,), xx.shape))
        ff = np.zeros(self._output_size)
        jj = np.zeros((self._output_size, self._input_size))
        if self._use_bound and np.dot(np.dot(xx - self._mu, self._hess), 
                                      xx - self._mu)**0.5 > self._alpha:
            return self._fj_bound(xx, 'fun_and_jac')
        else:
            for conf in self._configs:
                _f, _j = self._eval_one(conf, xx, 'fun_and_jac')
                ff[conf._output_mask] += _f
                jj[conf._output_mask[:, np.newaxis], conf._input_mask] += _j
            return ff, jj
    
    def _fj_bound(self, xx, target='fun'):
        beta = np.dot(np.dot(xx - self._mu, self._hess), xx - self._mu)**0.5
        xx_0 = (self._alpha * xx + (beta - self._alpha) * self._mu) / beta
        ff_0 = np.zeros(self._output_size)
        for conf in self._configs:
            ff_0[conf._output_mask] += self._eval_one(conf, xx_0, 'fun')
        if target != 'jac':
            ff = (beta * ff_0 - (beta - self._alpha) * self._f_mu) / self._alpha
            if target == 'fun':
                return ff
        grad_beta = np.dot(self._hess, xx - self._mu) / beta
        jj_0 = np.zeros((self._output_size, self._input_size))
        for conf in self._configs:
            jj_0[conf._output_mask[:, np.newaxis], 
                 conf._input_mask] += self._eval_one(conf, xx_0, 'jac')
        jj = jj_0 + np.outer((ff_0 - self._f_mu) / self._alpha - 
                             np.dot(jj_0, xx - self._mu) / beta, grad_beta)
        if target == 'jac':
            return jj
        elif target == 'fun_and_jac':
            return ff, jj
        raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))
    
    def fit(self, xx, yy):
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        if not (xx.ndim == 2 and xx.shape[-1] == self._input_size):
            raise ValueError(
                'xx should be a 2-d array, with shape (# of points, # of '
                'input_size), instead of {}.'.format(xx.shape))
        if not (yy.ndim == 2 and yy.shape[-1] == self._output_size):
            raise ValueError(
                'yy should be a 2-d array, with shape (# of points, # of '
                'output_size), instead of {}.'.format(yy.shape))
        if not xx.shape[0] == yy.shape[0]:
            raise ValueError('xx and yy have different # of points.')
        for ii in range(self._output_size):
            A = np.empty((xx.shape[0], 0))
            jj_l, jj_q, jj_c2, jj_c3 = self._recipe[ii]
            kk = [0]
            if jj_l >= 0:
                _A = np.empty((xx.shape[0], self._configs[jj_l]._a_shape[0]))
                _A[:, 0] = 1
                _A[:, 1:] = xx
                kk.append(kk[-1] + self._configs[jj_l]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_q >= 0:
                _A = np.empty((xx.shape[0], self._configs[jj_q]._a_shape[0]))
                _lsq_quad(xx, _A, xx.shape[0], self._configs[jj_q].input_size)
                kk.append(kk[-1] + self._configs[jj_q]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_c2 >= 0:
                _A = np.empty((xx.shape[0], self._configs[jj_c2]._a_shape[0]))
                _lsq_cubic_2(xx, _A, xx.shape[0], 
                             self._configs[jj_c2].input_size)
                kk.append(kk[-1] + self._configs[jj_c2]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_c3 >= 0:
                _A = np.empty((xx.shape[0], self._configs[jj_c3]._a_shape[0]))
                _lsq_cubic_3(xx, _A, xx.shape[0], 
                             self._configs[jj_c3].input_size)
                kk.append(kk[-1] + self._configs[jj_c3]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            b = np.copy(yy[:, ii])
            lsq = lstsq(A, b)[0]
            pp = 0
            if jj_l >= 0:
                qq = int(np.argwhere(self._configs[jj_l]._output_mask == ii))
                self._configs[jj_l].set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_q >= 0:
                qq = int(np.argwhere(self._configs[jj_q]._output_mask == ii))
                self._configs[jj_q].set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_c2 >= 0:
                qq = int(np.argwhere(self._configs[jj_c2]._output_mask == ii))
                self._configs[jj_c2].set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_c3 >= 0:
                qq = int(np.argwhere(self._configs[jj_c3]._output_mask == ii))
                self._configs[jj_c3].set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
    