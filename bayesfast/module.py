import numpy as np
import copy
from collections import namedtuple

__all__ = ['Module', 'SurrogateIndices', 'Surrogate']


class Module:
    
    def __init__(self, fun=None, jac=None, fun_and_jac=None, 
                 input_names=['__var__'], output_names=['__var__'], 
                 delete_names=None, label=None, concatenate_input=True, 
                 fun_args=(), fun_kwargs={}, jac_args=(), jac_kwargs={}, 
                 fun_and_jac_args=(), fun_and_jac_kwargs={}):
        self._fun_jac_init(fun, jac, fun_and_jac)
        self.input_names = input_names
        self.output_names = output_names
        self.delete_names = delete_names
        self.label = label
        self.concatenate_input = concatenate_input
        self.fun_args = fun_args
        self.fun_kwargs = fun_kwargs
        self.jac_args = jac_args
        self.jac_kwargs = jac_kwargs
        self.fun_and_jac_args = fun_and_jac_args
        self.fun_and_jac_kwargs = fun_and_jac_kwargs
        self._ncall_fun_and_jac = 0
        self._ncall_fun = 0
        self._ncall_jac = 0
    
    def _fun_jac_init(self, fun, jac, fun_and_jac):
        self.fun = fun
        self.jac = jac
        self.fun_and_jac = fun_and_jac
    
    def input(self, *args):
        if self._concatenate_input:
            if len(self._input_names) == 1:
                return [np.atleast_1d(args[0])]
            else:
                return [np.concatenate(args, axis=0)]
        else:
            return args
    
    @property
    def fun(self):
        if self.has_fun:
            self._ncall_fun += 1
            return lambda *args: self._fun(*self.input(*args))
        elif self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return lambda *args: self._fun_and_jac(*self.input(*args))[0]
        else:
            raise RuntimeError('No valid definition of fun is found.')
    
    @fun.setter
    def fun(self, function):
        if callable(function):
            self._fun = lambda *args: function(*args, *self._fun_args, 
                                               **self._fun_kwargs)
        elif function is None:
            self._fun = None
        else:
            raise ValueError('fun should be callable, or None if you want to '
                             'reset it.')
            
    @property
    def has_fun(self):
        return (self._fun is not None)
            
    __call__ = fun
    
    @property
    def jac(self):
        if self.has_jac:
            self._ncall_jac += 1
            return lambda *args: self._jac(*self.input(*args))
        elif self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return lambda *args: self._fun_and_jac(*self.input(*args))[1]
        else:
            raise RuntimeError('No valid definition of jac is found.')
    
    @jac.setter
    def jac(self, jacobian):
        if callable(jacobian):
            self._jac = lambda *args: jacobian(*args, *self._jac_args, 
                                               **self._jac_kwargs)
        elif jacobian is None:
            self._jac = None
        else:
            raise ValueError('jac should be callable, or None if you want to '
                             'reset it.')
            
    @property
    def has_jac(self):
        return (self._jac is not None)
    
    @property
    def fun_and_jac(self):
        if self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return lambda *args: self._fun_and_jac(*self.input(*args))
        elif self.has_fun and self.has_jac:
            self._ncall_fun += 1
            self._ncall_jac += 1
            return lambda *args: (self._fun(*self.input(*args)), 
                                  self._jac(*self.input(*args)))
        else:
            raise RuntimeError('No valid definition of fun_and_jac is found.')
    
    @fun_and_jac.setter
    def fun_and_jac(self, fun_jac):
        if callable(fun_jac):
            self._fun_and_jac = lambda *args: fun_jac(
                *args, *self.fun_and_jac_args, **self.fun_and_jac_kwargs)
        elif fun_jac is None:
            self._fun_and_jac = None
        else:
            raise ValueError('fun_and_jac should be callable, or None if you '
                             'want to reset it.')
    
    @property
    def has_fun_and_jac(self):
        return (self._fun_and_jac is not None)
    
    @property
    def ncall_fun(self):
        return self._ncall_fun
    
    @property
    def ncall_jac(self):
        return self._ncall_jac
    
    @property
    def ncall_fun_and_jac(self):
        return self._ncall_fun_and_jac
    
    @property
    def input_names(self):
        return copy.copy(self._input_names)
    
    @input_names.setter
    def input_names(self, name):
        if isinstance(name, str):
            name = [name]
        else:
            try:
                name = list(name)
                assert all(isinstance(nn, str) for nn in name)
                assert len(name) > 0
            except:
                raise ValueError('input_names should be a list of str(s), '
                                 'instead of {}'.format(name))
            if len(name) != len(set(name)):
                raise ValueError('input_names should be a list of unique '
                                 'name(s), instead of {}'.format(name))
        self._input_names = name
        
    @property
    def output_names(self):
        return copy.copy(self._output_names)
    
    @output_names.setter
    def output_names(self, name):
        if isinstance(name, str):
            name = [name]
        else:
            try:
                name = list(name)
                assert all(isinstance(nn, str) for nn in name)
                assert len(name) > 0
            except:
                raise ValueError('output_names should be a list of str(s), '
                                 'instead of {}'.format(name))
            if len(name) != len(set(name)):
                raise ValueError('output_names should be a list of unique '
                                 'name(s), instead of {}'.format(name))
        self._output_names = name
    
    @property
    def delete_names(self):
        return copy.copy(self._delete_names)
    
    @delete_names.setter
    def delete_names(self, name):
        if name is None:
            name = []
        else:
            if isinstance(name, str):
                name = [name]
            else:
                try:
                    name = list(name)
                    assert all(isinstance(nn, str) for nn in name)
                except:
                    raise ValueError(
                        'delete_names should be a list of str(s), or None, '
                        'instead of {}'.format(name))
                if len(name) != len(set(name)):
                    raise ValueError('delete_names should be a list of unique '
                                     'name(s), instead of {}'.format(name))
        self._delete_names = name
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, tag):
        if isinstance(tag, str) or tag is None:
            self._label = tag
        else:
            raise ValueError(
            'label should be a str or None, instead of {}.'.format(tag))
    
    @property
    def concatenate_input(self):
        return self._concatenate_input
    
    @concatenate_input.setter
    def concatenate_input(self, ci):
        self._concatenate_input = bool(ci)
    
    @property
    def fun_args(self):
        return copy.copy(self._fun_args)
    
    @fun_args.setter
    def fun_args(self, fargs):
        try:
            self._fun_args = tuple(fargs)
        except:
            self._fun_args = ()
            raise ValueError('fun_args should be a tuple.')
            
    @property
    def fun_kwargs(self):
        return copy.copy(self._fun_kwargs)
    
    @fun_kwargs.setter
    def fun_kwargs(self, fkwargs):
        try:
            self._fun_kwargs = dict(fkwargs)
        except:
            self._fun_kwargs = {}
            raise ValueError('fun_kwargs should be a dict.')
            
    @property
    def jac_args(self):
        return copy.copy(self._jac_args)
    
    @jac_args.setter
    def jac_args(self, jargs):
        try:
            self._jac_args = tuple(jargs)
        except:
            self._jac_args = ()
            raise ValueError('jac_args should be a tuple.')
            
    @property
    def jac_kwargs(self):
        return copy.copy(self._jac_kwargs)
    
    @jac_kwargs.setter
    def jac_kwargs(self, jkwargs):
        try:
            self._jac_kwargs = dict(jkwargs)
        except:
            self._jac_kwargs = {}
            raise ValueError('jac_kwargs should be a dict.')
            
    @property
    def fun_and_jac_args(self):
        return copy.copy(self._fun_and_jac_args)
    
    @fun_and_jac_args.setter
    def fun_and_jac_args(self, fjargs):
        try:
            self._fun_and_jac_args = tuple(fjargs)
        except:
            self._fun_and_jac_args = ()
            raise ValueError('fun_and_jac_args should be a tuple.')
            
    @property
    def fun_and_jac_kwargs(self):
        return copy.copy(self._fun_and_jac_kwargs)
    
    @fun_and_jac_kwargs.setter
    def fun_and_jac_kwargs(self, fjkwargs):
        try:
            self._fun_and_jac_kwargs = dict(fjkwargs)
        except:
            self._fun_and_jac_kwargs = {}
            raise ValueError('fun_and_jac_kwargs should be a dict.')
    
    def summary(self):
        raise NotImplementedError


SurrogateIndices = namedtuple('SurrogateIndices', ['start', 'length'])


class Surrogate(Module):
    
    def __init__(self, indices, input_names, output_names, delete_names=None, 
                 label=None, concatenate_input=True):
        super().__init__(input_names=input_names, output_names=output_names,
                         delete_names=delete_names, label=label,
                         concatenate_input=concatenate_input)
        self.indices = indices
        if not hasattr(self, '_fun'):
            self._fun = None
        if not hasattr(self, '_jac'):
            self._jac = None
        if not hasattr(self, '_fun_and_jac'):
            sele._fun_and_jac = None
    
    def _fun_jac_init(self, fun, jac, fun_and_jac):
        pass
        
    @property
    def indices(self):
        return self._indices
    
    @indices.setter
    def indices(self, ii):
        if isinstance(ii, SurrogateIndices):
            self._indices = ii
        else:
            raise ValueError('indices should be a SurrogateIndices.')
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError('Abstract Method.')
