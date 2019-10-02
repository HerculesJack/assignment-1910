cimport cython
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8
from libc.math cimport log, exp


__all__ = ['_from_original_f', '_from_original_f2', '_to_original_f', 
           '_to_original_f2', '_to_original_j', '_to_original_jj']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _from_original_f(const double[::1] x, const double[:, ::1] ranges, 
                     double[::1] out_f, const uint8[::1] lower_bounds, 
                     const uint8[::1] upper_bounds, const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = (x[i] - ranges[i, 0]) / (ranges[i, 1] - ranges[i, 0])
        if lower_bounds[i] and upper_bounds[i]:
            if tmp <= 0. or tmp >= 1.:
                raise ValueError('variable #{} out of bound.'.format(i))
            tmp = log(tmp / (1 - tmp))
        elif lower_bounds[i] and (not upper_bounds[i]):
            if tmp <= 0.:
                raise ValueError('variable #{} our of bound.'.format(i))
            tmp = log(tmp)
        elif (not lower_bounds[i]) and upper_bounds[i]:
            if tmp >= 1.:
                raise ValueError('variable #{} our of bound.'.format(i))
            tmp = log(1 - tmp)
        out_f[i] = tmp

        
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _from_original_f2(const double[:, ::1] x, const double[:, ::1] ranges, 
                      double[:, ::1] out_f, const uint8[::1] lower_bounds, 
                      const uint8[::1] upper_bounds, const size_t n, 
                      const size_t m):
    cdef size_t i
    for i in range(m):
        _from_original_f(x[i], ranges, out_f[i], lower_bounds, upper_bounds, n)
        

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_f(const double[::1] x, const double[:, ::1] ranges, 
                   double[::1] out_f, const uint8[::1] lower_bounds, 
                   const uint8[::1] upper_bounds, const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = x[i]
        if lower_bounds[i] and upper_bounds[i]:
            tmp = 1 / (1 + exp(-tmp))
        elif lower_bounds[i] and (not upper_bounds[i]):
            tmp = exp(tmp)
        elif (not lower_bounds[i]) and upper_bounds[i]:
            tmp = 1 - exp(tmp)
        tmp = ranges[i, 0] + tmp * (ranges[i, 1] - ranges[i, 0])
        out_f[i] = tmp


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_f2(const double[:, ::1] x, const double[:, ::1] ranges, 
                    double[:, ::1] out_f, const uint8[::1] lower_bounds, 
                    const uint8[::1] upper_bounds, const size_t n, 
                    const size_t m):
    cdef size_t i
    for i in range(m):
        _to_original_f(x[i], ranges, out_f[i], lower_bounds, upper_bounds, n)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_j(const double[::1] x, const double[:, ::1] ranges, 
                   double[::1] out_j, const uint8[::1] lower_bounds, 
                   const uint8[::1] upper_bounds, const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = x[i]
        if lower_bounds[i] and upper_bounds[i]:
            tmp = 1 / (1 + exp(-tmp))
            tmp = tmp * (1 - tmp)
        elif lower_bounds[i] and (not upper_bounds[i]):
            tmp = exp(tmp)
        elif (not lower_bounds[i]) and upper_bounds[i]:
            tmp = -exp(tmp)
        else:
            tmp = 1.
        tmp *= (ranges[i, 1] - ranges[i, 0])
        out_j[i] = tmp


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_jj(const double[::1] x, const double[:, ::1] ranges, 
                    double[::1] out_j, const uint8[::1] lower_bounds, 
                    const uint8[::1] upper_bounds, const size_t n):
    cdef size_t i
    cdef double tmp, tmp2
    for i in range(n):
        tmp = x[i]
        if lower_bounds[i] and upper_bounds[i]:
            tmp2 = exp(tmp)
            tmp = -tmp2 * (tmp2 - 1) / (tmp2 + 1) / (tmp2 + 1) / (tmp2 + 1)
        elif lower_bounds[i] and (not upper_bounds[i]):
            tmp = exp(tmp)
        elif (not lower_bounds[i]) and upper_bounds[i]:
            tmp = -exp(tmp)
        else:
            tmp = 0.
        tmp *= (ranges[i, 1] - ranges[i, 0])
        out_j[i] = tmp
        