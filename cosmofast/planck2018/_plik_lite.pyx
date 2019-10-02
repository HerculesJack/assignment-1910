cimport cython
from libc.stdlib cimport malloc, free
#from libc.math cimport log
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _quad_sym(const double[:, ::1] A, const double* x, 
                      const int m) nogil:
    cdef size_t i, j
    cdef double xAx
    xAx = 0.
    for i in prange(m, nogil=True):
    #for i in range(m):
        xAx += A[i, i] * x[i] * x[i]
        for j in range(i + 1, m):
            xAx += 2 * A[i, j] * x[i] * x[j]
    return xAx


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _mat_vec(const double[:, ::1] A, const double* x, double* Ax, 
                   const int m, double alpha=1.) nogil:
    cdef size_t i, j
    for i in prange(m, nogil=True, schedule='static'):
        Ax[i] = 0.
        for j in range(m):
            Ax[i] += A[i, j] * x[j]
        Ax[i] *= alpha


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _get_binned_cls(const double[::1] cls, double[::1] cls_b, 
                    const double[::1] weight, const int[::1] bin_m, 
                    const int[::1] bin_w, const size_t n_bin):
    cdef size_t i, j
    for i in range(n_bin):
        cls_b[i] = 0
        for j in range(bin_m[i], bin_m[i] + bin_w[i]):
            cls_b[i] += cls[j] * weight[j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_f(const double[::1] cls_b, const double ap, double[::1] out_f, 
                 const double[::1] X_data, const double[:, ::1] cov_inv, 
                 const size_t n_bin):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i
    try:
        #for i in prange(n_bin, nogil=True, schedule='static'):
        for i in range(n_bin):
            y0[i] = cls_b[i] / ap2
            y0[i] -= X_data[i]
        out_f[0] = -0.5 * _quad_sym(cov_inv, y0, n_bin)
    finally:
        free(y0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_j(const double[::1] cls_b, const double ap, double[:, ::1] out_j, 
                 const double[::1] X_data, const double[:, ::1] cov_inv, 
                 const size_t n_bin):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i
    try:
        for i in range(n_bin):
            y0[i] = cls_b[i] / ap2
            y0[i] -= X_data[i]
        _mat_vec(cov_inv, y0, &out_j[0, 0], n_bin, -1)
        out_j[0, n_bin] = 0.
        for i in range(n_bin):
            out_j[0, i] /= ap2
            out_j[0, n_bin] -= 2 * cls_b[i] / ap * out_j[0, i]
    finally:
        free(y0)
        

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_fj(const double[::1] cls_b, const double ap, double[::1] out_f, 
                  double[:, ::1] out_j, const double[::1] X_data, 
                  const double[:, ::1] cov_inv, const size_t n_bin):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i
    try:
        #for i in prange(n_bin, nogil=True, schedule='static'):
        for i in range(n_bin):
            y0[i] = cls_b[i] / ap2
            y0[i] -= X_data[i]
        _mat_vec(cov_inv, y0, &out_j[0, 0], n_bin, -1)
        out_f[0] = 0.
        out_j[0, n_bin] = 0.
        for i in range(n_bin):
            out_f[0] += 0.5 * out_j[0, i] * y0[i]
            out_j[0, i] /= ap2
            out_j[0, n_bin] -= 2 * cls_b[i] / ap * out_j[0, i]
    finally:
        free(y0)
        