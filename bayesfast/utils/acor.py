# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
#
#  Adapted from emcee, just changed the way to interpret x in integrated_time.
#
#  URL: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
#
#-------------------------------------------------------------------------------


from __future__ import division, print_function

import logging

import numpy as np

#__all__ = ["function_1d", "integrated_time", "AutocorrError"]
__all__ = ["integrated_time"]


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def integrated_time(x, c=5, tol=50, quiet=True):
    """Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.
    Args:
        x: The time series of shape (n_chain, n_step, n_dim). If 1-d, 
           interpreted as n_chain = 1 and n_dim = 1. If 2-d, interpreted as 
           n_chain = 1.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.
    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[np.newaxis, :, np.newaxis]
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    #n_t, n_w, n_d = x.shape
    n_w, n_t, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            #f += function_1d(x[:, k, d])
            f += function_1d(x[k, :, d])
        f /= n_w
        taus = 2.0*np.cumsum(f)-1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t/tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est, msg)
        logging.warning(msg)

    return tau_est


class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.
    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.
    """
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super(AutocorrError, self).__init__(*args, **kwargs)
