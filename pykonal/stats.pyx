cimport numpy as np

from . cimport constants

from libc.math cimport log, pi, sqrt

cdef class NormalDistribution(object):

    def __init__(self, mu: constants.REAL_t=0, sigma: constants.REAL_t=1):
        self.cy_mu = mu
        self.cy_sigma = sigma

    @property
    def mu(self):
        return (self.cy_mu)

    @property
    def sigma(self):
        return (self.cy_sigma)

    
    cpdef constants.REAL_t logpdf(NormalDistribution self, constants.REAL_t x):
        return (
            log(1 / (self.cy_sigma * sqrt(2 * pi))) 
            - (((x - self.cy_mu) / self.cy_sigma) ** 2) / 2
        )
