import types
import numpy as np
import pandas as pd
from . import PACKAGEDIR

import astropy.constants as const
import astropy.units as u

import theano.tensor as tt


import exoplanet as xo

ld_table = pd.read_csv('{}/data/limb_darkening.csv'.format(PACKAGEDIR), comment='#')

class ShineException(Exception):
    '''Raised when there is a shiney error.'''
    pass

def make_method(parameter):
    str = 'def {0}(self):\n\treturn xo.eval_in_model(self._model.{0}, model=self._model)'
    exec(str.format(parameter))
    return locals()['{}'.format(parameter)]

def add_method(self, method, name):
     self.__dict__[name] = types.MethodType(method, self)

def blackbody(lam, temp):
    # Lam in nm
    # Temp in K
    a1 = (const.h * const.c/const.k_B).value * 1e9
    log_boltz = a1 / (lam * temp)
    boltzm1 = np.exp(log_boltz) - 1
    a2 = (8 * np.pi * const.h * (const.c**2)).to(u.W*u.m**2).value * (1/boltzm1)
    bb_lam = a2 / lam**5
    return bb_lam



def blackbody_theano(lam, temp):
    # Lam in nm
    # Temp in K
    a1 = (const.h * const.c/const.k_B).value * 1e9
    log_boltz = a1 / (lam * temp)
    boltzm1 = tt.exp(log_boltz) - 1
    a2 = (8 * np.pi * const.h * (const.c**2)).to(u.W*u.m**2).value * (1/boltzm1)
    bb_lam = a2 / lam**5
    return bb_lam

def elipsoidal(t, P, t0, A):
    return -A * np.cos((2 * np.pi * (t-t0))/(0.5 * P))

def doppler(t, P, t0, A):
    return A * np.sin((2 * np.pi * (t-t0))/(P))

def reflection(t, albedo, radius, r_star, a, period, t0):
    A = albedo * (((radius*r_star*u.solRad)**2)/((a*u.AU)**2).to(u.solRad**2)).value
    return -A * np.cos((2 * np.pi * (t-t0))/(period))

def equilibrium_temp(r_star, a, teff):
    a1 = (((r_star*u.solRad))/((a*u.AU)).to(u.solRad)).value
    return teff * a1**0.5 * (0.25 * (1-(albedo*2/3)))**(1/4)


def flux_in_bandpass(albedo, r_star, a, teff):
    teq = equilibrium_temp(r_star, a, teff)
    f_in_bp = np.sum(blackbody_lambda(bandpass[:, 0]*u.nm, teq*u.K).value * bandpass[:, 1])
    f_in_bp /= norm
    return f_in_bp

def thermal(t, albedo, r_star, a, teff, period, t0, phase_shift):
    f_in_bp = flux_in_bandpass(albedo, r_star, a, teff)
    return -f_in_bp * np.cos((2 * np.pi * (t-t0))/(period) + phase_shift)
