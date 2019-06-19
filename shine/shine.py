import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import exoplanet as xo


import astropy.units as u
from astropy.constants import G
from astropy.modeling.blackbody import blackbody_lambda

import lightkurve as lk

from .utils import *
from . import PACKAGEDIR


bandpass = np.loadtxt('{}/data/bandpass.dat'.format(PACKAGEDIR))
bandpass[:, 1] /= np.trapz(bandpass[:,1], bandpass[:,0])


default_bounds = {'rprs_error':(-0.1, 0.1),
                'period_error':(-0.0005, 0.0005),
                't0_error':(-0.00001, 0.00001),
                 'inclination_error':(-0.05, 0.05)}



class Star(object):
    '''Primary star class'''

    def __init__(self, radius=1, mass=1, temperature=5777,
                 radius_error=(-0.1, 0.1), mass_error=(-0.1, 0.1), temperature_error=(-500, 500)):
        self.radius = u.Quantity(radius, u.solRad)
        self.temperature = u.Quantity(temperature, u.K)
        self.mass = u.Quantity(mass, u.solMass)
        self.radius_error = radius_error
        self.temperature_error = temperature_error
        self.mass_error = mass_error

        t = ld_table[(ld_table.teff == (self.temperature.value)//250 * 250) & (ld_table.met == 0) & (ld_table.logg == 5)]
        if len(t) == 0:
            raise ShineException('Can not find limb darkening parameters. This should not happen. Please report this error.')
        self.limb_darkening = [t.iloc[0].u, t.iloc[0].a]
        self.norm = np.sum(blackbody(bandpass[:, 0], self.temperature.value) * bandpass[:, 1])

    def __repr__(self):
        return ('Star: {}, {}, {}'.format(self.radius, self.mass, self.temperature))

class Planet(object):
    '''Companion class
    '''

    def __init__(self, host=None, rprs=0.01, period=10, t0=0, inclination=0.5*np.pi, albedo=0.5,
                     rprs_error=None, period_error=None, t0_error=None, inclination_error=None):
        self.host = host
        self.rprs = rprs
        self.albedo = albedo
        self._init_rprs = rprs
        self.period = u.Quantity(period, u.day)
        self._init_period = period
        self.t0 = t0
        self.inclination = inclination
        self._init_inclination = inclination
        self.rprs_error = rprs_error
        self.period_error = period_error
        self.t0_error = t0_error
        self.inclination_error = inclination_error
        self._validate_errors()

    def _validate_errors(self):
            '''Ensure the bounds are physical'''
            for key in ['rprs_error', 'period_error', 't0_error', 'inclination_error']:
                if getattr(self,key) is None:
                    setattr(self, key, default_bounds[key])
                if ~np.isfinite(getattr(self,key)[0]):
                    setattr(self,  key, tuple([default_bounds[key][0], getattr(self, key)[1]]))
                if ~np.isfinite(getattr(self, key)[1]):
                    setattr(self,  key, tuple([getattr(self, key)[0], default_bounds[key][1]]))

            if self.rprs + self.rprs_error[0] < 0:
                self.rprs_error = tuple([-self._init_rprs, self.rprs_error[1]])

            if self.period.value + self.period_error[0] < 0:
                self.period_error = tuple([-self._init_period, self.period_error[1]])

            if self.inclination + self.inclination_error[1] > 90.:
                self.inclination_error = tuple([self.inclination_error[0], 90. - self._init_inclination])

    @property
    def a(self):
        return ((((self.period)**2 * G * (self.host.mass)/(4*np.pi**2))**(1/3))).to(u.AU)

    def __repr__(self):
            return ('Planet: RpRs {} , P {}, t0 {}'.format(self.rprs, self.period,self.t0))



class Model(object):

    def __init__(self, tpf, host, planet, aperture_mask=None):
        self.tpf = tpf
        if aperture_mask == None:
            self.aperture_mask = tpf.pipeline_mask
        else:
            self.aperture_mask = aperture_mask
        self.lc = tpf.to_lightcurve(aperture_mask=self.aperture_mask).remove_nans()
        self.tpf = self.tpf[np.in1d(self.tpf.time, self.lc.time)]
        if self.lc.time_format == 'bkjd':
            self.lc.time += 2454833
        self.design_matrix = self.tpf.to_corrector(aperture_mask=self.aperture_mask).create_design_matrix(pld_order=2)
        self.host = host
        self.planet = planet
        self._logs2_prior = self.lc.estimate_cdpp() * 1e-6 * np.median(self.lc.flux)
        self._diag = np.asarray(self.lc.flux_err**2, np.float64)
        self._model = self._build_model()


    def __repr__(self):
        s = self.host.__repr__()
        s += '\n\t{}'.format(self.planet.__repr__())
        return s


    def _build_model(self, gp_timescale_prior=10, fractional_prior_width=10):
        time = np.asarray(self.lc.time, np.float64)
        lc_flux = np.asarray(self.lc.flux, np.float64)
        lc_flux_err = np.asarray(self.lc.flux_err, np.float64)

        # Covariance matrix diagonal


        with pm.Model() as model:

            mean = pm.Normal("mean", mu=np.nanmean(lc_flux), sd=np.nanstd(lc_flux))

            # Star Priors
            # ---------------------
            M_star = pm.Normal("M_star", mu=self.host.mass.value, sd=self.host.mass_error[1])
            R_star = pm.Normal("R_star", mu=self.host.radius.value, sd=self.host.radius_error[1])
            T_star = pm.Normal("T_star", mu=self.host.temperature.value, sd=self.host.temperature_error[1])


            # EB Model
            # ---------------------
            rprs = pm.Normal("rprs", mu=self.planet.rprs, sd=self.planet.rprs_error[1])
            pm.Potential("rprs_prior", tt.switch(rprs > 0, 0, np.inf))

            logP = pm.Normal("logP", mu=np.log(self.planet.period.value), sd=0.01)
            period = pm.Deterministic("period", pm.math.exp(logP))

            t0 = pm.Normal("t0", mu=self.planet.t0, sd=self.planet.t0_error[1])
            r = pm.Deterministic("r", rprs * R_star)
            logr = pm.Deterministic("logr", tt.log(rprs * R_star))
            inclination = pm.Normal("inclination", mu=self.planet.inclination, sd=self.planet.inclination_error[1])
            pm.Potential("r_prior", -logr)
            albedo = pm.Uniform("albedo", lower=0, upper=1)

            # Transit
            transit_orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, incl=inclination)
            u = xo.distributions.QuadLimbDark("u", testval=np.asarray([0.4, 0.3]))
            transit = xo.StarryLightCurve(u).get_light_curve(
                                                    orbit=transit_orbit, r=r, t=time)
            transit = pm.math.sum(transit, axis=-1)


            # Secondary Eclipse
            eclipse_orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0 + period/2, incl=inclination)
            eclipse = xo.StarryLightCurve([0, 0]).get_light_curve(
                                        orbit=eclipse_orbit, r=r, t=time)
            eclipse = pm.math.sum(eclipse, axis=-1)


            # Reflection
            re = pm.Deterministic('re', albedo * (rprs/transit_orbit.a)**2)
            reflection = -re * tt.cos(2 * np.pi * ((time - t0) / period)) + re
            reflection += ((eclipse)/r**2) * (2 * re)# + (2 * re)


            # Thermal
            teq = pm.Deterministic('teq', T_star * tt.sqrt(0.5*(1/transit_orbit.a)))
            norm = pm.Deterministic("norm", tt.sum(blackbody_theano(bandpass[:, 0], teq) * bandpass[:, 1])/self.host.norm)
            thermal = ((eclipse)/r**2) * norm + norm
            dT = pm.Uniform('dT', lower=0, upper=1000)
            phase_shift = pm.Normal('phase_shift', mu=0, sd=0.1)
            A1 = ((dT)**4/teq**4)
            thermal_cosine = -(A1 * norm) * tt.cos(2 * np.pi * ((time - t0) / period) + phase_shift) - (A1 * norm)
            thermal += thermal_cosine
            pm.Deterministic('thermal_cosine', thermal_cosine)


            # Doppler
            dp = pm.Uniform('dp', lower=0, upper=0.0001)
            doppler = pm.Deterministic('doppler', dp * np.sin((2 * np.pi * (time-t0))/(period)))

            # Elipsoidal
            ep = pm.Uniform('ep', lower=0, upper=0.0001)
            elipsoidal = pm.Deterministic('elipsoidal', -ep * np.cos((2 * np.pi * (time-t0))/(0.5 * period)))

            # Build the light curve
            eb_model =  transit + reflection + thermal + doppler + elipsoidal
            eb_model = ((eb_model + 1)/(1 + norm + (2 * re)))
            eb_model *= mean




            # GP and Motion Fitting
            # ---------------------
            # Create a Gaussian Process to model the long-term stellar variability
            # log(sigma) is the amplitude of variability, estimated from the raw flux scatter
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(lc_flux)), sd=3)
            # log(rho) is the timescale of variability with a user-defined prior
            logrho = pm.Normal("logrho", mu=np.log(gp_timescale_prior),
                               sd=np.log(fractional_prior_width*gp_timescale_prior))
            # Enforce that the scale of variability should be no shorter than 0.5 days
            pm.Potential("logrho_prior", tt.switch(logrho > np.log(self.planet.period.value * 1.1), 0, np.inf))
            # log(s2) is a jitter term to compensate for underestimated flux errors
            # We estimate the magnitude of jitter from the CDPP (normalized to the flux)
            logs2 = pm.Normal("logs2", mu=np.log(self._logs2_prior), sd=3)
            kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)

            # Store the GP and cadence mask to aid debugging
            model.gp = xo.gp.GP(kernel, time, self._diag + tt.exp(logs2))

            # The motion model regresses against the design matrix
            A = tt.dot(self.design_matrix.T, model.gp.apply_inverse(self.design_matrix))
            # To ensure the weights can be solved for, we need to perform ridge regression
            # to avoid an ill-conditioned matrix A. Here we define the size of the diagonal
            # along which we will add small values
            ridge = np.array(range(self.design_matrix.shape[1]))
            # Cast the ridge indices into tensor space
            ridge = tt.cast(ridge, 'int64')
            # Apply ridge regression by adding small numbers along the diagonal
            A = tt.set_subtensor(A[ridge, ridge], A[ridge, ridge] + 1e-6)

            # Corrected flux, with the EB model removed.
            cflux = np.reshape(lc_flux, (lc_flux.shape[0], 1))
            cflux -= tt.reshape(eb_model, (eb_model.shape[0], 1))

            B = tt.dot(self.design_matrix.T, model.gp.apply_inverse(cflux))
            weights = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(self.design_matrix, weights)[:, 0])
            pm.Deterministic("weights", weights)


            # Observables
            # ---------------------
            # Track Quanities
            pm.Deterministic("eb_model", eb_model)
            pm.Deterministic("thermal", thermal)
            pm.Deterministic("reflection", reflection)
            pm.Deterministic("transit", transit)
#            pm.Normal("obs", mu=eb_model, sd=lc_flux_err, observed=lc_flux)

            # Likelihood to optimize
            pm.Potential("obs", model.gp.log_likelihood(lc_flux - (motion_model + eb_model)))

            return model

    def _optimize(self, start=None, verbose=True):
        with self._model as model:
            if start is None:
                start = model.test_point

            map_soln = xo.optimize(start=start, vars=[model.mean], verbose=verbose)
            map_soln = xo.optimize(start=start, vars=[model.logrho, model.logsigma, model.mean], verbose=verbose)
            map_soln = xo.optimize(start=start, vars=[model.logrho, model.logsigma, model.logs2], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.logP, model.rprs, model.t0], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.u], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.ep, model.dp], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.inclination], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.albedo], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.albedo, model.dT, model.phase_shift], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.logP, model.rprs, model.t0, model.u], verbose=verbose)
            map_soln = xo.optimize(start=start, vars=[model.logrho, model.logsigma, model.logs2], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.albedo, model.dT], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.ep, model.dp], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.albedo, model.inclination, model.t0,
                                                        model.logP, model.u, model.ep, model.dp], verbose=verbose)
            map_soln = xo.optimize(start=map_soln, vars=[model.ep, model.dp, model.phase_shift, model.dT,
                                                        model.albedo, model.inclination, model.t0, model.logP, model.rprs, model.logrho, model.logsigma, model.logs2, model.mean, model.u], verbose=verbose)

            return map_soln


    def optimize(self, start=None, mask=True, sigma=3, verbose=True):
        # Optimize
        print('Initial Optimization')
        map_soln0 = self._optimize(start=start, verbose=verbose)
        if mask:
            print('Second Optimization')
            eb = map_soln0['eb_model']
            with self._model:
                gp = xo.eval_in_model(self._model.gp.predict(np.asarray(self.lc.time, np.float64)), map_soln0)
            motion_model = map_soln0['motion_model']
            p, t0 = map_soln0['period'], map_soln0['t0']
            f = ((self.lc - motion_model - gp).fold(p, t0) - lk.LightCurve(self.lc.time, eb).fold(p, t0).flux)
            f1 = f.bin(15, 'median')
            f -= np.interp(f.time, f1.time, f1.flux)
            bad = np.abs(f.flux - np.median(f.flux)) > sigma * f.flux.std()
            bad = np.in1d(self.lc.time, f.time_original[bad])
            self._diag[bad] += 1e12
            self.bad = bad
            self.map_soln = self._optimize(start=map_soln0, verbose=verbose)
        else:
            self.map_soln = map_soln0
