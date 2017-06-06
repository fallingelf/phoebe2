import logging
import numpy as np
from phoebe import dynamics
from phoebe import c, u

from scipy.optimize import minimize

logger = logging.getLogger("ETVS")
logger.addHandler(logging.NullHandler())

_ltte_scale_factor = (c.R_sun/c.c).to(u.d).value

def barycentric():
    """
    """
    raise NotImplementedError


def crossing(b, component, time, dynamics_method='keplerian', ltte=True, tol=1e-6):
    """
    tol in days
    """


    def projected_separation_sq(time, b, dynamics_method, cind1, cind2, ltte=True):
        """
        """
        #print "*** projected_separation_sq", time, dynamics_method, cind1, cind2, ltte


        times = np.asarray(time)

        if dynamics_method in ['nbody', 'rebound']:
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle(b, times, compute=None, ltte=ltte)

        elif dynamics_method=='bs':
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle_bs(b, times, compute=None, ltte=ltte)

        elif dynamics_method=='keplerian':
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            ts, corrected_ts, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls = dynamics.keplerian.dynamics_from_bundle(b, times, compute=None, ltte=ltte, return_euler=True)

        else:
            raise NotImplementedError

        x1prime = xs[cind1][0] * np.cos(elongans[cind1][0])
        x2prime = xs[cind2][0] * np.cos(elongans[cind2][0])
        y1prime = ys[cind1][0] * np.sin(elongans[cind1][0])
        y2prime = ys[cind2][0] * np.sin(elongans[cind2][0])

        # print elongans[cind1][0], elongans[cind2][0]
        # print x1prime, x2prime, y1prime, y2prime
        # print times, (x2prime-x1prime)**2 + (y2prime-y1prime)**2

        return (x2prime-x1prime)**2 + (y2prime-y1prime)**2

        # return (np.cos(ethetas[cind1][0])-np.cos(ethetas[cind2][0]))**2
        # return (xs[cind2][0]-xs[cind1][0])**2 + (ys[cind2][0]-ys[cind1][0])**2
        # return (xs[cind2][0]-xs[cind1][0])**2


    # TODO: optimize this by allowing to pass cind1 and cind2 directly (and fallback to this if they aren't)
    starrefs = b.hierarchy.get_stars()
    cind1 = starrefs.index(component)
    cind2 = starrefs.index(b.hierarchy.get_sibling_of(component))

    orb_period = b.get_value('period', component=b.hierarchy.get_parent_of(component), context='component')

    # TODO: use z/c to estimate initial guess
    # guess = time + np.mean(zs)*_ltte_scale_factor

    # TODO: find the best minimizer (ie most efficient)
    # res = minimize(projected_separation_sq, x0=[time], method='TNC', args=(b, dynamics_method, cind1, cind2, ltte), options={'xtol': tol}, bounds=((time-orb_period/2., time+orb_period/2.),))
    res = minimize(projected_separation_sq, x0=[time], method='TNC', args=(b, dynamics_method, cind1, cind2, ltte), options={'xtol': 1e-3, 'ftol': 1e-3}, bounds=((time-orb_period/2., time+orb_period/2.),))
    # res = minimize(projected_separation_sq, x0=[time], method='SLSQP', args=(b, dynamics_method, cind1, cind2, ltte), options={'ftol': tol}, bounds=((time-orb_period/2., time+orb_period/2.),))
    # res = minimize(projected_separation_sq, x0=[time], method='Nelder-Mead', args=(b, dynamics_method, cind1, cind2, ltte), options={'xatol': tol})
    # res = minimize(projected_separation_sq, x0=[time], method='Powell', args=(b, dynamics_method, cind1, cind2, ltte), options={'xtol': tol})

    if not res.success:
        logger.warning("Failed to find eclipse time near {} with msg: '{}'".format(time, res.message))
        exit()
        return np.nan

    # manually check bounds (not necessary for minimizers that accept bounds)
    if res.x[0] < time-orb_period/2.:
        return time-orb_period/2.
    elif res.x[0] > time+orb_period/2.:
        return time+orb_period/2.
    else:
        return res.x[0]
