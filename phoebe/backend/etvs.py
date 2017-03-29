import logging
import numpy as np
from phoebe import dynamics

from scipy.optimize import minimize

logger = logging.getLogger("ETVS")

def barycentric():
    """
    """
    raise NotImplementedError


def crossing(b, component, time, dynamics_method='keplerian', ltte=True, tol=1e-4):
    """
    tol in days
    """


    def projected_separation_sq(time, b, dynamics_method, cind1, cind2, ltte=True):
        """
        """
        #print "*** projected_separation_sq", time, dynamics_method, cind1, cind2, ltte


        times = np.array([time])

        if dynamics_method in ['nbody', 'rebound']:
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle(b, times, compute=None, ltte=ltte)

        elif dynamics_method=='bs':
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.nbody.dynamics_from_bundle_bs(b, times, compute=None, ltte=ltte)

        elif dynamics_method=='keplerian':
            # TODO: make sure that this takes systemic velocity and corrects positions and velocities (including ltte effects if enabled)
            ts, xs, ys, zs, vxs, vys, vzs = dynamics.keplerian.dynamics_from_bundle(b, times, compute=None, ltte=ltte, return_euler=False)

        else:
            raise NotImplementedError


        return (xs[cind2][0]-xs[cind1][0])**2 + (ys[cind2][0]-ys[cind1][0])**2


    # TODO: optimize this by allowing to pass cind1 and cind2 directly (and fallback to this if they aren't)
    starrefs = b.hierarchy.get_stars()
    cind1 = starrefs.index(component)
    cind2 = starrefs.index(b.hierarchy.get_sibling_of(component))

    orb_period = b.get_value('period', component=b.hierarchy.get_parent_of(component), context='component')

    res = minimize(projected_separation_sq, x0=[time], args=(b, dynamics_method, cind1, cind2, ltte), tol=tol, bounds=((time-orb_period, time+orb_period),))

    if not res.success:
        logger.warning("Failed to find eclipse time near {} with msg {}".format(time, res.message))
        return time

    return res.x[0]

