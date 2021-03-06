"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()

def _keplerian_v_nbody(b, ltte, period, plot=False):
    """
    test a single bundle for the phoebe backend's kepler vs nbody dynamics methods
    """

    # TODO: loop over ltte=True,False (once keplerian dynamics supports the switch)

    # b.add_compute(dynamics_method='bs')
    b.set_value('dynamics_method', 'bs')

    times = np.linspace(0, 5*period, 101)
    nb_ts, nb_xs, nb_ys, nb_zs, nb_vxs, nb_vys, nb_vzs = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, ltte=ltte)
    k_ts, k_xs, k_ys, k_zs, k_vxs, k_vys, k_vzs = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, ltte=ltte)

    assert(np.allclose(nb_ts, k_ts, 1e-8))
    for ci in range(len(b.hierarchy.get_stars())):
        # TODO: make rtol lower if possible
        assert(np.allclose(nb_xs[ci], k_xs[ci], rtol=1e-5, atol=1e-2))
        assert(np.allclose(nb_ys[ci], k_ys[ci], rtol=1e-5, atol=1e-2))
        assert(np.allclose(nb_zs[ci], k_zs[ci], rtol=1e-5, atol=1e-2))

        # nbody ltte velocities are wrong so only check velocities if ltte off
        if not ltte:
            assert(np.allclose(nb_vxs[ci], k_vxs[ci], rtol=1e-5, atol=1e-2))
            assert(np.allclose(nb_vys[ci], k_vys[ci], rtol=1e-5, atol=1e-2))
            assert(np.allclose(nb_vzs[ci], k_vzs[ci], rtol=1e-5, atol=1e-2))

def _phoebe_v_photodynam(b, period, plot=False):
    """
    test a single bundle for phoebe's nbody vs photodynam via the frontend
    """

    times = np.linspace(0, 5*period, 21)

    b.add_dataset('orb', times=times, dataset='orb01', components=b.hierarchy.get_stars())
    # photodynam and phoebe should have the same nbody defaults... if for some reason that changes,
    # then this will probably fail
    b.add_compute('photodynam', compute='pdcompute')
    # photodynam backend ONLY works with ltte=True, so we will run the phoebe backend with that as well
    # TODO: remove distortion_method='nbody' once that is supported
    b.add_compute('phoebe', dynamics_method='nbody', ltte=True, compute='phoebecompute')

    b.run_compute('pdcompute', model='pdresults')
    b.run_compute('phoebecompute', model='phoeberesults')

    for comp in b.hierarchy.get_stars():
        # TODO: check to see how low we can make atol (or change to rtol?)
        # TODO: look into justification of flipping x and y for both dynamics (photodynam & phoebe)
        # TODO: why the small discrepancy (visible especially in y, still <1e-11) - possibly a difference in time0 or just a precision limit in the photodynam backend since loading from a file??


        if plot:
            for k in ['xs', 'ys', 'zs', 'vxs', 'vys', 'vzs']:
                plt.cla()
                plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='phoeberesults', component=comp), 'r-')
                plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='pdresults', component=comp), 'b-')
                diff = abs(b.get_value(k, model='phoeberesults', component=comp) - b.get_value(k, model='pdresults', component=comp))
                print "*** max abs: {}".format(max(diff))
                plt.xlabel('t')
                plt.ylabel(k)
                plt.show()

        assert(np.allclose(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value('times', model='pdresults', component=comp, unit=u.d), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('xs', model='phoeberesults', component=comp, unit=u.AU), b.get_value('xs', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('ys', model='phoeberesults', component=comp, unit=u.AU), b.get_value('ys', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('zs', model='phoeberesults', component=comp, unit=u.AU), b.get_value('zs', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vxs', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vxs', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vys', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vys', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vzs', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vzs', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))

def _frontend_v_backend(b, ltte, period, plot=False):
    """
    test a single bundle for the frontend vs backend access to both kepler and nbody dynamics
    """

    # TODO: loop over ltte=True,False

    times = np.linspace(0, 5*period, 101)
    b.add_dataset('orb', times=times, dataset='orb01', components=b.hierarchy.get_stars())
    b.add_compute('phoebe', dynamics_method='keplerian', compute='keplerian', ltte=ltte)
    b.add_compute('phoebe', dynamics_method='bs', compute='nbody', ltte=ltte)


    # NBODY
    # do backend Nbody
    b_ts, b_xs, b_ys, b_zs, b_vxs, b_vys, b_vzs = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, compute='nbody', ltte=ltte)

    # do frontend Nbody
    b.run_compute('nbody', model='nbodyresults')


    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', model='nbodyresults', component=comp, unit=u.d), b_ts, rtol=0, atol=1e-6))
        assert(np.allclose(b.get_value('xs', model='nbodyresults', component=comp, unit=u.solRad), b_xs[ci], rtol=1e-7, atol=1e-4))
        assert(np.allclose(b.get_value('ys', model='nbodyresults', component=comp, unit=u.solRad), b_ys[ci], rtol=1e-7, atol=1e-4))
        assert(np.allclose(b.get_value('zs', model='nbodyresults', component=comp, unit=u.solRad), b_zs[ci], rtol=1e-7, atol=1e-4))
        if not ltte:
            assert(np.allclose(b.get_value('vxs', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vxs[ci], rtol=1e-7, atol=1e-4))
            assert(np.allclose(b.get_value('vys', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vys[ci], rtol=1e-7, atol=1e-4))
            assert(np.allclose(b.get_value('vzs', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vzs[ci], rtol=1e-7, atol=1e-4))




    # KEPLERIAN
    # do backend keplerian
    b_ts, b_xs, b_ys, b_zs, b_vxs, b_vys, b_vzs = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, compute='keplerian', ltte=ltte)


    # do frontend keplerian
    b.run_compute('keplerian', model='keplerianresults')


    # TODO: loop over components and assert
    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', model='keplerianresults', component=comp, unit=u.d), b_ts, rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('xs', model='keplerianresults', component=comp, unit=u.solRad), b_xs[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('ys', model='keplerianresults', component=comp, unit=u.solRad), b_ys[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('zs', model='keplerianresults', component=comp, unit=u.solRad), b_zs[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vxs', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vxs[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vys', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vys[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vzs', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vzs[ci], rtol=0, atol=1e-08))



def test_binary(plot=False):
    """
    """
    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    # system = [sma (AU), period (d)]
    system1 = [0.05, 2.575]
    system2 = [1., 257.5]
    system3 = [40., 65000.]

    for system in [system1,system2,system3]:
        for q in [0.5,1.]:
            for ltte in [True, False]:

                b = phoebe.default_binary()
                b.set_default_unit_all('sma', u.AU)
                b.set_default_unit_all('period', u.d)

                b.set_value('sma@binary',system[0])
                b.set_value('period@binary', system[1])
                b.set_value('q', q)
                _keplerian_v_nbody(b, ltte, system[1], plot=plot)

                b = phoebe.default_binary()
                b.set_default_unit_all('sma', u.AU)
                b.set_default_unit_all('period', u.d)

                b.set_value('sma@binary',system[0])
                b.set_value('period@binary', system[1])
                b.set_value('q', q)
                _frontend_v_backend(b, ltte, system[1], plot=plot)

    #for system in [system1,system2,system3]:
    #for q in [0.5,1.]:
        #b = phoebe.Bundle.default_binary()
        #b.set_default_unit_all('sma', u.AU)
        #b.set_default_unit_all('period', u.d)

        #b.set_value('sma@binary',system[0])
        #b.set_value('period@binary', system[1])
        #b.set_value('q', q)
        #_phoebe_v_photodynam(b, system[1], plot=plot)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    test_binary(plot=True)

    # TODO: create tests for both triple configurations (A--B-C, A-B--C) - these should first be default bundles