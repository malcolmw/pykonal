import nose
import numpy as np
import os
import pkg_resources
import pykonal
import unittest



class EikonalSolverTestCase(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_uniform_velocity_cartesian(self):
        fname = pkg_resources.resource_filename(
            'pykonal',
            os.path.join(
                'pykonal',
                'tests',
                'data',
                'test_EikonalSolver_uniform_velocity_cartesian.npz'
            )
        )

        solver = pykonal.EikonalSolver()

        with np.load(fname) as inf:
            uu = inf['uu']
            solver.vgrid.min_coords     = inf['vgrid_min_coords']
            solver.vgrid.node_intervals = inf['vgrid_node_intervals']
            solver.vgrid.npts           = inf['vgrid_npts']
            solver.vv                   = inf['vv']
            solver.pgrid.min_coords     = inf['pgrid_min_coords']
            solver.pgrid.node_intervals = inf['pgrid_node_intervals']
            solver.pgrid.npts           = inf['pgrid_npts']
            solver.add_source(inf['src'])
        solver.solve()
        np.testing.assert_array_almost_equal(uu, solver.uu)


if __name__ == '__main__':
    nose.main()
