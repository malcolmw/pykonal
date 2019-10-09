import nose
import numpy as np
import pykonal
import unittest


def random_grid():
    grid                 = pykonal.Grid3D()
    grid.min_coords      = 100 \
       * np.sign(np.random.randint(-1, 1, 3))\
       * np.random.rand(3)
    grid.node_intervals  = np.random.rand(3) * 100
    grid.npts            = np.random.randint(1, 100, 3)

    while True:
        vv   = np.random.randint(1, 1e4) * np.random.rand(*grid.npts)
        vv   = vv.astype(pykonal.DTYPE_REAL)
        if vv.min() > 0:
            break
    return (grid, vv)


class LinearInterpolator3DTestCase(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_interpolate_2D(self):
        grid                = pykonal.Grid3D(coord_sys='cartesian')
        grid.min_coords     = 0, 0, 0
        grid.node_intervals = 1, 1, 1
        grid.npts           = 10, 10, 1
        vv                  = np.random.rand(10, 10, 1)
        interpolator = pykonal.LinearInterpolator3D(grid, vv)
        interpolator([5, 5, 5])


    def test_interpolate(self):

        for i in range(10):
            grid, vv = random_grid()
            vmin, vmax = vv.min(), vv.max()
            delta = grid.max_coords - grid.min_coords
            ii = pykonal.LinearInterpolator3D(grid, vv)
            for j in range(10):
                xyz = (grid.min_coords + np.random.rand(3) * delta).astype(pykonal.DTYPE_REAL)
                v1, v2 = ii.interpolate(xyz), ii(xyz)
                self.assertEqual(v1, v2)
                self.assertGreater(v1, vmin)
                self.assertLess(v1, vmax)


    def test_interpolate_OutOfBoundsError(self):
        for i in range(10):
            grid, vv = random_grid()
            delta = grid.max_coords - grid.min_coords
            ii = pykonal.LinearInterpolator3D(grid, vv)
            with self.assertRaises(pykonal.OutOfBoundsError):
                ii.interpolate(grid.max_coords + 2 * delta)
            with self.assertRaises(pykonal.OutOfBoundsError):
                ii(grid.max_coords + 2 * delta)
            with self.assertRaises(pykonal.OutOfBoundsError):
                ii.interpolate(grid.min_coords - 2 * delta)
            with self.assertRaises(pykonal.OutOfBoundsError):
                ii(grid.min_coords - 2 * delta)


    def test_interpolate_error_float(self):

        for i in range(10):
            grid, vv = random_grid()
            delta = grid.max_coords - grid.min_coords
            ii = pykonal.LinearInterpolator3D(
                grid,
                np.full(
                    vv.shape,
                    fill_value=pykonal.ERROR_REAL,
                    dtype=pykonal.DTYPE_REAL
                )
            )
            for j in range(10):
                xyz = (grid.min_coords + np.random.rand(3) * delta).astype(pykonal.DTYPE_REAL)
                v1, v2 = ii.interpolate(xyz), ii(xyz)
                self.assertEqual(v1, v2)
                self.assertEqual(v1, pykonal.ERROR_REAL)


    def test_interpolate_edge_case_lower(self):
        for i in range(10):
            grid, vv = random_grid()
            ii = pykonal.LinearInterpolator3D(grid, vv)
            v1, v2 = ii.interpolate(grid.min_coords), ii(grid.min_coords)
            self.assertEqual(v1, v2)
            self.assertEqual(v1, vv[0, 0, 0])


    def test_interpolate_edge_case_upper(self):
        for i in range(10):
            grid, vv = random_grid()
            ii = pykonal.LinearInterpolator3D(grid, vv)
            v1, v2 = ii.interpolate(grid.max_coords), ii(grid.max_coords)
            self.assertEqual(v1, v2)
            self.assertEqual(v1, vv[-1, -1, -1])

    def test_periodic(self):
        grid                = pykonal.Grid3D(coord_sys='spherical')
        grid.min_coords     = 1, 0, 0
        grid.node_intervals = 1, np.pi/20, np.pi/20
        grid.npts           = 2, 21, 40
        vv                  = np.cos(grid.nodes[...,2])
        interpolator = pykonal.LinearInterpolator3D(grid, vv)
        interpolator([1.5, np.pi/2, 39.5*np.pi/20])


if __name__ == '__main__':
    nose.main()
