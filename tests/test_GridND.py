import nose
import numpy as np
import pykonal
import unittest


def random_grid():
    grid                 = pykonal.GridND(ndim=3)
    grid.min_coords      = 100 \
       * np.sign(np.random.randint(-1, 1, 3))\
       * np.random.rand(grid.ndim)
    grid.node_intervals  = np.random.rand(3) * 100
    grid.npts            = np.random.randint(1, 100, 3)
    return (grid)


class GridNDTestCase(unittest.TestCase):


    def setUp(self):
        self.grid                 = pykonal.GridND(ndim=3)
        self.grid.min_coords      = -10, 0, 10
        self.grid.node_intervals  = 0.1, 1, 10.
        self.grid.npts            = 100, 1, 10


    def tearDown(self):
        pass


    def test_iax_null(self):
        self.assertNotIn(0, self.grid.iax_null)
        self.assertIn(1, self.grid.iax_null)
        self.assertNotIn(2, self.grid.iax_null)


    def test_mesh(self):
        grid = random_grid()
        self.assertTrue(
            np.all(
                np.min(grid[...], axis=(0, 1, 2)) == grid.min_coords
            )
        )
        self.assertTrue(
            np.all(
                np.max(grid[...], axis=(0, 1, 2)) == grid.max_coords
            )
        )


    def test_min_coords(self):
        for i in range(10):
            grid = random_grid()
            self.assertEqual(grid.min_coords.dtype, pykonal.DTYPE_FLOAT)


    def test_max_coords(self):
        for i in range(10):
            grid = random_grid()
            self.assertEqual(grid.max_coords.dtype, pykonal.DTYPE_FLOAT)
            self.assertTrue(
                np.all(grid.max_coords >= grid.min_coords)
            )


    def test_node_intervals(self):
        for i in range(10):
            grid = random_grid()
            self.assertEqual(grid.node_intervals.dtype, pykonal.DTYPE_FLOAT)

    def test_npts(self):
        for i in range(10):
            grid = random_grid()
            with self.assertRaises(TypeError):
                grid.npts = grid.npts[0]
            with self.assertRaises(ValueError):
                grid.npts = grid.npts[:2]
            self.assertEqual(grid.npts.dtype, pykonal.DTYPE_INT)


if __name__ == '__main__':
    nose.main()
