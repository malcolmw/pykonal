import heap
import numpy as np
import pykonal
import pykonal_raw
import time


nx, ny = 50, 50
vv = np.ones((nx, ny), dtype=np.float32)

t0 = time.time()
uu_raw = pykonal_raw.pykonal(vv)
traw = time.time() - t0

t0 = time.time()
uu = pykonal.pykonal(vv)
topt = time.time() - t0


print(f'Grid size: {nx}, {ny}')
print(f'Elapsed time (raw): {traw: .6f} seconds')
print(f'Elapsed time (optimized): {topt: .6f} seconds')
if np.all(uu_raw == uu):
    print(f'Equivalency achieved' )
else:
    print(f'Equivalency test failed')
    for ix in np.random.randint(nx, size=3):
        for iy in np.random.randint(ny, size=3):
            error = np.abs(uu_raw[ix,iy]-uu[ix,iy])
            error /= np.mean([uu_raw[ix,iy], uu[ix,iy]])
            error *= 100
            print(f'[{ix:3d}, {iy:3d}] - '
                  f'{uu_raw[ix, iy]:10.6f} :: {uu[ix, iy]:10.6f} '
                  f'({error:.5f}%)')
