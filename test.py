import matplotlib.pyplot as plt
import numpy as np
import pykonal

nx, ny = 20, 20
vv = np.ones((nx, ny), dtype=np.float32)
uu = pykonal.pykonal(vv)

plt.close('all')
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('First order')
c = ax.imshow(uu)
cbar = fig.colorbar(c, ax=ax, orientation='horizontal')
cbar.set_label('% error')
ax.set_xlim(0.5, nx-0.5)
ax.set_ylim(ny-0.5, 0.5)
plt.show()
