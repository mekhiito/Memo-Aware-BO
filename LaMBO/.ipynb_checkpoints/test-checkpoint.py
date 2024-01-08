import numpy as np

a = np.array([0.5, 1, -1, -2, 3])

print(np.where(a < 0)[0].shape)