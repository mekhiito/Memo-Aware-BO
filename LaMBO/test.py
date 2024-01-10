import numpy as np
import random

a = np.array([0.5, 1, -1, -2, 3])
probs = np.array([0.01, 0.1, 0.5, 0.09, 0.4])
sigma = np.array(random.choices([-1,1], k=4))
sigma[-1] = -1
print(sigma)
print(np.where(sigma == -1)[0][0])