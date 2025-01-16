<<<<<<< HEAD
# Script to test the functionality of the matrix counting estimates provided in this package.

import numpy as np

#from estimates.linear_time import log_Omega_S_DM

#print(log_Omega_S_DM(0))

print(np.random.dirichlet(np.ones(5)))
print(np.random.dirichlet(10 * np.ones(5)))
print(np.random.dirichlet(10 ** (-3) * np.ones(5)))
=======
# Script to test the functionality of the matrix counting estimates provided in this package.

import numpy as np

from estimates.linear_time import log_Omega_S_DM

print(log_Omega_S_DM(0))
>>>>>>> f25b5b1caa0b2f45b3e011f11766d9572cbc9330
