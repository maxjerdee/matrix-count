<<<<<<< HEAD
# Run the SIS c++ script on the test margins

# Import packages
from os import listdir
from os.path import isfile, join
import numpy as np
import re
from math import log,exp
from numpy import loadtxt,zeros,sum
from numpy import log as nplog
from scipy.special import gammaln
from scipy.sparse import coo_matrix
from scipy.special import gamma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from threading import Thread
import traceback
from subprocess import Popen, PIPE
from threading import Timer
import sys

# Specify working directories
## TODO: change accordingly

inputs_folder = '../../data/more_verbose_test_margins'
outputs_folder = '../../outputs/more_verbose_test_margins'

# inputs_folder = '../../data/facebook_friends_test_margins'
# outputs_folder = '../../outputs/facebook_friends_test_margins'

# inputs_folder = '../../data/test_margins'
# outputs_folder = '../../outputs/test_margins'

# inputs_folder = '../../data/verbose_test_margins'
# outputs_folder = '../../outputs/verbose_test_margins'

# inputs_folder = '../../data/polbooks_test_margins'
# outputs_folder = '../../outputs/polbooks_test_margins'

# inputs_folder = '../data/real_degrees/margins'
# outputs_folder = '../outputs/real_degrees'

# Run C++ script
# ../generate_samples -i ../data/test_margins/margin_0800_020_1.txt -o ../outputs/test_margins/margin_0800_020_1.txt -T 10000 -t 600 
def process(filename):

  def kill(process):
    process.kill() 
    print('Timeout on', filename)

  ping = Popen(f"../generate_samples -i {inputs_folder}/{filename} -o {outputs_folder}/{filename} -T 10000 -t 3000",shell=True)
  my_timer = Timer(60*60, kill, [ping]) # timeout
  try:
      my_timer.start()
      stdout, stderr = ping.communicate()
  finally:
      my_timer.cancel()
  
  # except Exception as e:
  #   print('Error on',file)
  #   print(e)
  #   print(traceback.format_exc())
  #   pass


input_files = listdir(inputs_folder)
#print(input_files)

from joblib import Parallel, delayed

input_files = np.array(input_files)
input_files.sort()

r = Parallel(n_jobs=36,verbose=10)(delayed(process)(file) for file in input_files)
=======
# Run the SIS c++ script on the test margins

from os import listdir
from os.path import isfile, join
import numpy as np
import re
from math import log,exp
from numpy import loadtxt,zeros,sum
from numpy import log as nplog
from scipy.special import gammaln
from scipy.sparse import coo_matrix
from scipy.special import gamma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from threading import Thread
import traceback
from subprocess import Popen, PIPE
from threading import Timer
import sys

inputs_folder = '../data/test_margins'
outputs_folder = '../outputs/test_margins'

# inputs_folder = '../data/real_degrees/margins'
# outputs_folder = '../outputs/real_degrees'

# ../generate_samples -i ../data/test_margins/margin_0800_020_1.txt -o ../outputs/test_margins/margin_0800_020_1.txt -T 10000 -t 600 
def process(filename):

  def kill(process):
    process.kill() 
    print('Timeout on', filename)

  ping = Popen(f"../generate_samples -i {inputs_folder}/{filename} -o {outputs_folder}/{filename} -T 10000 -t 3000",shell=True)
  my_timer = Timer(60*60, kill, [ping]) # timeout
  try:
      my_timer.start()
      stdout, stderr = ping.communicate()
  finally:
      my_timer.cancel()
  
  # except Exception as e:
  #   print('Error on',file)
  #   print(e)
  #   print(traceback.format_exc())
  #   pass


input_files = listdir(inputs_folder)
#print(input_files)

from joblib import Parallel, delayed

input_files = np.array(input_files)
input_files.sort()

r = Parallel(n_jobs=3,verbose=10)(delayed(process)(file) for file in input_files)
# r = Parallel(n_jobs=36,verbose=10)(delayed(process)(file) for file in input_files)
>>>>>>> f25b5b1caa0b2f45b3e011f11766d9572cbc9330
