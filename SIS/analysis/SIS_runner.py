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

inputs_folder = 'test-inputs'
outputs_folder = 'test-outputs'

# inputs_folder = 'randomInputs'
# outputs_folder = 'ecRandomOutputs'
# outputs_folder = 'gcRandomOutputs'
# outputs_folder = 'chenRandomOutputs'

# inputs_folder = 'sizeInputs'
# outputs_folder = 'sizeOutputs'

# inputs_folder = 'uniformInputs'
# outputs_folder = 'ecUniformOutputs'

# inputs_folder = 'gaussInputs'
# outputs_folder = 'gaussOutputs'

# inputs_folder = 'specialInputs'
# outputs_folder = 'specialOutputs'

# inputs_folder = 'specialInputs2'
# outputs_folder = 'specialOutputs2'

#./contingencyMonteCarlo -i inputs7/q1_04q2_04n0800t0n2n1v0.txt -o tOut.txt -e 24 -T 0.99 -t 10000000 -s 250 -S 10000000

def process(file):

  def kill(process):
    process.kill() 
    print('Timeout on', file)

  # Maximum of 1 trillion iterations per step
  # ./SIS-EC-ln -i inputs4/q1_02q2_02n3200t0.txt -o outputs4/q1_02q2_02n3200t0.txt -t 1000000 -S 50
  ping = Popen('./SIS-EC-ZeroOne -i '+inputs_folder+'/'+file+' -o '+outputs_folder+'/'+file+' -t 10000000 -H CGM',shell=True)
  # ping = Popen('./SIS-EC-ln -i '+inputs_folder+'/'+file+' -o '+outputs_folder+'/'+file+' -t 10000000',shell=True)
  # ping = Popen('./SIS-GC-ln -i '+inputs_folder+'/'+file+' -o '+outputs_folder+'/'+file+' -t 10000000',shell=True)
  # ping = Popen('./SIS-Chen -i '+inputs_folder+'/'+file+' -o '+outputs_folder+'/'+file+' -t 10000000',shell=True)
  my_timer = Timer(1*60*60, kill, [ping]) # timeout
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

# np.random.seed(0)
# np.random.shuffle(input_files)
#print(input_files[400])


filtered_files = []
# for file in input_files:
#   if int(re.findall('t[0-9]+',file)[0][1:]) == int(sys.argv[1]): 
#     filtered_files.append(file)
filtered_files = input_files

r = Parallel(n_jobs=36,verbose=10)(delayed(process)(file) for file in filtered_files)
