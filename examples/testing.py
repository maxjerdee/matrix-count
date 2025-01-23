from matrix_count import _util
import numpy as np



print(_util._log_sum_exp([1,2,3])) # 3.4076059644443806

print(np.real(np.exp(_util._log_sum_exp([np.log(4),np.log(-1+0j)+np.log(1),np.log(-1+0j)+np.log(1),np.log(0)]))))