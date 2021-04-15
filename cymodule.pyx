import cython
import numpy as np
cimport numpy as cnp

# pythonから呼び出す関数
def func1(list x,list w,int n):
	cdef:
		int i
		double sum
	sum = 0
	for i in range(n):
		sum += x[i]*w[i]
	return sum
