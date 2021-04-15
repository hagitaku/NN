import cython
import math
import random
import numpy as np
import copy

cdef class Neuron:
	cdef:
		int n
		double u
		double eta
		double alpha
		double delta
		list w
	def __init__(self,num):
		self.n=num
		self.alpha=2.0
		self.eta=0.05
		self.delta=0
		self.w=[random.random() for i in range(self.n)]
	def reset(self):
		for i in range(self.n):
			self.w[i]=random.random()
	def calc(self,x):
		ret=0.0
		self.u=func1(x,self.w,self.n)
		ret=1.0/(1.0+math.exp(-self.alpha*self.u))
		return ret
	def learn(self,inp,out,sumdelta):
		self.delta=self.alpha*out*(1.0-out)*sumdelta
		for i in range(self.n):
			self.w[i]=self.w[i]+self.eta*self.delta*inp[i]
	def getdelta(self):
		return self.delta
	def getw(self,ind):
		if ind>len(self.w):
			return 0
		return self.w[ind]


class NN:
	net=[]
	table=[]
	def __init__(self,entry = 1,end = 1,depth = 1,width = 1):
		self.net.append([Neuron(entry) for i in range(width)])
		self.table.append([0 for i in range(width)])
		for i in range(depth):
			self.net.append([Neuron(width) for i in range(width)])
			self.table.append([0 for i in range(width)])
		self.net.append([Neuron(width) for i in range(end)])
		self.table.append([0 for i in range(end)])

	def calc(self,inp):
		for i in range(len(self.net)):
			for j in range(len(self.net[i])):
				if i==0:
					self.table[i][j]=self.net[i][j].calc(inp)
				else:
					self.table[i][j]=self.net[i][j].calc(self.table[i-1])
		return copy.deepcopy(self.table[len(self.table)-1])

	def learn(self,anss,inp):
		for i in range(len(anss)):
			if len(anss[i])!=len(self.table[len(self.table)-1]):
				return False#学習失敗
		for now in range(len(anss)):
			ans=anss[now]
			self.calc(inp[now])
			for i in reversed(range(len(self.net))):
				for j in range(len(self.net[i])):
					if i ==0:
						for k in range(len(self.net[i+1])):
							self.net[i][j].learn(inp[now], self.table[i][j], self.net[i+1][k].getdelta() * self.net[i+1][k].getw(j))
					elif i==len(self.table)-1:
						#出力層なのでansとの差から学習する
						self.net[i][j].learn(self.table[i - 1], self.table[i][j], ans[j] - self.table[i][j])
					else:
						for k in range(len(self.net[i+1])):
							self.net[i][j].learn(self.table[i-1], self.table[i][j], self.net[i+1][k].getdelta() * self.net[i+1][k].getw(j))
		return True

# pythonから呼び出す関数
cdef func1(list x,list w,int n):
	cdef:
		int i
		double sum
	sum = 0
	for i in range(n):
		sum += x[i]*w[i]
	return sum
