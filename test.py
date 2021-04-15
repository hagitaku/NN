import random
import math
import numpy as np
import time
import sample

"""
#import gym

def game():
	env=gym.make('CartPole-v0')
	while(1):
		env.reset()
		env.render()
		for step in range(800):
			action = env.action_space.sample()
			observe, reward, done, _ = env.step(action)
			print(observe, reward, done, _)
			env.render()
			if done:
				break
"""
count=0

class Neuron:
	n=u=eta=alpha=delta=0
	w=[]
	def __init__(self,num):
		self.n=num
		self.alpha=1.0
		self.eta=0.05
		self.w=[random.random() for i in range(self.n)]
	def reset(self):
		for i in range(self.n):
			self.w[i]=random.random()
	def calc(self,x):
		ret=0.0
		self.u=sample.func1(x,self.w,self.n)
		ret=1.0/(1.0+math.exp(-self.alpha*self.u))
		return ret
	def learn(self,inp,out,sumdelta):
		self.delta=self.alpha*out*(1.0-out)*sumdelta
		for i in range(self.n):
			self.w[i]=self.w[i]+self.eta*self.delta*inp[i]

class NN:
	net=[[]]
	table=[[]]
	def __init__(self,entry = 1,end = 1,depth = 1,width = 1):
		self.net[0]=[Neuron(entry) for i in range(width)]
		self.table[0]=[0 for i in range(width)]
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
		return self.table[len(self.table)-1]

	def learn(self,anss,inp):
		for i in range(len(anss)):
			if len(anss[i])!=len(self.table[len(self.table)-1]):
				return False#学習失敗
		nga=0
		for now in range(len(anss)):
			ans=anss[now]
			self.calc(inp[now])
			for i in reversed(range(len(self.net))):
				for j in range(len(self.net[i])):
					if i ==0:
						for k in range(len(self.net[i+1])):
							nga+=1
							self.net[i][j].learn(inp[now], self.table[i][j], self.net[i+1][k].delta * self.net[i+1][k].w[j])
					elif i==len(self.table)-1:
						#出力層なのでansとの差から学習する
						nga+=1
						self.net[i][j].learn(self.table[i - 1], self.table[i][j], ans[j] - self.table[i][j])
					else:
						for k in range(len(self.net[i+1])):
							nga+=1
							self.net[i][j].learn(self.table[i-1], self.table[i][j], self.net[i+1][k].delta * self.net[i+1][k].w[j])
		return True




if __name__=="__main__":
	are=NN(entry = 2,end = 1,depth = 1,width = 2)
	inp=[
	[0.0,0.0],
	[0.0,1.0],
	[1.0,0.0],
	[1.0,1.0],
	]
	data=[[0.0],[1.0],[1.0],[0.0]]
	gosa=0.0001
	result=[0 for i in range(4)]
	now=0
	start = time.time()
	while(1):
		sum=0
		for i in range(4):
			result[i] = are.calc(inp[i]);
			for ind in range(len(result[i])):
				sum+=0.5*math.pow(data[i][ind] - result[i][ind],2)
		if sum<gosa:
			break
		are.learn(data,inp)
		if now%1000==0:
			print(sum)
	print(time.time()-start)
	print(result)
