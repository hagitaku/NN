import random
import math
import numpy as np
import time
import cymodule


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




if __name__=="__main__":
	are=cymodule.NN(entry = 3,end = 1,depth = 3,width = 2)
	inp=[
	[0.0,0.0,0.0],
	[0.0,0.0,1.0],
	[0.0,1.0,0.0],
	[0.0,1.0,1.0],
	[1.0,0.0,0.0],
	[1.0,0.0,1.0],
	[1.0,1.0,0.0],
	[1.0,1.0,1.0]
	]
	data=[
	[0.0],
	[0.0],
	[0.0],
	[0.0],
	[0.0],
	[0.0],
	[0.0],
	[1.0]
	]
	gosa=0.00001#許容誤差
	result=[[] for i in range(len(inp))]
	now=0
	while(1):
		sum=0
		for i in range(len(inp)):
			result[i] = are.calc(inp[i])
			for ind in range(len(result[i])):
				sum+=0.5*math.pow(data[i][ind] - result[i][ind],2)
		if sum<gosa:
			break
		if now%10000==0:
			print("sum:",sum)
			print(*result,sep="\n")
		are.learn(data,inp)
		now+=1

	print("sum:",sum)
	print(*result,sep="\n")
