import gym
import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from statistics import mean, median
from collections import Counter
from time import sleep

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def get_random_games():
	for ep in range(5):
		env.reset()
		for t in range(goal_steps):
			sleep(1e-2)
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break

get_random_games()

for _ in range(1000):
	env.render()