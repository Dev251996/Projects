#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
from gym import wrappers
import numpy as np
import pygame
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')

from IPython import display

class Agent:
    def decide(self, t):
        if t > 50 and t < 100:
            action = 0 #push left
        else:
            action = 2 #push right
        return action

agent = Agent()

def play_game(env, agent):
    observation = env.reset()
    score = 0
    for t in range(200):
        print("Step: ", t)
        env.render()
        action = agent.decide(t)
        observation, reward, done, info, _ = env.step(action)
        score += reward
        print("Step Results: ", observation, reward, done, info)
        if done:
            break  
    return score 
            
import itertools
env = gym.make('MountainCar-v0', render_mode="human")
final_score = play_game(env, agent)
print("Final score: ", final_score)
env.close()


# In[ ]:




