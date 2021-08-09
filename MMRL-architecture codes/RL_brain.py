# -*- coding: UTF-8 -*-
"""



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # actions: 0,1,2,3,4

    def choose_action(self, observation):
        #猎人坐标
        self.check_state_exist(observation)
        #贪婪
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        #用loc函数返回五个动作的q值
        q_predict = self.q_table.loc[s, a] 
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  
        else:
            q_target = r 
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        #print(self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            

  #  def plot_cost(self):
   #     plt.plot(np.arange(20), self.cost_his)
  #      plt.ylabel('Cost')
   #     plt.xlabel('training steps')
   #     plt.show()