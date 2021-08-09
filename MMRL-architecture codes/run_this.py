# -*- coding: UTF-8 -*-
"""
The implementation of Multiple model-based reinforcement learning

"""
from maze_env import Maze
from RL_brain import QLearningTable
import matplotlib.pyplot as plt
import numpy as np
#from module import MultipleModel

#N_MODULE = 6
EPISODES = 20
MAX_STEP = 100

step_list = []
def update():
    print("--- Interaction starts ---")

    for episode in range(EPISODES):  
        step = 0
        #print("--- Episode: %d ---" % episode)
        observation = env.reset()
        #print(observation)


        observation[0] = observation[0] / 70
        observation[1] = observation[1] / 70
        s = np.zeros((7, 7))
        s = s.astype(np.int)
        observation = list(map(lambda x: int(x), observation))
        s[observation[0], observation[1]] = 1
        #print(s)

        prayCoor = env.getFruitCoord()
        prayCoor[0] = prayCoor[0] / 70
        prayCoor[1] = prayCoor[1] / 70
        prayCoor = list(map(lambda x: int(x), prayCoor))
        s[prayCoor[0], prayCoor[1]] = 2

        print(s)

        #prayCoor = env.getFruitCoord()
        #print(prayCoor)
        while True: 
            action = RL.choose_action(str(observation)) 
            #print(observation)猎人坐标
            #print(action)
            #observation_是step函数给出的下一步的状态
            observation_, reward, done = env.step(action)
            #prayCoor = env.getFruitCoord()
            #print(prayCoor)
            env.render()
            #print(observation_)
            #Q-learning参数，用来更新q表
            RL.learn(str(observation), action, reward, str(observation_))
            #状态更新
            observation = observation_  
            step += 1
            if done or (step >= MAX_STEP): 
                break
        print('The episode: %d,  with %d steps.' % (episode, step))
        step_list.append(step)
    print('over')  # end of game
    env.destroy()




def plot_step():
    plt.plot(list(np.arange(EPISODES) + 1), step_list)
    plt.xlabel('Episodes')
    plt.ylabel('Step')
    plt.show()


if __name__ == "__main__":
    env = Maze()  
    RL = QLearningTable(actions=list(range(env.n_actions)))  
    #MM = MultipleModel(actions=list(range(env.n_actions)), n_modules=list(range(N_MODULE)))
    env.after(500, update)  
    env.mainloop()
    plot_step()
