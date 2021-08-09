from matplotlib import pyplot as plt

from DQN_brain import Net
from DQN_brain import DQN
from maze_env import Maze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#参数
BATCH_SIZE = 32
LR = 0.01                   # 学习率
EPSILON = 0.9               # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 2000      # 记忆库大小
N_ACTIONS = 5  # 棋子的动作0，1，2，3,4
N_STATES = 1
EPISODES = 40


step_list = []
MAX_STEP = 100

#输入，1是小球，2是终点
def trans_torch(list1):
    list1=np.array(list1)
    l1=np.where(list1==1,1,0)
    #print(l1)
    l2=np.where(list1==2,1,0)
    #print(l2)
    b=np.array([l1,l2])
    return b

#训练部分
def update():
    print("--- Interaction starts ---")
    #400步
    study = 1
    for i_episode in range(EPISODES):
        step = 1
        #print(i_episode,'epoch')
        #s = np.zeros((7, 7))
        #s = s.astype(np.int)

        s = [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0], ]
        #猎人坐标
        observation = env.reset()
        observation[0] = observation[0] / 70
        observation[1] = observation[1] / 70
        observation = list(map(lambda x: int(x), observation))
        s[observation[0]][observation[1]] = 1

        prayCoor = env.getFruitCoord()
        prayCoor[0] = prayCoor[0] / 70
        prayCoor[1] = prayCoor[1] / 70
        prayCoor = list(map(lambda x: int(x), prayCoor))
        s[prayCoor[0]][prayCoor[1]] = 2

        s = trans_torch(s)
        while True:
            #env.display()   # 显示实验动画
            a = dqn.choose_action(s) #选择动作
            # 选动作, 得到环境反馈
            observation_,r,done = env.step(a)

            if done or (step >= MAX_STEP):
                break


           # s_ = np.zeros((7, 7))
           # s_ = s_.astype(np.int)
            s_ = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],]

            #print(observation_)
            #print(done)
            observation_[0] = observation_[0] / 70
            observation_[1] = observation_[1] / 70
            observation_ = list(map(lambda x: int(x), observation_))
            s_[observation_[0]][observation_[1]] = 1

            prayCoor = env.getFruitCoord()
            prayCoor[0] = prayCoor[0] / 70
            prayCoor[1] = prayCoor[1] / 70
            prayCoor = list(map(lambda x: int(x), prayCoor))
            s_[prayCoor[0]][prayCoor[1]] = 2

            s_=trans_torch(s_)


            env.render()
            # 存记忆
            dqn.store_transition(s, a, r, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                if study==1:
                    print('2000经验池')
                    study=0
                dqn.learn() # 记忆库满了就进行学习
            step += 1
            s = s_
        print('The episode: %d,  with %d steps.' % (i_episode, step))
        step_list.append(step)
    print('over')  # end of game
    env.destroy()



def plot_step():
    plt.plot(list(np.arange(EPISODES) + 1), step_list)
    plt.xlabel('Episodes')
    plt.ylabel('Step')
    plt.show()

if __name__ == "__main__":
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env = Maze()
    dqn = DQN()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    #MM = MultipleModel(actions=list(range(env.n_actions)), n_modules=list(range(N_MODULE)))
    env.after(500, update)
    env.mainloop()
    plot_step()

