
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#参数
BATCH_SIZE = 32
LR = 0.01                   # 学习率
EPSILON = 0.9               # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 2000      # 记忆库大小
N_ACTIONS = 5  # 棋子的动作0，1，2，3
N_STATES = 1




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1=nn.Conv2d(2,49,7,1,0)
        self.f1=nn.Linear(49,25)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2=nn.Linear(25,5)
        self.f2.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x=self.c1(x)
        x=F.relu(x)
        x = x.view(x.size(0),-1)
        x=self.f1(x)
        x=F.relu(x)
        action=self.f2(x)
        return action

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #DQN需要使用两个神经网络
        #eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0 # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0 # 记忆库记数
        self.memory = list(np.zeros((MEMORY_CAPACITY, 5))) # 初始化记忆库用numpy生成一个(2000,4)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample,x为场景
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x) #将场景输入Q估计神经网络
            #torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data.numpy() # 返回动作最大值
        else:   # 选随机动作
            action = np.array([np.random.randint(0, N_ACTIONS)]) # 比如np.random.randint(0,2)是选择1或0
        return action

    def store_transition(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s,a,r,s_]
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每100次
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s=[]
        b_a=[]
        b_r=[]
        b_s_=[]
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append(np.array(self.memory[i][1],dtype=np.int32))
            b_r.append(np.array([self.memory[i][2]],dtype=np.int32))
            b_s_.append(self.memory[i][3])

        #b_s = list(map(lambda x: int(x), b_s))
        b_s = torch.FloatTensor(b_s)#取出s
        b_a = torch.LongTensor(b_a) #取出a
        b_r = torch.FloatTensor(b_r) #取出r
        b_s_ = torch.FloatTensor(b_s_) #取出s_
        #print(b_s[0])
        #print(b_a[0])
        #print(self.eval_net(b_s))
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        #print('1')
        #print([b_s])
        # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()
        # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0]
        # shape (batch, 1) DQL核心公式
        loss = self.loss_func(q_eval, q_target) #计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad() #
        loss.backward() #反向传递
        self.optimizer.step()