# -*- coding: UTF-8 -*-
"""

"""

import numpy as np
import time
import sys
from  PIL import Image, ImageTk, ImageGrab
import tkinter as tk
import itertools as it
import random

UNIT = 70   # pixels
MAZE_H = 7  # grid height
MAZE_W = 7  # grid width
INTERVAL = 5
window_width = MAZE_W*UNIT
window_height = MAZE_H*UNIT

x_range = range(0, MAZE_H)
y_range = range(0, MAZE_W)
all_locations = np.array(list(it.product(x_range,y_range)))*UNIT
all_locations = all_locations.tolist()

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)

        self.prey_move_directions = ['ne', 'nw', 'se', 'sw']
        # the move direction of the prey, at the beginning of each trial, one of 4 directions is randomly selected
        # and a prey is placed at a random position in the grid world.

        self.title('MMRL-Maze')
        x_cordinate = int((self.winfo_screenwidth() / 2) - (window_width / 2))
        y_cordinate = int((self.winfo_screenheight() / 2) - (window_height / 2))
        self.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        #self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',height=MAZE_H * UNIT,width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):  # create |
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):  # create --
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        """
        # create origin
        origin = np.array([20, 20])
        
        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        
        
        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect -- the agent
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        """

        # create the agent and the fruit(or the prey)
        agent_image = Image.open("C:\\Users\\Administrator\\Desktop\\tutorials\\tkinter\\Q_Learning_maze-MF-1\\images\\agent.png")
        agent_image = agent_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.agent_img = ImageTk.PhotoImage(agent_image)
        self.agent = self.canvas.create_image(0, 0, anchor="nw", image=self.agent_img)

        # fruit_image = Image.open(interaction_images['orange'])
        fruit_image = Image.open("C:\\Users\\Administrator\\Desktop\\tutorials\\tkinter\\Q_Learning_maze-MF-1\\images\\orange.png")
        fruit_image = fruit_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.fruit_img = ImageTk.PhotoImage(fruit_image)
        # 这里猎物的位置要随机出现， 同时猎物的移动方向也确定了下来
        all_locations.remove([0, 0])
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations)-1)]
        self.fruit = self.canvas.create_image(x_prey, y_prey, anchor="nw", image=self.fruit_img)
        self.prey_direction = self.prey_move_directions[random.randint(0, 3)]
        #self.prey_direction = 'sw'

        # 初始化 wall 的图片
        wall_image = Image.open("C:\\Users\\Administrator\\Desktop\\tutorials\\tkinter\\Q_Learning_maze-MF-1\\images\\wall.png")
        wall_image = wall_image.resize((UNIT - 2, UNIT - 2), Image.ANTIALIAS)
        self.wall_img = ImageTk.PhotoImage(wall_image)

        # bind the function
        self.canvas.bind("<Button-1>", lambda event: self.drawRect(event))

        # pack all
        self.canvas.pack()

    def drawRect(self, event):
        click_x = (event.x // UNIT) * UNIT
        click_y = (event.y // UNIT) * UNIT
        click_x_color = click_x + UNIT // 2
        click_y_color = click_y + UNIT // 2
        color = ImageGrab.grab().getpixel((event.x_root - event.x + click_x_color, event.y_root - event.y + click_y_color))
        if color[0] == 51:  # 这块的处理方式需要考虑一下，看能不能尝试使用 del 的方法来处理
            #print('get the point')
            self.canvas.create_rectangle(click_x+1, click_y+1, click_x+UNIT-1, click_y+UNIT-1, fill='white', outline='white')
        else:
            self.canvas.create_image(click_x+1, click_y+1, anchor="nw", image=self.wall_img)


    def reset(self):
        #self.update()
        #time.sleep(1)
        self.canvas.delete(self.agent)  # reset the position of the agent and update the canvas
        self.agent = self.canvas.create_image(0, 0, anchor="nw", image=self.agent_img)
        self.update()
        time.sleep(1)
        #print('reset is done')
        return self.canvas.coords(self.agent)  # return observation

    # 当赋予agent一个动作的时候，需要agent去执行这个动作，并且给与反馈
    def step(self, action):
        s = self.canvas.coords(self.agent)  # 获取当前agent在maze当中的坐标
        base_action = np.array([0, 0])
        if action == 0:   # north
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # south
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # east
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # west
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 4:   # stay
            pass
        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        # move prey
        fruit_coords = list(map(int, self.canvas.coords(self.fruit)))
        base_fruit = np.array([0, 0])
        if   self.prey_direction == 'ne':
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == 0:
                base_fruit[0] = fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'nw':
            if (fruit_coords[0]==0 and fruit_coords[1]==(MAZE_H-1)*UNIT) or (fruit_coords[0]==(MAZE_W-1)*UNIT and fruit_coords[1]==0):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == 0:
                base_fruit[0] = (MAZE_H-1)*UNIT-fruit_coords[1]-fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT-fruit_coords[0]-fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] -= UNIT
        elif self.prey_direction == 'se':
            if (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == 0) or (fruit_coords[0] == 0 and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == (MAZE_W-1)*UNIT or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = (MAZE_H-1)*UNIT - fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = (MAZE_W-1)*UNIT - fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] += UNIT
                base_fruit[1] += UNIT
        elif self.prey_direction == 'sw':  # testici
            if (fruit_coords[0] == 0 and fruit_coords[1] == 0) or (fruit_coords[0] == (MAZE_W-1)*UNIT and fruit_coords[1] == (MAZE_H-1)*UNIT):
                pass
            elif fruit_coords[0] == 0 or fruit_coords[1] == (MAZE_H-1)*UNIT:
                base_fruit[0] = fruit_coords[1] - fruit_coords[0]
                base_fruit[1] = fruit_coords[0] - fruit_coords[1]
            else:
                base_fruit[0] -= UNIT
                base_fruit[1] += UNIT
        self.canvas.move(self.fruit, base_fruit[0], base_fruit[1])  # move fruit

        #print(self.canvas.coords(self.agent))
        #print(self.canvas.coords(self.fruit))

        s_ = self.canvas.coords(self.agent)  # next state
        if s_ == self.canvas.coords(self.fruit):  # reward function
            reward = 10
            done = True
            s_ = 'terminal'
        else:
            reward = -1
            done = False
        return s_, reward, done

    def render(self):
        self.update()  # 执行一个动作之后对于画布的刷新
        time.sleep(0.1)  # 给画布更新足够的时


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()