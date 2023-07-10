# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

"""
    The Coursework of 6CCS3AIN Artificial Intelligence Reasoning and Desicion Making.
    an MDP-Solver for pacman to make a decision in the maze.
    Written by Weimin Meng, King's College London
    K21022587, MSc Artificial Intelligence, weimin.meng@kcl.ac.uk
    26/11/2021
"""

from pacman import Directions
from game import Agent
import api
import random
import game
import util

import csv

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        # print "Starting up MDP1Agent!"
        self.name = "Pacman"

        # the variables records the neccessary state of the layout(type, size).
        self.type = 'small'
        self.row = 6
        self.col = 6

        # the variables records the neccessary state of the map(whole map(dic set),
        # food, ghosts, walls, pacman)
        self.map = {}
        self.food = []
        self.capsules = []
        self.ghosts = []
        self.walls = []

        # the variables records the parameters of Bellman Equation(reward, gama, iteration time, ghost near distance).
        self.reward = 0
        self.gamma = 0
        self.iteration = 0
        self.area = 1

        self.flag = True
    # Gets run after an MDP1Agent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        """
        The function registers an initial state for a game.

        Parameters:
            -state: The current state of the pacman MDP1Agent.

        Returns:

        """
        # read the size and the type of the maze.
        self.row = api.corners(state)[-1][0]
        self.col = api.corners(state)[-1][1]
        if self.row > 6 or self.col > 6:
            self.type = 'medium'

        # initialize the food and walls on the map.
        self.walls = api.walls(state)

        # initial the Bellman Equation parameters.
        self.reward = -1
        self.gamma = 0.8
        if self.type == 'small':
            self.iteration = 20
        else:
            self.iteration = 40
            self.area = 3

    # This is what gets run in between multiple games
    def final(self, state):
        # clear the record variables.
        self.food = []
        self.capsules = []
        self.ghosts = []
        self.walls = []
        print "Looks like the game just ended!"

    # calculate the utility from the probability and the input utility map for each position.
    def cal_utilities(self, pos):
        """
            The function calculate each coordinate in the map.

                Parameters:
                    -pos: The current coordinate needed to be calculated the utility.
                Returns:
                    -The consequence of the agent action.
        """
        # 0: north, 1: east, 2: south, 3:west
        utilities = [0, 0, 0, 0]
        north_pos = (pos[0], pos[1] + 1)
        east_pos = (pos[0] + 1, pos[1])
        south_pos = (pos[0], pos[1] - 1)
        west_pos = (pos[0] - 1, pos[1])

        # direction: North, left west and right east are all 0.1 possible to move.
        if self.map[north_pos] != -5: utility = 0.7 * self.map[north_pos]
        else: utility = 0.7 * self.map[pos]
        if self.map[west_pos] != -5: utility += 0.1 * self.map[west_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[east_pos] != -5: utility += 0.1 * self.map[east_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[south_pos] != -5: utility += 0.1 * self.map[south_pos]
        else: utility += 0.1 * self.map[pos]
        utilities[0] = utility

        # direction: East, left north and right south are all 0.1 possible to move.
        if self.map[east_pos] != -5: utility = 0.7 * self.map[east_pos]
        else: utility = 0.7 * self.map[pos]
        if self.map[north_pos] != -5: utility += 0.1 * self.map[north_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[south_pos] != -5: utility += 0.1 * self.map[south_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[west_pos] != -5: utility += 0.1 * self.map[west_pos]
        else: utility += 0.1 * self.map[pos]
        utilities[1] = utility

        # direction: South, left east and right west are all 0.1 possible to move.
        if self.map[south_pos] != -5: utility = 0.7 * self.map[south_pos]
        else: utility = 0.7 * self.map[pos]
        if self.map[east_pos] != -5: utility += 0.1 * self.map[east_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[west_pos] != -5: utility += 0.1 * self.map[west_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[north_pos] != -5: utility += 0.1 * self.map[north_pos]
        else: utility += 0.1 * self.map[pos]
        utilities[2] = utility

        # direction: West, left South and right North are all 0.1 possible to move.
        if self.map[west_pos] != -5: utility = 0.7 * self.map[west_pos]
        else: utility = 0.7 * self.map[pos]
        if self.map[south_pos] != -5: utility += 0.1 * self.map[south_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[north_pos] != -5: utility += 0.1 * self.map[north_pos]
        else: utility += 0.1 * self.map[pos]
        if self.map[east_pos] != -5: utility += 0.1 * self.map[east_pos]
        else: utility += 0.1 * self.map[pos]
        utilities[3] = utility

        return utilities

    # value iteration of MDP
    def value_iteration(self, state):
        """
        The function using Bellman Equation calculates the value in the map.

        Parameters:
            -state: The current state of the pacman MDP1Agent.

        Returns:
        """
        # read the map of one action, for food, set the reward as 5, for wall, set
        # the reward of -5. if the pacman has already eaten the food, then set the
        # reward as 0, if the ghost is on the coordinate, then set it as -100.
        update_area = set()

        self.ghosts = api.ghosts(state)
        self.food = api.food(state)
        self.capsules = api.capsules(state)

        self.map = {}
        self.map.update(dict.fromkeys(self.food, 5))
        self.map.update(dict.fromkeys(self.walls, -5))
        self.map.update(dict.fromkeys(self.ghosts, -100))

        for i in range(self.row):
            for j in range(self.col):
                if (i, j) not in self.map.keys():
                    self.map[(i, j)] = 0

        tmp_map1 = str(self.map)

        # For each ghost, set the coordinates of all non-walls within the distance of 3 steps or 1 step of it
        # to 4*distance^2-64 for the medium size and -10*(5-distance) for the small size
        # in order to recalculate their reward value. !!!wall
        if self.type == 'small':
            for g in self.ghosts:
                ghost=(int(g[0]), int(g[1]))

                # produce the ghost area.
                ghost_area = set()
                for i in range(ghost[0]-self.area, ghost[0]+self.area+1):
                    for j in range(ghost[1]-self.area, ghost[1]+self.area+1):
                        if i > 0 and j > 0 and i < self.row and j < self.col:
                            if i < ghost[0] and j < ghost[1]: # down-left area
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m,j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(j, ghost[1]):
                                    if (i,n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i,j)])
                            elif i < ghost[0] and j == ghost[1]:  #left
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i < ghost[0] and j > ghost[1]: #up-left
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(ghost[1]+1, j+1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j > ghost[1]:  # up
                                wall_flag = False
                                for n in range(ghost[1] + 1, j+1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j > ghost[1]: #up-right
                                wall_flag = False
                                for m in range(ghost[0]+1, i+1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(ghost[1]+1, j+1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j == ghost[1]:  # right
                                wall_flag = False
                                for m in range(ghost[0] + 1, i+1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j < ghost[1]: #down-right
                                wall_flag = False
                                for m in range(ghost[0]+1, i+1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j < ghost[1]:  # down
                                wall_flag = False
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            else: # itself
                                ghost_area.update([(i, j)])

                # update the map
                update_area  = update_area | ghost_area
                for area in ghost_area:
                    if area not in self.ghosts and area not in self.walls:
                        distance = max(abs(ghost[0] - area[0]), abs(ghost[1] - area[1]))
                        penalty = -100 / (2 * distance) #4 * (distance ** 2) - 64 #
                        if penalty < self.map[area]:
                            self.map[area] = penalty

            update_area = set(self.food) - update_area
        else:
            for g in self.ghosts:
                ghost=(int(g[0]), int(g[1]))
                ghost_area = set()
                for i in range(ghost[0] - self.area, ghost[0] + self.area + 1):
                    for j in range(ghost[1] - self.area, ghost[1] + self.area + 1):
                        if i > 0 and j > 0 and i < self.row and j < self.col:
                            if i < ghost[0] and j < ghost[1]:  # down-left area
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i < ghost[0] and j == ghost[1]:  # left
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i < ghost[0] and j > ghost[1]:  # up-left
                                wall_flag = False
                                for m in range(i, ghost[0]):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(ghost[1] + 1, j + 1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j > ghost[1]:  # up
                                wall_flag = False
                                for n in range(ghost[1] + 1, j + 1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j > ghost[1]:  # up-right
                                wall_flag = False
                                for m in range(ghost[0] + 1, i + 1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(ghost[1] + 1, j + 1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j == ghost[1]:  # right
                                wall_flag = False
                                for m in range(ghost[0] + 1, i + 1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j < ghost[1]:  # down-right
                                wall_flag = False
                                for m in range(ghost[0] + 1, i + 1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j < ghost[1]:  # down
                                wall_flag = False
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    ghost_area.update([(i, j)])
                            else:  # itself
                                ghost_area.update([(i, j)])

                # update the map
                update_area = update_area | ghost_area
                for area in ghost_area:
                    if area not in self.ghosts and area not in self.walls:
                        distance = max(abs(ghost[0] - area[0]), abs(ghost[1] - area[1]))
                        penalty = -10 * (6 - distance)  # 4 * (distance ** 2) - 64 #
                        if penalty < self.map[area]:
                            self.map[area] = penalty

            update_area = set(self.food) - update_area

        tmp_map2 = str(self.map)

        # calculate the max utility and substitute into Bellman Equation, in order to
        # get the new utilities map.
        for l in range(self.iteration):
            for i in range(self.row):
                for j in range(self.col):
                    if (i,j) not in self.walls and (i,j) not in update_area and (i,j) not in self.ghosts and (i,j) not in self.capsules:
                        utilities = self.cal_utilities((i, j))
                        # Take the maximum utility in four directions as the utility at (i,j) and update the map.
                        self.map[(i, j)] = self.reward + self.gamma * max(utilities)

        # with open('data.csv', 'ab') as csvfile:
        #     fieldnames=['pacman','ghosts','food','capsules','walls','noup','map','ghost_map','iteration']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     if self.flag:
        #         writer.writeheader()
        #         self.flag = False
        #     writer.writerow({'pacman':api.whereAmI(state),'ghosts':api.ghosts(state)
        #                      ,'food':api.food(state),'capsules':api.capsules(state)
        #                      ,'walls':api.walls(state),'noup':update_area
        #                      ,'map':tmp_map1,'ghost_map':tmp_map2
        #                      ,'iteration':self.map})

    # Choose the Direction by the max utility of the pacman.
    def MoveDirection(self, state):
        """Get the maximum utility to make move decision

        Args:
            state: The state of an agent (configuration, speed, scared, etc).
            map: The value map before update

        Returns:
            The move direction decision
        """
        # according to the pacman coordinate, give the max utility.
        pacman = api.whereAmI(state)
        utilities = self.cal_utilities(pacman);
        index = utilities.index(max(utilities))

        # Return the decision direction by the max utility
        if index == 0: return Directions.NORTH
        if index == 1: return Directions.EAST
        if index == 2: return Directions.SOUTH
        if index == 3: return Directions.WEST

    # For now I just move randomly
    def getAction(self, state):
        """
            The function makes decision of action in each step of the pacman MDP1Agent.

            Parameters:
                -state: The current state of the pacman MDP1Agent.
            Returns:
                -The consequence of the agent action.
        """
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # get the utilities map and make a decision from the max utility in this main function by MD
        self.value_iteration(state)

        # choose the direction between utility map and the legal options.
        return api.makeMove(self.MoveDirection(state), legal)
