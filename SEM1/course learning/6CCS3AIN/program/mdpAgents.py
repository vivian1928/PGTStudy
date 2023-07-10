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

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    """
        The Coursework of 6CCS3AIN Artificial Intelligence Reasoning and Desicion Making.
        an MDP-Solver for pacman to make decisions in the maze.
        Written by Weimin Meng, King's College London
        K21022587, MSc Artificial Intelligence, weimin.meng@kcl.ac.uk
        26/11/2021

        References:
         -Books:
            K. Leyton-Brown and Y. Shoham, Essentials of game theory. San Rafael, Calif: Morgan & Claypool Publishers, 2010, pp. 53-55.
            S. Russell and P. Norvig, Artificial intelligence_ A Modern Approach. Harlow: Pearson, 2021, pp. 557-566.
         -Codes:
            M. Chen, "GitHub - wpddmcmc/Pacman_MDP: Markov Decision Process solver for PACMAN", GitHub, 2021. [Online]. Available: https://github.com/wpddmcmc/Pacman_MDP.
            "Berkeley AI Materials", Ai.berkeley.edu, 2021. [Online]. Available: http://ai.berkeley.edu/reinforcement.html.
    """

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):

        """
            The constructor for the MDPAgent gets run when first invoke pacman.py.

            Parameters:
                None
            Returns:
                None
        """

        self.name = "Pacman"

        # The variables records the neccessary state of the map(type: default smallGrid,
        # size: 6 * 6).
        self.type = '' # The type of the map, 'small' or 'medium'.
        self.row = 0 # The row number of the map, 6 or 19.
        self.col = 0 # The column number of the map, 6 or 10.

        # The variables records the neccessary elements in the state of the map( map(dict),
        # food, capsules, ghosts, normalghosts, edibleghosts, walls)
        self.map = {} # A dict of the map, e.g.{(1, 1):0, (1, 2): -5, (1, 3): -5, ... ,(self.row - 1, self.col - 1): 0}
        self.food = [] # The list of the food in one state, e.g.[(1, 1),(3, 3)]
        self.capsules = [] # The list of the capsules in one state, e.g.[(1, 10),(19, 1)]
        self.ghosts = [] # The list of the ghosts in one state, e.g.[(8, 5),(11, 5)]
        self.normalghosts = [] # The list of the normal ghosts in one state, e.g.[(8, 5),(11, 5)]
        self.edibleghosts = [] # The list of the edible ghosts in one state, e.g.[(8, 5),(11, 5)]
        self.walls = []

        # the variables records the parameters of Bellman Equation(reward, gamma, numbers of iteration
        # , range of the ghosts fields).
        self.reward = 0
        self.gamma = 0
        self.iteration = 0
        self.area = 0

    # Gets run after an MDP1Agent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        """
            The function gets run after an MDPAgent is created and
            registers an initial state for one game.

            Parameters:
                None
            Returns:
                None

        """

        # read the size and the type of the current maze.
        self.row = api.corners(state)[-1][0]
        self.col = api.corners(state)[-1][1]
        if self.row > 6 and self.col > 6:
            self.type = 'medium'
        else:
            self.type = 'small'

        # initialize the fixed walls on the map.
        self.walls = api.walls(state)

        # initialize the Bellman Equation parameters, for smallGrid, the size of ghosts fields
        # is 1, and 3 for mediumClassic.
        self.reward = -1
        self.gamma = 0.8
        if self.type == 'small':
            self.iteration = 25
            self.gamma = 0.5
            self.area = 1
        else:
            self.iteration = 50
            self.gamma = 0.8
            self.area = 3

    # This is what gets run in between multiple games.
    def final(self, state):

        """
            The final function runs between multiple games.

            Parameters:
                None
            Returns:
                None
        """

        # Clear the record variables.
        self.food = []
        self.capsules = []
        self.ghosts = []
        self.normalghosts = []
        self.edibleghosts = []
        self.walls = []
        # print "Looks like the game just ended!"

    # calculate the utility from the probability and the input utility map for each position.
    def cal_utilities(self, pos):
        """
            The function calculate each coordinate's expected utility in the map.

            Parameters:
                -pos(tuple:(x, y)): The current coordinate needed to be calculated the expected utility.
            Returns:
                -utilities(list: 0-north, 1-east, 2-south, 3-west): The expected utility of
                the agent's action in 4 direction.
        """
        # The position of the list represents the expected utility in that direction
        # 0: north, 1: east, 2: south, 3:west
        utilities = [0, 0, 0, 0]
        north_pos = (pos[0], pos[1] + 1)
        east_pos = (pos[0] + 1, pos[1])
        south_pos = (pos[0], pos[1] - 1)
        west_pos = (pos[0] - 1, pos[1])

        # According to the reference codes and some experiments, the forward probability
        # is set to 0.7, and the backward, the left and the right probabilities are 0.1

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
            The function using Bellman Equation calculates the value iteration in the map.

            Parameters:
                -state: The current state of the pacman MDP1Agent.
            Returns:
                None.
        """

        # Read the map of one action, in general, for food, set the reward as 5, for wall, set
        # the penalty of -5. If the pacman has already eaten the food, then set the
        # reward as 0, if the ghost is on the coordinate, then set it as -100.

        # Read the situations of the ghosts.
        update_area = [] # Record the size of the area near the ghost that needs to be updated in the Bellman Equation.
        self.ghosts = [] # Record all the ghosts.
        ghosts_list = api.ghosts(state)
        for g in ghosts_list: # The ghosts have decimal situations and need to be set to integer coordinates.
            self.ghosts.append((int(g[0]), int(g[1])))
        self.normalghosts = [] # Record the normal ghosts.
        self.edibleghosts = [] # Record the edible ghosts.
        self.ghoststates = api.ghostStates(state)
        for gs in self.ghoststates: # Read the current state of the ghosts.
            if gs[1] == 0:
                self.normalghosts.append((int(gs[0][0]),int(gs[0][1])))
            else:  #edible ghosts
                self.edibleghosts.append((int(gs[0][0]),int(gs[0][1])))

        # Read the situations of the food and capsules, since capsules and
        # food are regarded as the same, so put them together.
        self.food = api.food(state)
        self.capsules = api.capsules(state)
        for capsule in self.capsules:
            self.food.append(capsule)
        self.pacman = api.whereAmI(state)

        # Set the initial state of the map: the reward and the penalty.
        self.map = {} # The form of the map is dict, e.g. {(3, 3): 5}

        # Food Set.
        # In smallGrid, since eating (3,3) points first and entering the pit on the left
        # may be eaten by ghosts, so set (1,1) points the highest priority.
        if self.type == 'small':
            self.pacman = api.whereAmI(state)
            # If (1,1) is still in the food, then it needs to set (1,1) as the highest priority to eat.
            if (1, 1) in self.food:
                for food in self.food:
                    if food == (1, 1):
                        self.map[food] = 5
                    else:
                        self.map[food] = -5
            else:
                for food in self.food:
                    self.map[food] = 5
        else:
            # In the mediumClassic, If there are edible ghosts, priorly to eat the edible ghosts rather than food.
            if len(self.food) == 1 and min(
                    ((self.ghosts[0][0] - self.food[0][0]) ** 2 + (self.ghosts[0][1] - self.food[0][1]) ** 2) ** 0.5, \
                    ((self.ghosts[1][0] - self.food[0][0]) ** 2 + (self.ghosts[1][1] - self.food[0][1]) ** 2) ** 0.5) \
                    > (((self.pacman[0] - self.food[0][0]) ** 2 + (self.pacman[1] - self.food[0][1]) ** 2) ** 0.5):
                self.ghosts = []
                self.edibleghosts = []
                self.normalghosts = []

            if len(self.edibleghosts) > 0:
                self.map.update(dict.fromkeys(self.food, 0))
            else:
                self.map.update(dict.fromkeys(self.food, 5))

        # Walls Set.
        self.map.update(dict.fromkeys(self.walls, -5))

        # Ghosts Set.
        # For normal ghosts, all need to be set to -100 and 5 for the edible ghosts same as the food.
        if len(self.edibleghosts) > 0:
            self.map.update(dict.fromkeys(self.edibleghosts, 5))
            self.map.update(dict.fromkeys(self.normalghosts, -100))
        else:
            self.map.update(dict.fromkeys(self.ghosts, -100))

        # Others Set.
        # For paths that have no food, or other locations, set to 0.
        for i in range(self.row):
            for j in range(self.col):
                if (i, j) not in self.map.keys():
                    self.map[(i, j)] = 0

        # For each ghost, set the coordinates of all non-walls within the distance of 1 step of it
        # to -100/(2*distance) for the small size maze, i.e.for the locations of one grid around the
        # ghost that are not walls, set them to -50.
        # SmallGrid map.
        if self.type == 'small':
            for g in self.ghosts:
                ghost=(int(g[0]), int(g[1]))

                # According to the size, traverse all the coordinates around the ghost,
                # and record the ghost area and the update area, in which the max distance of
                # the ghost area is 1, the size of the update area is 1*1.
                ghost_area = set()
                for i in range(ghost[0]-self.area, ghost[0]+self.area+1):
                    for j in range(ghost[1]-self.area, ghost[1]+self.area+1):
                        if i > 0 and j > 0 and i < self.row and j < self.col:
                            if i < ghost[0] and j < ghost[1]: # Down-left area.
                                wall_flag = False
                                for m in range(i, ghost[0]): # Check from left to right if it is blocked by a wall.
                                    if (m,j) in self.walls:
                                        wall_flag = True
                                        break
                                for n in range(j, ghost[1]): # Check from down to up if it is blocked by a wall.
                                    if (i,n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag: # Record the updated area and the ghost field.
                                    update_area.append((i, j))
                                    ghost_area.update([(i,j)])
                            elif i < ghost[0] and j == ghost[1]:  # Left area.
                                wall_flag = False
                                for m in range(i, ghost[0]): # Check from left to right if it is blocked by a wall.
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag: # Record the updated area and the ghost field.
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i < ghost[0] and j > ghost[1]: # Up-left area.
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
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j > ghost[1]:  # Up area.
                                wall_flag = False
                                for n in range(ghost[1] + 1, j+1):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j > ghost[1]: # Up-right area.
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
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j == ghost[1]:  # Right area.
                                wall_flag = False
                                for m in range(ghost[0] + 1, i+1):
                                    if (m, j) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i > ghost[0] and j < ghost[1]: # Down-right area.
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
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            elif i == ghost[0] and j < ghost[1]:  # Down area.
                                wall_flag = False
                                for n in range(j, ghost[1]):
                                    if (i, n) in self.walls:
                                        wall_flag = True
                                        break
                                if not wall_flag:
                                    update_area.append((i, j))
                                    ghost_area.update([(i, j)])
                            else: # Ghost itself.
                                update_area.append((i, j))
                                ghost_area.update([(i, j)])

                # Set the coordinate penalty value that is inversely proportional
                # to the distance in the ghosts field.
                for area in ghost_area:
                    if area not in self.ghosts and area not in self.walls:
                        distance = max(abs(ghost[0] - area[0]), abs(ghost[1] - area[1]))
                        penalty = -100 / (2 * distance)
                        if penalty < self.map[area]: # For multiple ghosts, the smaller the more representative the penalty value of the current coordinate.
                            self.map[area] = penalty

            noup = [] # The food locations that have not been updated.
            for f in self.food:
                if f not in update_area:
                    noup.append(f)
        # MediumClassic map.
        else:
            # For normal ghosts, set all coordinates on its vertical, horizontal, and
            # the two diagonals to -10 * (6-distance) according to the distance as the
            # ghost area, and set a 4 * 4 update area of the update area.
            for g in self.normalghosts:
                ghost=(int(g[0]), int(g[1]))
                ghost_area = set()

                for i in range(ghost[0] - self.area - 1, ghost[0] + self.area + 2):
                    for j in range(ghost[1] - self.area - 1, ghost[1] + self.area + 2):
                        if i > 0 and j > 0 and i < self.row and j < self.col:
                            update_area.append((i, j)) # 4 * 4 update area.
                            if i == ghost[0] or j == ghost[1] or abs(ghost[0] - i) == abs(ghost[1] - j):  # Vertical, horizontal, two diagonals.
                                if abs(ghost[0] - i) <= 3 and abs(ghost[1] - j) <= 3: # The max distance is 3.
                                    ghost_area.update([(i, j)])

                # Set the coordinate penalty value that is -10 * (6 - distance)
                # i.e. -30, -40, -50.
                for area in ghost_area:
                    if area not in self.ghosts and area not in self.walls:
                        distance = max(abs(ghost[0] - area[0]), abs(ghost[1] - area[1]))
                        penalty = -10 * (6 - distance)
                        if penalty < self.map[area]:
                            self.map[area] = penalty

            # If there are no edible ghosts, the surrounding ghosts are the update area.
            # If there are edible ghosts, the full map update ensures that the agent can find edible ghosts.
            noup = []
            if len(self.edibleghosts) == 0:
                for f in self.food:
                    if f not in update_area:
                        noup.append(f)

        # Calculate the max utility and substitute into Bellman Equation, in order to
        # get the new utilities map.
        for l in range(self.iteration):
            for i in range(self.row):
                for j in range(self.col):
                    # The non-wall, non-ghost, and non-capsule coordinates in the Needed-updated area.
                    if (i,j) not in self.walls and (i,j) not in noup and (i,j) not in self.ghosts and (i,j) not in self.capsules:
                        utilities = self.cal_utilities((i, j))
                        # Take the maximum utility in four directions as the utility at (i,j) and update the map.
                        self.map[(i, j)] = self.reward + self.gamma * max(utilities)


    # Choose the Direction by the max utility of the pacman.
    def getMove(self, state):
        """
            The function makes a move decision from the max utility of the pacman in the
            iterative expected utilities map.

            Parameters:
                None
            Returns:
                -Direction: The move direction decision.
        """
        # According to the pacman coordinate, call the function to calculate the expected utilities from four direction
        # of the pacman and find the max utility direction.
        pacman = api.whereAmI(state)
        utilities = self.cal_utilities(pacman)
        index = utilities.index(max(utilities))

        # Return the direction.
        if index == 0: return Directions.NORTH
        if index == 1: return Directions.EAST
        if index == 2: return Directions.SOUTH
        if index == 3: return Directions.WEST

    # For now I just move randomly
    def getAction(self, state):
        """
            The function makes decision of action in each step of the pacman MDPAgent.

            Parameters:
                -state: The current state of the pacman MDP1Agent.
            Returns:
                -Direction: The consequence action of the agent carries out.
        """
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Get the iterative utilities map and make a decision from the max utility in this main function by MDP.
        self.value_iteration(state)

        # Choose the direction between utility map and the legal options.
        return api.makeMove(self.getMove(state), legal)
