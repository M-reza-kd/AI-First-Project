# multiAgents.py
# --------------
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
import time
from collections import deque
from pprint import pprint

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        print("legalMoves:", legalMoves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        for i in range(len(scores)):
            print(legalMoves[i], " : ", scores[i])
        print(legalMoves[chosenIndex])
        print("after action:")
        print(gameState)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        # print("currentGameState:\n", currentGameState, "\n*****\n")
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"
        mapStr = successorGameState.__str__()
        GameMap = mapStr.splitlines()
        GameMap.pop()
        GameMap.reverse()
        num_rows = len(GameMap)
        num_cols = len(GameMap[0])

        # Define the directions: up, down, left, right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        def find_nearest_food(start_point):
            # Initialize the visited set and the queue for BFS
            visited = set()
            queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

            print("start_point : ", start_point[1], start_point[0])
            while queue:
                current_point, distance = queue.popleft()
                row, col = current_point
                # Check if the current point contains food

                if row < num_rows and col < num_cols and (GameMap[row][col] == '.' or GameMap[row][col] == 'o'):
                    return distance

                # Explore the neighbors in all four directions
                for direction in directions:
                    new_row = row + direction[0]
                    new_col = col + direction[1]

                    # Check if the new position is within the map boundaries and not a wall
                    if 0 <= new_row < num_rows and 0 <= new_col < num_cols and GameMap[new_row][new_col] != '%' and \
                            GameMap[new_row][new_col] != 'G':
                        new_point = (new_row, new_col)

                        # Check if the new point has not been visited before
                        if new_point not in visited:
                            visited.add(new_point)
                            queue.append((new_point, distance + 1))

            # If no food is found, return [] to indicate that there is no reachable food
            return 100

        def find_nearest_ghost(start_point):
            visited = set()
            queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

            print("start_point : ", start_point[1], start_point[0])
            while queue:
                current_point, distance = queue.popleft()
                row, col = current_point
                # Check if the current point contains food

                if row < num_rows and col < num_cols and GameMap[row][col] == 'G':
                    return distance

                # Explore the neighbors in all four directions
                for direction in directions:
                    new_row = row + direction[0]
                    new_col = col + direction[1]

                    # Check if the new position is within the map boundaries and not a wall
                    if 0 <= new_row < num_rows and 0 <= new_col < num_cols and GameMap[new_row][new_col] != '%':
                        new_point = (new_row, new_col)

                        # Check if the new point has not been visited before
                        if new_point not in visited:
                            visited.add(new_point)
                            queue.append((new_point, distance + 1))

            # If no food is found, return [] to indicate that there is no reachable food
            return 100

        nearest_food = find_nearest_food(newPos)
        score = 1 / nearest_food
        nearest_ghost = find_nearest_ghost(newPos)
        if nearest_ghost <= 2:
            score += -100
        return score + successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state


        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def mini_max(game_state, remind_depth, agent_num):
            if remind_depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            legalMoves = game_state.getLegalActions(agent_num)

            if agent_num == 0:
                max_eval_score = float('-inf')
                for move in legalMoves:
                    successorGameState = game_state.generateSuccessor(agent_num, move)
                    max_eval_score = max(max_eval_score, mini_max(successorGameState, remind_depth - 1, 1))
                return max_eval_score
            else:
                num_agents = game_state.getNumAgents()
                min_eval_score = float('inf')
                if agent_num + 1 == num_agents:
                    remind_depth -= 1
                for move in legalMoves:
                    successorGameState = game_state.generateSuccessor(agent_num, move)
                    min_eval_score = min(min_eval_score, mini_max(successorGameState, remind_depth, (agent_num + 1) % num_agents))
                return min_eval_score

        legalActions = gameState.getLegalActions(0)
        scores = []

        for action in legalActions:
            scores.append(mini_max(gameState.generateSuccessor(0, action), self.depth, 0))

        max_score = max(scores)
        bestActions = [legalActions[ind] for ind in range(len(scores)) if scores[ind] == max_score]

        return random.choice(bestActions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
