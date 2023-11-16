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


def find_nearest_food(start_point, GameMap, num_rows, num_cols):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = set()
    queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

    while queue:
        current_point, distance = queue.popleft()
        row, col = current_point
        # Check if the current point contains food

        if row < num_rows and col < num_cols and GameMap[row][col] == '.':
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

    # If no food is found, return 100 to indicate that there is no reachable food
    return 100


def find_nearest_ghost(start_point, GameMap, num_rows, num_cols):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = set()
    queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

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

    # If no food is found, return 100 to indicate that there is no reachable food
    return 100


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

    @staticmethod
    def evaluationFunction(currentGameState, action):
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
        GameMap = successorGameState.__str__().splitlines()[:-1][::-1]
        num_rows, num_cols = len(GameMap), len(GameMap[0])

        # print("currentGameState :", currentGameState.getScore(), "successorGameState :", successorGameState.getScore())
        score = successorGameState.getScore()
        nearest_food = find_nearest_food(newPos, GameMap, num_rows, num_cols)
        score += 1 / (nearest_food * 100)
        nearest_ghost = find_nearest_ghost(newPos, GameMap, num_rows, num_cols)
        newGhostStates = successorGameState.getGhostStates()
        minScaredTimes = min([ghostState.scaredTimer for ghostState in newGhostStates])
        if minScaredTimes == 0 and nearest_ghost > 0:
            score -= (15 / nearest_ghost)
        # print("nearest_food : ", nearest_food, "nearest_ghost : ", nearest_ghost, "score : ", score)
        return score


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
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def __init__(self, evalFn='betterEvaluationFunctionForMinimax', depth='3'):
        super().__init__(evalFn, depth)

    def mini_max(self, game_state, depth, agent_num):
        if game_state.isWin() or game_state.isLose() or (agent_num == 0 and depth == self.depth):
            return self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        legalMoves = game_state.getLegalActions(agent_num)
        next_agent = agent_num + 1
        best_action = Directions.STOP

        if agent_num == 0:
            max_eval_score = float('-inf')
            for move in legalMoves:
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.mini_max(successorGameState, depth, next_agent)
                if score > max_eval_score:
                    max_eval_score = score
                    best_action = move
            if depth == 0:
                # print("list of valid movement : ", legalMoves)
                # print("Score :", max_eval_score, "Best Action :", best_action)
                return best_action
            else:
                return max_eval_score

        else:
            min_eval_score = float('inf')
            if agent_num + 1 == num_agents:
                depth += 1
                next_agent = 0

            for move in legalMoves:
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.mini_max(successorGameState, depth, next_agent)
                min_eval_score = min(min_eval_score, score)
            return min_eval_score

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
        return self.mini_max(gameState, 0, 0)


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


def betterEvaluationFunctionForMinimax(currentGameState):
    newPos = currentGameState.getPacmanPosition()

    "*** YOUR CODE HERE ***"
    GameMap = currentGameState.__str__().splitlines()[:-1][::-1]
    num_rows, num_cols = len(GameMap), len(GameMap[0])

    # print("currentGameState :", currentGameState.getScore(), "successorGameState :", currentGameState.getScore())
    score = currentGameState.getScore()
    nearest_food = find_nearest_food(newPos, GameMap, num_rows, num_cols)
    score += 1 / (nearest_food * 100)
    nearest_ghost = find_nearest_ghost(newPos, GameMap, num_rows, num_cols)
    newGhostStates = currentGameState.getGhostStates()
    minScaredTimes = min([ghostState.scaredTimer for ghostState in newGhostStates])
    if minScaredTimes == 0 and 3 > nearest_ghost > 0:
        score -= (1 / nearest_ghost)
    # print("nearest_food : ", nearest_food, "nearest_ghost : ", nearest_ghost, "score : ", score)
    return score


# Abbreviation
better = betterEvaluationFunction
