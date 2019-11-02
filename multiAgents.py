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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        newFoodDistance = [manhattanDistance(f, newPos) for f in newFoodList]
        minFoodDist = min(newFoodDistance, default=0.001) if min(newFoodDistance, default=0.001) > 0 else 0.0001
        countFood = len(newFoodList) if len(newFoodList) > 0 else 0.001

        newGhostPositions = [g.getPosition() for g in newGhostStates]
        newGhostDistances = [manhattanDistance(g, newPos) for g in newGhostPositions]
        minGhostDist = min(newGhostDistances, default=0.001) if min(newGhostDistances, default=0.001) <= 4 else 4
        countGhosts = len(newGhostDistances)

        capsuleList = successorGameState.getCapsules()
        capsuledDistance = [manhattanDistance(f, newPos) for f in capsuleList]
        minCapsuledDistance = min(capsuledDistance, default=0.001) if min(capsuledDistance, default=0.001) > 0 else 1
        counCapsule= len(capsuleList) if len(capsuleList) > 0 else 0.001

        ScaredTimesFactor =  sum(newScaredTimes) - 5 * minGhostDist if sum(newScaredTimes) > 3 and minGhostDist < 4 else 1

        tieBreaker = random.gauss(2,0.05) * 0.000001 if action != 'Stop' else 0

        score = 1 / (countFood+counCapsule) + minGhostDist + ScaredTimesFactor - 0.0000001 * minFoodDist + tieBreaker\
                - 0.00000003 * minCapsuledDistance

        return score
        #return successorGameState.getScore()


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        #number_of_agents = gameState.getNumAgents()
        #actions_of_pacman = gameState.getLegalActions(0)
        #actions_of_first = gameState.getLegalActions(1)

        # gameState.generateSuccessor(agentIndex, action)

        is_win = gameState.isWin()
        is_lose = gameState.isLose()
        max_game_depth = self.depth
        evaluation_func = self.evaluationFunction

        agent_depth = 0

        if is_win or is_lose or agent_depth >= max_game_depth:
            return evaluation_func(gameState)

        #print(number_of_agents , actions_of_pacman, actions_of_first, is_win, is_lose, max_game_depth, score)

        #for agent_number in range(number_of_agents):
        #    legal_action_list = gameState.getLegalActions(agent_number)
        #    for legal_action in legal_action_list:
        #        new_gameState = gameState.generateSuccessor(agent_number, legal_action)
        #        new_gameState_is_win = new_gameState.isWin()
        #        if new_gameState_is_win:
        #            return legal_action

        #first move is for agent0
        agent_index = 0
        turn_depth = 0

        max_action = None
        max_value = float("-inf")

        legal_action_list = gameState.getLegalActions(0)
        for legal_action in legal_action_list:
            #print('pacman: ', agent_index, agent_depth, legal_action, max_action, max_value)
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index+1, turn_depth=turn_depth)
            #print('next action value: ', new_action_value)

            if max_action is None or max_value is None or max_value < new_action_value:
                #print('max_action:', max_action, ' value: ', max_value)
                max_action = legal_action
                max_value = new_action_value
        #print('max_action - value_for_pacman', max_action, max_value)
        return max_action

    def get_action_value(self, gameState, agent_index, turn_depth):
        max_game_depth = self.depth
        is_win = gameState.isWin()
        is_lose = gameState.isLose()
        evaluation_func = self.evaluationFunction

        if is_win or is_lose or turn_depth >= max_game_depth:
            return evaluation_func(gameState)
        elif agent_index == 0:
            #print('max_agent_called: ',  agent_index, turn_depth)
            return self.get_value_for_max_agent(gameState, agent_index, turn_depth)
        else:
            #print('min_agent_called: ',  agent_index, turn_depth)
            return self.get_value_for_min_agent(gameState, agent_index, turn_depth)

    def get_value_for_max_agent(self, gameState, agent_index, turn_depth):
        #next_move_values = list()
        next_move_values = float("-inf")

        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1, turn_depth=turn_depth)
            next_move_values = max(next_move_values, new_action_value)
        return next_move_values

    def get_value_for_min_agent(self, gameState, agent_index, turn_depth):
        next_move_values = float("inf")
        last_agent_index = gameState.getNumAgents() - 1
        # next_move_values = list()
        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)

            if agent_index == last_agent_index:
                new_action_value = self.get_action_value(new_game_state, agent_index=0,
                                                         turn_depth=turn_depth+1)
            else:
                new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1,
                                                         turn_depth=turn_depth)
            # next_move_values.append(new_action_value)
            next_move_values = min(next_move_values, new_action_value)

        # return min(next_move_values)
        return next_move_values


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        agent_index = 0
        turn_depth = 0

        max_action = None
        max_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        legal_action_list = gameState.getLegalActions(0)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index+1, turn_depth=turn_depth,
                                                     alpha=alpha, beta=beta)

            if max_action is None or max_value is None or max_value < new_action_value:
                max_action = legal_action
                max_value = new_action_value
            alpha = max(max_value, alpha)
        return max_action

    def get_action_value(self, gameState, agent_index, turn_depth, alpha, beta):
        max_game_depth = self.depth
        is_win = gameState.isWin()
        is_lose = gameState.isLose()
        evaluation_func = self.evaluationFunction

        if is_win or is_lose or turn_depth >= max_game_depth:
            return evaluation_func(gameState)
        elif agent_index == 0:
            return self.get_value_for_max_agent(gameState, agent_index, turn_depth, alpha, beta)
        else:
            return self.get_value_for_min_agent(gameState, agent_index, turn_depth, alpha, beta)

    def get_value_for_max_agent(self, gameState, agent_index, turn_depth, alpha, beta):
        next_move_values = float("-inf")

        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1, turn_depth=turn_depth
                                                     , alpha=alpha, beta=beta)
            next_move_values = max(next_move_values, new_action_value)
            if next_move_values > beta:
                return next_move_values
            alpha = max(next_move_values, alpha)
        return next_move_values

    def get_value_for_min_agent(self, gameState, agent_index, turn_depth, alpha, beta):
        next_move_values = float("inf")
        last_agent_index = gameState.getNumAgents() - 1
        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)

            if agent_index == last_agent_index:
                new_action_value = self.get_action_value(new_game_state, agent_index=0,
                                                         turn_depth=turn_depth+1, alpha=alpha, beta=beta)
            else:
                new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1,
                                                         turn_depth=turn_depth, alpha=alpha, beta=beta)
            next_move_values = min(next_move_values, new_action_value)

            if next_move_values < alpha:
                return next_move_values

            beta = min(next_move_values, beta)
        return next_move_values



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
        #util.raiseNotDefined()
        agent_index = 0
        turn_depth = 0

        max_action = None
        max_value = float("-inf")

        legal_action_list = gameState.getLegalActions(0)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1, turn_depth=turn_depth)

            if max_action is None or max_value is None or max_value < new_action_value:
                max_action = legal_action
                max_value = new_action_value

        return max_action

    def get_action_value(self, gameState, agent_index, turn_depth):
        max_game_depth = self.depth
        is_win = gameState.isWin()
        is_lose = gameState.isLose()
        evaluation_func = self.evaluationFunction

        if is_win or is_lose or turn_depth >= max_game_depth:
            return evaluation_func(gameState)
        elif agent_index == 0:
            return self.get_value_for_max_agent(gameState, agent_index, turn_depth)
        else:
            return self.get_value_for_min_agent(gameState, agent_index, turn_depth)


    def get_value_for_max_agent(self, gameState, agent_index, turn_depth):
        next_move_values = float("-inf")

        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)
            new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1, turn_depth=turn_depth)
            next_move_values = max(next_move_values, new_action_value)

        return next_move_values

    def get_value_for_min_agent(self, gameState, agent_index, turn_depth):
        next_move_values = 0
        last_agent_index = gameState.getNumAgents() - 1
        legal_action_list = gameState.getLegalActions(agent_index)
        for legal_action in legal_action_list:
            new_game_state = gameState.generateSuccessor(agent_index, legal_action)

            if agent_index == last_agent_index:
                new_action_value = self.get_action_value(new_game_state, agent_index=0,
                                                         turn_depth=turn_depth + 1)
            else:
                new_action_value = self.get_action_value(new_game_state, agent_index=agent_index + 1,
                                                         turn_depth=turn_depth)
            next_move_values = next_move_values + new_action_value
        return next_move_values


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    score = scoreEvaluationFunction(currentGameState)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    newFoodList = newFood.asList()
    newFoodDistance = [manhattanDistance(f, newPos) for f in newFoodList]
    minFoodDist = min(newFoodDistance, default=0.001) if min(newFoodDistance, default=0.001) > 0 else 0.0001
    countFood = len(newFoodList) if len(newFoodList) > 0 else 0.001

    newGhostPositions = [g.getPosition() for g in newGhostStates]
    newGhostDistances = [manhattanDistance(g, newPos) for g in newGhostPositions]
    minGhostDist = min(newGhostDistances, default=0.001) if min(newGhostDistances, default=0.001) <= 4 else 4

    capsuleList = currentGameState.getCapsules()
    capsuledDistance = [manhattanDistance(f, newPos) for f in capsuleList]
    minCapsuledDistance = min(capsuledDistance, default=0.001) if min(capsuledDistance, default=0.001) > 0 else 1
    counCapsule = len(capsuleList) if len(capsuleList) > 0 else 0.001

    score = 1 / (countFood + counCapsule) + minGhostDist - 0.000001 * minFoodDist - 0.00000003 * minCapsuledDistance + score

    return score

# Abbreviation
better = betterEvaluationFunction
