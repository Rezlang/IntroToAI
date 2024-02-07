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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Sum distances to ghosts - numFood
        score = successorGameState.getScore()
        foods = newFood.asList()
        score += sum(2 * (1 / manhattanDistance(newPos, foodPos)) for foodPos in foods)

        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghost) <= 3.5:
                score -= 40
            elif manhattanDistance(newPos, ghost) <= 5:
                score -= 20
            elif manhattanDistance(newPos, ghost) <= 10:
                score -= 10
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
        def minValue(state, agentIndex, depth):
            # ghosts
            legalActions = state.getLegalActions(agentIndex)
            # if there are no legal actions or the game is over, evaluate the state
            if not legalActions or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # move to next agent
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                # if its pacman, go to the next depth
                return min(maxValue(state.generateSuccessor(agentIndex, action), 0, depth + 1) for action in legalActions)
            else:
                # otherwise recursively call next agent
                return min(minValue(state.generateSuccessor(agentIndex, action), nextAgent, depth) for action in legalActions)

        def maxValue(state, agentIndex, depth):
            # pacman
            legalActions = state.getLegalActions(agentIndex)
            # if there are no legal actions or the game is over, evaluate the state
            if not legalActions or depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # go to the ghosts
            return max(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legalActions)

        # get pacman's initial possible actions
        actions = gameState.getLegalActions(0)
        # if there are no actions, return
        if not actions: 
            return None

        # check the heuristic value of successor ghost states and store the min value with that state
        actionValues = [(action, minValue(gameState.generateSuccessor(0, action), 1, 1)) for action in actions]
        # choose the action with the maximum evaluated value
        bestAction = max(actionValues, key=lambda x: x[1])[0]
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # ghost code
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():   #check if game is over
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1):
                    minvalue = min (minvalue,maxLevel(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue
        
        # pacman code
        def maxLevel(gameState,depth,alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:   #check if game is over
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # call min next for ghosts
            score = minLevel(nextState,0,1,alpha,beta)
            # choose the best action
            if score > currentScore:
                currentScore = score
                returnAction = action
            # update the values
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # pacman code
        def maxLevel(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            totalmaxvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,expectLevel(successor,currDepth,1))
            return maxvalue
        
        # ghost code
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   # check if game is over
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return  0
            return float(totalexpectedvalue)/float(numberofactions)
        
        #begin algorithm by getting next possible pacman actions
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a expect level. Hence calling expectLevel for successors of the root.
            score = expectLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # get distances of food
    foodList = newFood.asList()
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    # get distances of ghosts
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    # final score calculations
    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    # food reciprocal calculations
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
    
    # food and game score accounted for in final heuristic score
    score += currentGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    # ghost score is calculated differently if the ghosts are scared or not
    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score

# Abbreviation
better = betterEvaluationFunction
