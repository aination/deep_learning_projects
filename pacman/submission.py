from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current GameState (defined in pacman.py)
    and a proposed action and returns a rough estimate of the resulting successor
    GameState's value.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Don't forget to limit the search depth using self.depth. Also, avoid modifying
      self.depth directly (e.g., when implementing depth-limited search) since it
      is a member variable that should stay fixed throughout runtime.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    self.nv = 0
    def recursion_func(state,agent,d):
        ren = []
        self.nv += 1
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state),Directions.NORTH)
        if agent == 0: # if Pacman
            for action in state.getLegalActions(agent):
                if (action == Directions.STOP or action == None):
                    continue
                ns = state.generateSuccessor(agent,action)
                va = (recursion_func(ns,agent+1,d)[0],action)
                ren.append(va)
            return max(ren)
        for action in state.getLegalActions(agent):
            if (action == Directions.STOP or action == None):
                continue
            ns = state.generateSuccessor(agent,action)
            if agent == gameState.getNumAgents() - 1:
                va = (recursion_func(ns,agent+1,d-1)[0],action)
            else:
                va = (recursion_func(ns,agent+1,d)[0],action)
            ren.append(va)
        return min(ren)
    result = recursion_func(gameState,self.index,self.depth)
    return result[1]
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    self.nv = 0
    def recusion_func(alpha,beta,state,agent,d):
        res = []
        self.nv += 1
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state), Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state), Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), Directions.NORTH)
        if agent == 0: # if Pacman
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                ns = state.generateSuccessor(agent,action)
                v = recusion_func(alpha,beta,ns,agent+1,d)
                if v != None:
                    va = (v[0],action)
                    node_v = v[0]
                    if node_v >= beta:
                        return
                    else:
                        alpha = max(node_v,alpha)
                    res.append(va)
            if len(res) > 0:
                return max(res)
            return
        else: # if ghost
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                ns = state.generateSuccessor(agent,action)
                if agent == gameState.getNumAgents() - 1: # if last ghost
                    v = recusion_func(alpha,beta,ns,agent+1,d-1)
                else:
                    v = recusion_func(alpha,beta,ns,agent+1,d)
                if v != None:
                    va = (v[0],action)
                    node_v = v[0]
                    if node_v <= alpha:
                        return
                    else:
                        beta = min(node_v,beta)
                    res.append(va)
            if len(res) > 0:
                return min(res)
            return
    result = recusion_func(-999999999999999,999999999999999,gameState,self.index,self.depth)
    return result[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def recusion_func(state,agent,d):
        ret = []
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state),Directions.NORTH)
        if agent == 0: # if Pacman
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                ns = state.generateSuccessor(agent,action)
                va = (recusion_func(ns,agent+1,d)[0],action)
                ret.append(va)
            return max(ret)
        if agent > 0:
            result = (0,Directions.STOP)
            for action in state.getLegalActions(agent): # if ghost
                pa = 1.0 / len(state.getLegalActions(agent))
                if action == None or action == Directions.STOP:
                    continue
                ns = state.generateSuccessor(agent,action)
                if agent == gameState.getNumAgents() - 1: # if last ghost
                    result = (result[0]+pa*(recusion_func(ns,agent+1,d-1)[0]),action)
                else:
                    result = (result[0]+pa*(recusion_func(ns,agent+1,d)[0]),action)
            return result
    result = recusion_func(gameState,self.index,self.depth)
    return result[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """
  # START_YOUR_CODE
  ncg = 0
  ngp = 0
  score = 4.5*currentGameState.getScore()
  if currentGameState.isWin():
      return 9999999999999
  if currentGameState.isLose():
      return -9999999999999
  loc = currentGameState.getPacmanPosition()
  ghost_pos = list(currentGameState.getGhostPositions())
  fd = currentGameState.getFood().asList()
  closestfood = 9999999999999
  closestcapsule = 9999999999999
  for i in range(0,len(fd)):
      closestfood = min(util.manhattanDistance(fd[i],loc),closestfood)
  for i in range(0,len(currentGameState.getCapsules())):
      closestcapsule = min(util.manhattanDistance(currentGameState.getCapsules()[i],loc),closestcapsule)
  i = 1
  ghcs = 9999999999999
  sg = 9999999999999

  while i <= currentGameState.getNumAgents() - 1:
      ghostPos = currentGameState.getGhostPosition(i)
      if ghostPos[1] >= 5 and ghostPos[0] >= 7 and ghostPos[0] <= 12:
          ngp = ngp + 1
      ghost_state = currentGameState.getGhostState(i)
      if ghost_state.scaredTimer > 0:
          next_scared_ghost = util.manhattanDistance(loc,ghostPos)
          sg = min(sg,next_scared_ghost)
          ncg = ncg + 1
      else:
          ghost_dist = util.manhattanDistance(loc,ghostPos)
          ghcs = min(ghcs,ghost_dist)
      i = i + 1
  if  loc[0] >= 7 and loc[1] >= 5 and loc[0] <= 12:
      score -= 99999999
  score -= 3.7 * 1/float(ghcs)
  score -= 1.9 * closestfood
  score += 1400 * 1/float(sg)
  score += 2 * ngp
  score -= 3.5 * len(fd)
  score -= 105 * len(currentGameState.getCapsules())
  return score
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction