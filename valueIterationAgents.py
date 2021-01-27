# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        # loop for the number of iterations we have
        for i in range(self.iterations):

            # save the values from the previous iteration before we start updating them
            oldIterationValues = self.values.copy()

            # loop through each state to update its value
            for state in self.mdp.getStates():

                # this list will hold the values for trying each action,
                # we will later take the max of this
                updatedValuesForState = []

                # loop through each action
                # for each action we need to calc the possible value of doing that action
                for action in self.mdp.getPossibleActions(state):

                    # for each action, there are many scenarios we could end up in, defined by the transition function
                    # sum each of these scenarios together defined by the value iteration equation
                    # transition = (nextState, probability)
                    tempSum = 0
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        probability = transition[1]
                        nextState = transition[0]
                        tempSum += probability * (self.mdp.getReward(state, action, nextState) + (self.discount*oldIterationValues[nextState]))
                    updatedValuesForState.append(tempSum)

                # If there is nothing in the list, I think this means it is a terminal state (no actions)
                if len(updatedValuesForState) == 0:
                    self.values[state] = 0
                else:
                    # take the max of all options
                    self.values[state] = max(updatedValuesForState)



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        QValue = 0
        # transition = (nextState, probability)
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            probability = transition[1]
            nextState = transition[0]
            QValue += self.mdp.getReward(state, action, nextState) + (self.discount * probability * self.values[nextState])

        return QValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # the policy will be the best action to take given all value options
        policy = None
        
        # if we are a terminal state, just return None
        if self.mdp.isTerminal(state):
            return None
        else:
            # initalize max value as negative infinity
            maxValue = float("-inf")

            # check each action and calculate value
            # update policy and maxValue if we find one better than we currently have
            for action in self.mdp.getPossibleActions(state):
                tempSum = 0
                for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                    probability = transition[1]
                    nextState = transition[0]
                    tempSum += probability * self.values[nextState]
                if tempSum > maxValue:
                    maxValue = tempSum
                    policy = action
            
            return policy


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # get number of states on grid, we will need this to calc the state we are updating on 
        # the current iteration
        numStates = len(self.mdp.getStates())

        # loop for all iterations, each time update next state
        for i in range(self.iterations):

            # save the values from the previous iteration before we start updating them
            oldIterationValues = self.values.copy()

            # get state for this iteration
            # we use the the iteration MOD numStates to get the current state
            state = self.mdp.getStates()[i % numStates]

            # if terminal pass, else calc value for state
            if self.mdp.isTerminal(state):
                pass
            else: 
                updatedValuesForState = []

                # loop through each action
                # for each action we need to calc the possible value of doing that action
                for action in self.mdp.getPossibleActions(state):

                    # for each action, there are many scenarios we could end up in, defined by the transition function
                    # sum each of these scenarios together defined by the value iteration equation
                    # transition = (nextState, probability)
                    tempSum = 0
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        probability = transition[1]
                        nextState = transition[0]
                        tempSum += probability * (self.mdp.getReward(state, action, nextState) + (self.discount*oldIterationValues[nextState]))
                    updatedValuesForState.append(tempSum)

                    # If there is nothing in the list, I think this means it is a terminal state (no actions)
                    if len(updatedValuesForState) == 0:
                        self.values[state] = 0
                    else:
                        # take the max of all options
                        self.values[state] = max(updatedValuesForState)
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

