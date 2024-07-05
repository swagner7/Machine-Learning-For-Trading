""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Samuel Wagner (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: swagner38 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903756749 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import random as rand  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class QLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This is a Q learner object.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		 	   			  		 			     			  	 
    :type num_states: int  		  	   		 	   			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		 	   			  		 			     			  	 
    :type num_actions: int  		  	   		 	   			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type alpha: float  		  	   		 	   			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type gamma: float  		  	   		 	   			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type rar: float  		  	   		 	   			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	   			  		 			     			  	 
    :type radr: float  		  	   		 	   			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	   			  		 			     			  	 
    :type dyna: int  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """


    def __init__(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        num_states=100,  		  	   		 	   			  		 			     			  	 
        num_actions=4,  		  	   		 	   			  		 			     			  	 
        alpha=0.2,  		  	   		 	   			  		 			     			  	 
        gamma=0.9,  		  	   		 	   			  		 			     			  	 
        rar=0.5,  		  	   		 	   			  		 			     			  	 
        radr=0.99,  		  	   		 	   			  		 			     			  	 
        dyna=0,  		  	   		 	   			  		 			     			  	 
        verbose=False,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros((num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states)) # Transition matrix for Dyna-Q
        self.R = np.zeros((num_states, num_actions)) # Reward matrix for Dyna-Q
        self.experiences = [] # List to store experiences for Dyna-Q

  		  	   		 	   			  		 			     			  	 
    def querysetstate(self, s):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s: The new state  		  	   		 	   			  		 			     			  	 
        :type s: int  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action
  		  	   		 	   			  		 			     			  	 
    def query(self, s_prime, r):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Update the Q table and return an action  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s_prime: The new state  		  	   		 	   			  		 			     			  	 
        :type s_prime: int  		  	   		 	   			  		 			     			  	 
        :param r: The immediate reward  		  	   		 	   			  		 			     			  	 
        :type r: float  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]))
        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime, :])
        self.rar *= self.radr

        if self.dyna > 0:
            self.experiences.append((self.s, self.a, s_prime, r))
            self.dyna_update()

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r = {r}")

        self.s = s_prime
        self.a = action
        return action

    def dyna_update(self):
        """
        Dyna-Q update method to simulate experiences
        """
        # Convert experiences to numpy array for efficient indexing
        experiences_arr = np.array(self.experiences)

        # Update Q-table using real experiences
        s, a, s_prime, r = experiences_arr[:, 0], experiences_arr[:, 1], experiences_arr[:, 2], experiences_arr[:, 3]
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                       self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :], axis=1))

        # Update model (T and R)
        self.T.fill(0)  # Reset transition matrix
        self.R.fill(0)  # Reset reward matrix
        np.add.at(self.T, (s, a, s_prime), 1)  # Update transition counts
        np.add.at(self.R, (s, a), r)  # Update reward counts

        # Generate simulated experiences in batch
        sim_states = np.random.randint(0, self.num_states, size=(self.dyna,))
        sim_actions = np.random.randint(0, self.num_actions, size=(self.dyna,))

        # Get next states and rewards using T and R matrices
        sim_next_states = np.argmax(self.T[sim_states, sim_actions, :], axis=1)
        sim_rewards = self.R[sim_states, sim_actions]

        # Update Q-table using simulated experiences
        self.Q[sim_states, sim_actions] = \
            (1 - self.alpha) * self.Q[sim_states, sim_actions] + \
            self.alpha * (sim_rewards + self.gamma * np.max(self.Q[sim_next_states, :], axis=1))

    def author(self):
        return 'swagner38'


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		 	   			  		 			     			  	 
