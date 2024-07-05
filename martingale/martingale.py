""""""  		  	   		 	   			  		 			     			  	 
"""Assess a betting strategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
Student Name: Samuel Wagner 		  	   		 	   			  		 			     			  	 
GT User ID: swagner38 		  	   		 	   			  		 			     			  	 
GT ID: 903756749		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def author():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return "swagner38"  # replace tb34 with your Georgia Tech username.
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def gtid():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	   			  		 			     			  	 
    :rtype: int  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return 903756749  # replace with your GT ID number
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	   			  		 			     			  	 
    :type win_prob: float  		  	   		 	   			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	   			  		 			     			  	 
    :rtype: bool  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    result = False  		  	   		 	   			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		 	   			  		 			     			  	 
        result = True

    return result  		  	   		 	   			  		 			     			  	 


def run_sim_no_bankroll(episodes, win_prob):

    episode_winnings = np.zeros([episodes, 1001]) # initialize array

    for episode_counter in range(episodes): # for each episode...
        spin_counter = 1 # initialize spin counter

        while episode_winnings[episode_counter][spin_counter - 1] < 80 and spin_counter < 1000: # establish cutoff criteria
            won = False # initialize win/lose Boolean
            bet_amount = 1 # initizlie bet amount

            while not won:
                won = get_spin_result(win_prob) # get the randomized spin

                if won == True: # if spin is sucessful
                    episode_winnings[episode_counter][spin_counter] = episode_winnings[episode_counter][spin_counter - 1] + bet_amount # add winnings and append value
                else:
                    episode_winnings[episode_counter][spin_counter] = episode_winnings[episode_counter][spin_counter - 1] - bet_amount # subtract bet and append value
                    bet_amount = bet_amount * 2 # double bet amount

                spin_counter += 1 # increment spin counter

        episode_winnings[episode_counter][spin_counter:1001] = 80 # fill forward

    return(episode_winnings)

def run_sim_bankroll(episodes, win_prob):

    episode_winnings = np.zeros([episodes, 1001]) # initialize array

    for episode_counter in range(episodes): # for each episode...
        spin_counter = 1 # initialize spin counter

        while episode_winnings[episode_counter][spin_counter - 1] < 80 and spin_counter < 1000 and episode_winnings[episode_counter][spin_counter - 1] > -256: # establish cutoff criteria
            won = False # initialize win/lose Boolean
            bet_amount = 1 # initizlie bet amount

            while not won:
                won = get_spin_result(win_prob) # get the randomized spin

                if won == True: # if spin is sucessful
                    episode_winnings[episode_counter][spin_counter] = episode_winnings[episode_counter][spin_counter - 1] + bet_amount # add winnings and append value
                else:
                    episode_winnings[episode_counter][spin_counter] = episode_winnings[episode_counter][spin_counter - 1] - bet_amount # subtract bet and append value

                    if (256 + episode_winnings[episode_counter][spin_counter]) >= (bet_amount * 2): # check if there is enough left in the bankroll to double bet amount
                        bet_amount = bet_amount * 2 # if so, double bet
                    else:
                        bet_amount = 256 + episode_winnings[episode_counter][spin_counter] # if not, bet what is left

                spin_counter += 1 # increment spin counter


        if episode_winnings[episode_counter][spin_counter-1] >= 80:
            episode_winnings[episode_counter][spin_counter:1001] = 80 # fill forward
        elif episode_winnings[episode_counter][spin_counter-1] <= -256:
            episode_winnings[episode_counter][spin_counter:1001] = -256  # fill forward

    return(episode_winnings)


def test_code():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Method to test your code  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    win_prob = 0.474  # probability of hitting black based on Roulette wiki (converted from odds)
    np.random.seed(gtid())  # do this only once

    # Experiment 1 ------------
    # Figure 1
    winnings = run_sim_no_bankroll(episodes=10, win_prob=win_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(winnings.transpose())
    plt.axis([0, 300, -256, 100])
    plt.title('Monte Carlo Martingale Roulette Strategy Simulator (10 Episodes)')
    plt.xlabel('Spin #')
    plt.ylabel('Winnings ($)')
    plt.savefig('Ex1_Fig1.png')

    # Figure 2
    winnings = run_sim_no_bankroll(episodes=1000, win_prob=win_prob)
    spin_avg = np.mean(winnings, axis = 0)
    std_dev = np.std(winnings, axis = 0)
    spin_ubound = spin_avg+std_dev
    spin_lbound = spin_avg-std_dev

    plt.figure(figsize=(10, 6))
    plt.plot(spin_avg.transpose())
    plt.plot(spin_ubound.transpose())
    plt.plot(spin_lbound.transpose())
    plt.axis([0, 300, -256, 100])
    plt.title('Monte Carlo Martingale Roulette Strategy Simulator Average (1000 Episodes)')
    plt.legend(['Mean', 'Mean + sigma', 'Mean - signma'])
    plt.xlabel('Spin #')
    plt.ylabel('Avg Winnings ($)')
    plt.savefig('Ex1_Fig2.png')

    # Figure 3
    spin_med = np.median(winnings, axis = 0)
    spin_ubound = spin_med+std_dev
    spin_lbound = spin_med-std_dev

    plt.figure(figsize=(10, 6))
    plt.plot(spin_med.transpose())
    plt.plot(spin_ubound.transpose())
    plt.plot(spin_lbound.transpose())
    plt.axis([0, 300, -256, 100])
    plt.title('Monte Carlo Martingale Roulette Strategy Simulator Median (1000 Episodes)')
    plt.xlabel('Spin #')
    plt.ylabel('Winnings ($)')
    plt.legend(['Median', 'Median + sigma', 'Median - signma'])
    plt.savefig('Ex1_Fig3.png')

    win_counter = 0
    for i in range(winnings.shape[0]):
        if winnings[i][-1] == 80:
            win_counter += 1

    print('Prob of $80:', 100*win_counter/winnings.shape[0])

    # Experiment 2 ----------------
    # Figure 4
    winnings = run_sim_bankroll(episodes=1000, win_prob=win_prob)
    spin_avg = np.mean(winnings, axis=0)
    std_dev = np.std(winnings, axis = 0)
    spin_ubound = spin_avg+std_dev
    spin_lbound = spin_avg-std_dev

    plt.figure(figsize=(10, 6))
    plt.plot(spin_avg.transpose())
    plt.plot(spin_ubound.transpose())
    plt.plot(spin_lbound.transpose())
    plt.axis([0, 300, -256, 100])
    plt.title('Monte Carlo Martingale Roulette Strategy (w/ Bankroll) Simulator Mean (1000 Episodes)')
    plt.xlabel('Spin #')
    plt.ylabel('Winnings ($)')
    plt.legend(['Mean', 'Mean + sigma', 'Mean - signma'])
    plt.savefig('Ex2_Fig4.png')

    win_counter = 0
    for i in range(winnings.shape[0]):
        if winnings[i][-1] == 80:
            win_counter += 1

    print('Prob of win:', 100*win_counter/winnings.shape[0])


    # Figure 5
    spin_med = np.median(winnings, axis = 0)
    spin_ubound = spin_med+std_dev
    spin_lbound = spin_med-std_dev

    plt.figure(figsize=(10, 6))
    plt.plot(spin_med.transpose())
    plt.plot(spin_ubound.transpose())
    plt.plot(spin_lbound.transpose())
    plt.axis([0, 300, -256, 100])
    plt.title('Monte Carlo Martingale Roulette Strategy (w/ Bankroll) Simulator Median (1000 Episodes)')
    plt.xlabel('Spin #')
    plt.ylabel('Winnings ($)')
    plt.legend(['Median', 'Median + sigma', 'Median - signma'])
    plt.savefig('Ex2_Fig5.png')

  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    test_code()  		  	   		 	   			  		 			     			  	 
