""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import math  		  	   		 	   			  		 			     			  	 
import sys  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bgl
import InsaneLearner as inl
import matplotlib.pyplot as plt
  		  	   		 	   			  		 			     			  	 

def get_data(file):
    """
    :param file: testing dataset
    :return: X and Y in same np array
    """
    with open(file) as f:
        alldata = np.genfromtxt(f, delimiter=",")
        # Cleaning
        alldata = alldata[1:, 1:]  # drops row/date column and headers

        # Spliting datasets to match add_evidence requirement
        num_cols = alldata.shape[1]
        X = alldata[:, 0:num_cols - 1]
        Y = alldata[:, -1]
    return alldata


if __name__ == "__main__":

    data = get_data(sys.argv[1])

    # compute how much of the data is training and testing  		  	   		 	   			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # separate out training and testing data  		  	   		 	   			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		 	   			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		 	   			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		 	   			  		 			     			  	 
    test_y = data[train_rows:, -1]


    # Experiment 1 ----------------------------------------------------------

    max_leaf_size = 25
    in_sample_rmse_vect = []
    out_sample_rmse_vect = []

    for i in range(1, max_leaf_size):
        learner = dtl.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)

        in_sample_pred_y = learner.query(train_x)
        in_sample_rmse = math.sqrt(((train_y - in_sample_pred_y) ** 2).sum() / train_y.shape[0])
        in_sample_rmse_vect.append(in_sample_rmse)

        out_sample_pred_y = learner.query(test_x)
        out_sample_rmse = math.sqrt(((test_y - out_sample_pred_y) ** 2).sum() / test_y.shape[0])
        out_sample_rmse_vect.append(out_sample_rmse)

    plt.figure(1)
    plt.plot(range(1, max_leaf_size), in_sample_rmse_vect, label = 'In-Sample', color = 'red')
    plt.plot(range(1, max_leaf_size), out_sample_rmse_vect, label = 'Out-of-Sample', color = 'blue')
    plt.xlabel('Leaf Size')
    plt.ylabel('Error (RMSE)')
    plt.title('Overfitting Assessment by Analysis of RMSE')
    plt.legend()
    plt.savefig("Ex1")


    # Experiment 2 ----------------------------------------------------------

    bags = 50
    in_sample_bag_rmse_vect = []
    out_sample_bag_rmse_vect = []

    for i in range(1, max_leaf_size):
        learner = bgl.BagLearner(learner=dtl.DTLearner, kwargs={'leaf_size': i}, bags=bags, verbose=False)
        learner.add_evidence(train_x, train_y)

        in_sample_pred_y = learner.query(train_x)
        in_sample_rmse = math.sqrt(((train_y - in_sample_pred_y) ** 2).sum() / train_y.shape[0])
        in_sample_bag_rmse_vect.append(in_sample_rmse)

        out_sample_pred_y = learner.query(test_x)
        out_sample_rmse = math.sqrt(((test_y - out_sample_pred_y) ** 2).sum() / test_y.shape[0])
        out_sample_bag_rmse_vect.append(out_sample_rmse)

    plt.figure(2)
    plt.plot(range(1, max_leaf_size), in_sample_rmse_vect, label = 'In-Sample', color = 'red', alpha = 1)
    plt.plot(range(1, max_leaf_size), out_sample_rmse_vect, label = 'Out-of-Sample', color = 'blue', alpha = 1)
    plt.plot(range(1, max_leaf_size), in_sample_bag_rmse_vect, label = 'In-Sample (w/ bagging)', color = 'red', alpha = 0.25)
    plt.plot(range(1, max_leaf_size), out_sample_bag_rmse_vect, label = 'Out-of-Sample (w/ bagging)', color = 'blue', alpha = 0.25)
    plt.xlabel('Leaf Size')
    plt.ylabel('Error (RMSE)')
    plt.title('Effect of Bagging on Overfitting by Analysis of RMSE')
    plt.legend()
    plt.savefig("Ex2")


    # Experiment 3 ----------------------------------------------------------

    import time

    max_leaf_size = 50
    dt_mae_vect = []
    dt_time_vect = []
    rt_mae_vect = []
    rt_time_vect = []

    for i in range(1, max_leaf_size):
        learner = dtl.DTLearner(leaf_size=i, verbose=False)
        t0 = time.time()
        learner.add_evidence(train_x, train_y)
        t1 = time.time()
        delta = t1-t0
        dt_time_vect.append(delta)

        pred_y = learner.query(test_x)
        dt_mae = float(abs(test_y - pred_y).sum() / test_y.shape[0])
        dt_mae_vect.append(dt_mae)

        learner = rtl.RTLearner(leaf_size=i, verbose=False)
        t0 = time.time()
        learner.add_evidence(train_x, train_y)
        t1 = time.time()
        delta = t1-t0
        rt_time_vect.append(delta)

        pred_y = learner.query(test_x)
        rt_mae = float(abs(test_y - pred_y).sum() / test_y.shape[0])
        rt_mae_vect.append(rt_mae)

    plt.figure(3).set_figwidth(8)
    plt.plot(range(1, max_leaf_size), dt_mae_vect, label = 'Classic Decision Tree', color = 'red')
    plt.plot(range(1, max_leaf_size), rt_mae_vect, label = 'Random Tree', color = 'blue')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Effect of Leaf Size on Error (MAE) by Algorithm')
    plt.legend()
    plt.savefig("Ex3_1")

    plt.figure(4).set_figwidth(8)
    plt.plot(range(1, max_leaf_size), dt_time_vect, label = 'Classic Decision Tree', color = 'red')
    plt.plot(range(1, max_leaf_size), rt_time_vect, label = 'Random Tree', color = 'blue')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time')
    plt.title('Effect of Leaf Size on Time by Algorithm')
    plt.legend()
    plt.savefig("Ex3_2")

