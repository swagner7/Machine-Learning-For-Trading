import numpy as np

class RTLearner(object):
    """
    This is a Random Tree Regression Learner.
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "903756749"  # replace with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        """
        Build the random tree recursively
        """
        if len(np.unique(data_y)) == 1 or len(data_x) <= self.leaf_size:
            return np.mean(data_y)

        # Randomly select a feature to split on
        split_index = np.random.randint(data_x.shape[1])
        split_value = np.median(data_x[:, split_index])

        left_mask = data_x[:, split_index] <= split_value
        right_mask = np.logical_not(left_mask)

        if np.all(left_mask) or np.all(right_mask):
            return np.mean(data_y)

        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        return np.array([split_index, split_value, left_tree, right_tree])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        if self.tree is None:
            raise Exception("The model has not been trained yet!")

        predictions = np.empty(points.shape[0])

        for i, point in enumerate(points):
            predictions[i] = self.traverse_tree(point, self.tree)

        return predictions

    def traverse_tree(self, point, node):
        """
        Traverse the tree to predict the output for a single point
        """
        if isinstance(node, (float, np.float64)):  # Leaf node
            return node

        split_index, split_value, left_tree, right_tree = node

        if point[split_index] <= split_value:
            return self.traverse_tree(point, left_tree)
        else:
            return self.traverse_tree(point, right_tree)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
