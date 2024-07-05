import numpy as np

class DTLearner(object):
    """
    This is a Decision Tree Regression Learner.
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
        Build the decision tree recursively
        """
        if len(np.unique(data_y)) == 1 or len(data_x) <= self.leaf_size:
            return np.mean(data_y)

        # Find the best split
        best_split_index, best_split_value = self.find_best_split(data_x, data_y)

        if best_split_index is None:
            return np.mean(data_y)

        left_mask = data_x[:, best_split_index] <= best_split_value
        right_mask = np.logical_not(left_mask)

        if np.all(left_mask) or np.all(right_mask):
            return np.mean(data_y)

        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        return [best_split_index, best_split_value, left_tree, right_tree]

    def find_best_split(self, data_x, data_y):
        """
        Find the best feature and value to split on based on correlation coefficient
        """
        best_correlation = -1  # Initialize with a negative value
        best_split_index = None
        best_split_value = None

        for feature_index in range(data_x.shape[1]):
            correlation = np.abs(np.corrcoef(data_x[:, feature_index], data_y)[0, 1])

            if correlation > best_correlation:
                best_correlation = correlation
                best_split_index = feature_index

                # Choose median value as split value
                best_split_value = np.median(data_x[:, feature_index])

        return best_split_index, best_split_value

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

        predictions = []

        for point in points:
            predictions.append(self.traverse_tree(point, self.tree))

        return np.array(predictions)

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
