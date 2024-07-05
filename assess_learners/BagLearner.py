import numpy as np

class BagLearner(object):
    """
    BagLearner class implementing Bootstrap Aggregation.
    """

    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        """
        Constructor method for BagLearner.
        :param learner: The learner class to be used for creating an ensemble.
        :param kwargs: Keyword arguments passed to the learner's constructor.
        :param bags: Number of learners to be trained using Bootstrap Aggregation.
        :param boost: Boolean indicating whether boosting should be applied (optional, not implemented).
        :param verbose: Boolean indicating whether to generate output.
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

    def author(self):
        """
        Returns the GT username of the student.
        """
        return "903756749"  # replace with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to the bagged learners.
        :param data_x: A set of feature values used to train the learners.
        :param data_y: The value we are attempting to predict given the X data.
        """
        for _ in range(self.bags):
            learner_instance = self.learner(**self.kwargs)
            indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            learner_instance.add_evidence(data_x[indices], data_y[indices])
            self.learners.append(learner_instance)

    def query(self, points):
        """
        Estimate a set of test points using the ensemble of learners.
        :param points: A numpy array with each row corresponding to a specific query.
        :return: The predicted result of the input data according to the trained model.
        """
        if not self.learners:
            raise Exception("No learners have been trained yet!")

        predictions = np.zeros((points.shape[0], len(self.learners)))
        for i, learner in enumerate(self.learners):
            predictions[:, i] = learner.query(points)

        return np.mean(predictions, axis=1)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
