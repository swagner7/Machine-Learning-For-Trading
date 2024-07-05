import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.bag_learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20, verbose=verbose) for _ in range(20)]
    def author(self):
        return "903756749"
    def add_evidence(self, data_x, data_y):
        for learner in self.bag_learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        predictions = np.zeros((points.shape[0], len(self.bag_learners)))
        for i, learner in enumerate(self.bag_learners):
            predictions[:, i] = learner.query(points)
        return np.mean(predictions, axis=1)
