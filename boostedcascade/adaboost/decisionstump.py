
# Author: Donny

import numpy as np

class DecisionStumpClassifier:
    """A decision stump classifier

    Parameters
    ----------
    steps_ : int, optional
        The steps to train on each feature.
    """
    def __init__(self, steps_=400):
        # self.features : int
        #   Number of features.
        # self.bestn : int
        #   Best feature.
        # self.bestd : int of -1 or 1
        #   Best direction.
        # self.bestp : float
        #   Best position.
        # self.steps : int
        #   Count of training iterations.
        self.steps = steps_
        pass

    def train(self, X_, y_, W_):
        """Train the decision stump with the training set {X, y}

        Parameters
        ----------
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the training samples.
        y_ : np.array of shape = [n_samples]
            The class labels of the training samples.
            Currently only supports class -1 and 1.
        W_ : np.array of shape = [n_samples]
            The weights of each samples.

        Returns
        -------
        err : float
            The sum of weighted errors.
        """

        X = np.array(X_)
        y = np.array(y_)
        W = np.array(W_)
        steps = self.steps

        n_samples, n_features = X.shape
        assert n_samples == y.size

        bestn = 0
        bestd = 1
        bestp = 0
        minerr = W.sum()
        for n in range(n_features):
            err, d, p = self._optimize(X[:, n], y, W, steps)
            if err < minerr:
                minerr = err
                bestn = n
                bestd = d
                bestp = p
        
        self.features = n_features
        self.bestn = bestn
        self.bestd = bestd
        self.bestp = bestp

        return minerr

    def _optimize(self, X, y, W, steps):
        """Get optimal direction and position to divided X.

        Parameters
        ----------
        X : np.array of shape = [n_samples]
            The inputs of a certain feature of the training samples.
        y : np.array of shape = [n_samples]
            The class labels of the training samples.
        W : np.array of shape = [n_samples]
            The weights of each samples.
        steps : int
            Count of training iterations.

        Returns
        -------
        err : float
            The sum of weighted errors.
        d : int of value -1 or 1
            The optimal direction.
        p : float
            The optimal position.
        """

        X = X.flatten(1)

        min_x, max_x = X.min(), X.max()
        len_x = max_x - min_x
        
        bestd = 1
        bestp = min_x
        minerr = W.sum()

        if len_x > 0.0:
            for p in np.arange(min_x, max_x, len_x/steps):
                for d in [-1, 1]:
                    gy = np.ones((y.size))
                    gy[X*d < p*d] = -1
                    err = np.sum((gy != y)*W)
                    if err < minerr:
                        minerr = err
                        bestd = d
                        bestp = p

        return minerr, bestd, bestp

    def predict(self, test_set_):
        """Predict the classes of input samples

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predict result of the testing samples.
        """

        test_set = np.array(test_set_)
        n_samples, n_features = test_set.shape

        assert n_features == self.features

        single_feature = test_set[:, self.bestn]
        h = np.ones((n_samples))
        h[single_feature*self.bestd < self.bestp*self.bestd] = -1
        return h
