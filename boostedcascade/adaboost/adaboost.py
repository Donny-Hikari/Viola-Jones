
# Author: Donny

import numpy as np
import copy

from .decisionstump import DecisionStumpClassifier

class AdaBoostClassifier:
    """An AdaBoost classifier.

    Parameters
    ----------
    n_classes_ : int
        count of maximal weak classifiers
    weak_classifier_ : A weak classifier class or a factory which 
                       can return this kind of class.
        A weak classifier class, with:
            function train(self, X, y, W), where:
                    param X : [array] accepts inputs of the training samples.
                    param y : [array] accepts class labels of the training samples.
                    param W : [array] accepts weights of each samples
                    return  : [float] the sum of weighted errors
            function predict(self, test_set), where:
                    param test_set : accepts test samples
                    return         : classify result
    """
    def __init__(self,
                 weak_classifier_ = DecisionStumpClassifier()):
        # Maximal weak classifiers
        # self.mxWC = n_classes_
        self.mxWC = 0
        # Class of weak classifier
        self.WCClass = weak_classifier_
        # self.WCs : [self.WCClass list]
        #   List of Weak classifiers.
        # self.nWC : [int]
        #   Number of weak classifiers used. (<= mxWC)
        self.nWC = 0
        # self.alpha : [float list]
        #   Alpha contains weights of each weak classifier,
        #   ie. votes they have.
        # self.features : [int]
        #   Number of features.
        # self.sum_eval : [float]
        #   Sum of votes result of all evaluated
        #   weak classifiers.

    def train(self, X_, y_, n_classes_, is_continue=False, verbose=False):
        """Train the AdaBoost classifier with the training set (X, y).

        Parameters
        ----------
        X_ : array-like of shape = [n_samples, n_features]
            The inputs of the training samples.
        y_ : array-like of shape = [n_samples]
            The class labels of the training samples.
            Currently only supports class 0 and 1.
        """

        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = np.array(y_).flatten(1)
        y[y == 0] = -1
        n_samples, n_features = X.shape

        assert n_samples == y.size
        
        if not is_continue or self.nWC == 0:
            # Initialize weak classifiers
            self.mxWC = n_classes_
            self.WCs = [copy.deepcopy(self.WCClass) for a in range(self.mxWC)]
            self.nWC = 0
            self.alpha = np.zeros((self.mxWC))
            self.features = n_features
            self.sum_eval = 0

            # Initialize weights of inputs samples
            W = np.ones((n_samples)) / n_samples
        else:
            self.mxWC = self.nWC + n_classes_
            self.WCs = np.concatenate((self.WCs[0:self.nWC], [copy.deepcopy(self.WCClass) for a in range(n_classes_)]))
            self.alpha = np.concatenate((self.alpha[0:self.nWC], np.zeros((n_classes_))))
            W = self.W

        for a in range(self.nWC, self.mxWC):
            if verbose: print('Training %d-th weak classifier' % a)
            err = self.WCs[a].train(X, y, W, verbose)
            
            if err == 0: err = 1e-6
            elif err == 1: err = 1 - 1e-6

            h = self.WCs[a].predict(X).flatten(1)
            self.alpha[a] = 0.5 * np.log((1 - err) / err)
            W = W * np.exp(-self.alpha[a]*y*h)
            W = W / W.sum()
            self.nWC = a+1
            if verbose: print('%d-th weak classifier: err = %f' % (a, err))
            if self._evaluate(a, h, y) == 0:
                print(self.nWC, "weak classifiers are enought to make error rate reach 0.0")
                break

        self.W = W

    def _evaluate(self, t, h, y):
        """Evaluate current model.

        Parameters
        ----------
        t : int
            Index of the last weak classifier.
        h : np.array of shape = [n_samples]
            The predict result by the t-th weak classifier.
        y : np.array of shape = [n_samples]
            The class labels of the training samples.
        
        Returns
        -------
        cnt : int
            Count of mis-classified samples.
        """

        self.sum_eval = self.sum_eval + h*self.alpha[t]
        yPred = np.sign(self.sum_eval)
        return np.sum(yPred != y)

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
        CI : np.array of shape = [n_samples]
            The confidence of the predict results.
        """

        hsum = self.weightedSum(test_set_)
        CI = abs(hsum) / np.sum(abs(self.alpha))

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0

        return yPred, CI

    def weightedSum(self, test_set_):
        """Return the weighted sum of all weak classifiers

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        hSum : np.array of shape = [n_samples]
            The predict result of the testing samples.
        """

        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)

        assert test_set.shape[1] == self.features

        hsum = 0
        for i in range(self.nWC):
            hsum = hsum + self.alpha[i] * self.WCs[i].predict(test_set)

        return hsum
