# 
# boostedcascade.py
#   Boosted cascade proposed by Viola & Jones
# 
# Author : Donny
# 

import numpy as np
from sklearn.utils import shuffle as skshuffle
import copy

from .adaboost import AdaBoostClassifier
from .adaboost import DecisionStumpClassifier
from . import HaarlikeFeature

class BoostedCascade:
    """Boosted cascade proposed by Viola & Jones

    Parameters
    ----------
    Ftarget : float
        The target maximum false positvie rate.
    f : float
        The maximum acceptable false positive rate per layer.
    d : float
        The minimum acceptable detection rate per layer.
    adaboostClassifier : AdaBoostClassifier
        A AdaBoost classifier instance.
    validset_rate : float, optional
        The ratio of valid set in the whole training set.
    CIsteps : float, optional.
        The steps of decreasing confidence threshold in each cascaded classifier.
    """

    Haarlike = HaarlikeFeature()

    def __init__(self,
                 Ftarget, f, d,
                 adaboostClassifier = AdaBoostClassifier(
                     200,
                     weak_classifier_ = DecisionStumpClassifier()
                 ),
                 validset_rate = 0.3,
                 CIsteps = 0.1):
        # Target false positive rate.
        self.Ftarget = Ftarget
        # The maximum acceptable false positive rate per layer.
        self.f = f
        # The minimum acceptable detection rate per layer
        self.d = d
        # Class of strong classifier, usu. AdaBoostClassifier
        self.SCClass = adaboostClassifier
        # The ratio of valid set in the whole training set.
        self.validset_rate = validset_rate
        # The steps of decreasing confidence threshold in each cascaded classifier.        
        self.CIsteps = CIsteps

        # The window size of detector
        self.detecWndW, self.detectWndH = (-1, -1)

        self.P = self.N = -1
        self.features_cnt = -1
        self.features_descriptions = -1

        # self.SCs : list of self.SCClass
        #   The strong classifiers.
        # self.thresholds : list of float
        #   The thresholds of each strong classifier.
        # self.SCn : list of int
        #   The number of features used.

    def savefeaturesdata(self, filename):
        numpy.save(filename+'-P.bcf', self.P)
        numpy.save(filename+'-N.bcf', self.N)
        numpy.save(filename+'-features_cnt.bcf', self.features_cnt)
        numpy.save(filename+'-features_descriptions.bcf', self.features_descriptions)

    def loadfeaturesdata(self, filename):
        self.P = numpy.load(filename+'-P.bcf')
        self.N = numpy.load(filename+'-N.bcf')
        self.features_cnt = numpy.load(filename+'-features_cnt.bcf')
        self.features_descriptions = numpy.load(filename+'-features_descriptions.bcf')

    def prepare(self, P_, N_, shuffle=True, verbose=False):
        """Prepare the data for training.

        Parameters
        ----------
        P_ : array-like of shape = [n_positive_samples, height, width]
            The positive samples.
        N_ : array-like of shape = [n_negetive_samples, height, width]
            The negetive samples.
        shuffle : bool
            Whether to shuffle the data or not.
        """
        assert np.shape(P_) == np.shape(N_), "Window sizes mismatch."
        self.detectWndH, self.detecWndW = np.shape(P_)

        features_cnt, descriptions = \
            self.Haarlike.determineFeatures(self.detecWndW, self.detectWndH)

        if shuffle:
            # If P_ is a list, this is faster than
            # P_ = np.array(skshuffle(P_, random_state=1))
            P_ = skshuffle(np.array(P_), random_state=1)
            N_ = skshuffle(np.array(N_), random_state=1)

        P = np.zeros((len(P_), features_cnt))
        N = np.zeros((len(N_), features_cnt))
        for i in range(len(P_)):
            if verbose: print('Preparing positive data NO.%d.' % i)
            P[i] = self.Haarlike.extractFeatures(P_[i])
        for j in range(len(N_)):
            if verbose: print('Preparing negative data NO.%d.' % j)
            N[j] = self.Haarlike.extractFeatures(N_[i])

        self.P = P
        self.N = N
        self.features_cnt = features_cnt
        self.features_descriptions = descriptions

    def train(self):
        """Train the boosted cascade model.
        """
        assert self.detecWndW != -1 and self.detectWndH != -1 and \
               self.P != -1 and self.N != -1 and \
               self.features_cnt != -1 and \
               self.features_descriptions != -1, \
               "Please call prepare first."

        P = self.P
        N = self.N

        divlineP = int(len(P)*self.validset_rate)
        validP = P[0:divlineP]
        P = P[divlineP:len(P)]

        divlineN = int(len(N)*self.validset_rate)
        validN = N[0:divlenN]
        N = N[divlineN:len(N)]

        validset_X = np.vstack(validP, validN)
        validset_y = np.vstack(np.ones(len(validP)), np.zeros(len(validN)))
        validset_X, validset_y = skshuffle(validset_X, validset_y, random_state=1)
        
        self.SCs = []
        self.thresholds = []
        self.SCn = []
        self._initEvaluate(validset_y)

        f0 = f1 = 1.0
        D0 = D1 = 1.0
        i = 0
        while f1 > self.Ftarget:
            i += 1
            f0 = f1
            D0 = D1
            n = 0
            
            while f1 > self.f * f0:
                n += 1
        
                training_X = np.vstack(P, N)
                training_X = training_X[:, 0:n]
                training_y = np.vstack(np.ones(len(P)), np.zeros(len(N)))
                training_X, training_y = skshuffle(training_X, training_y, random_state=1)
        
                classifier = copy.deepcopy(self.SCClass)
                classifier.train(training_X, training_y)
                self.SCs.append(classifier)
                self.thresholds.append(1.0)
                self.SCn.append(n)
                
                ySync, f1, D1, _ = self._evaluate(validset_X, validset_y)
                ind = len(self.thresholds)-1
                while D1 > self.d * D0:
                    self.thresholds[ind] -= self.CIsteps # guess so
                    ySync, f1, D1, _ = self._evaluate(validset_X, validset_y)

                self._updateEvaluate(ySync)

            if f1 > self.Ftarget:
                _, _, _, N = self,_evaluate(self.N, np.zeros(len(self.N)))

    def _initEvaluate(self, validset_y):
        """Initialize before evaluating the model.

        Parameters
        ----------
        validset_y : np.array of shape = [n_samples]
            The ground truth of the valid set.
        """
        self._eval = {}
        self._eval.PySelector = (y_ == 1)
        self._eval.NySelector = (y_ == 0)
        self._eval.cP = np.sum(y_[self._eval.PySelector])
        self._eval.cN = np.sum(y_[self._eval.NySelector])
        self._eval.ySync = np.ones(len(X_)) # All exist possible positive
        pass

    def _evaluate(self, X_, y_):
        """Evaluate the model, but won't update any parameter of the model.

        Parameters
        ----------
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the valid set.
        y_ : np.array of shape = [n_samples]
            The ground truth of the valid set.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result.
        f : float
            The false positive rate.
        d : float
            The detection rate.
        fp : np.array
            The false positives.
        """
        ind = len(self.thresholds)-1
        ySync = self._eval.ySync

        yiPred, CI = self.SCs[ind].predict(X_[ySync == 1][:, 0:self.SCn[ind]])
        # Reject if confidence is less that thresholds
        yiPred[yiPred == 1][CI[yiPred == 1] < self.thresholds[ind]] = 0
        ySync[ySync == 1][yiPred == 0] = 0 # Exclude all rejected

        fp = (ySync[self._eval.NySelector] == 1)
        dp = (ySync[self._eval.PySelector] == 1)
        f = np.sum(fp) / self._eval.cN
        d = np.sum(dp) / self._eval.cP

        return ySync, f, d, fp

    def _updateEvaluate(self, ySync_):
        """Update the parameter of the evaluating model.

        Parameters
        ----------
        ySync_ : np.array of shape = [n_samples]
            The classifier result generated by function 'evaluate'.
        """
        self._eval.ySync = ySync # Update ySync

    def _weakPredict(self, wcself, X_):
        """Predict function for weak classifiers.

        Parameters
        ----------
        wcself : instance of WeakClassifier
            The weak classifier.
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result of the testing samples.
        """

        description = self.features_descriptions[wcself.bestn]

        feature = np.zeros(len(X_))
        for ind in range(X_):
            feature[ind] = self.Haarlike._getFeatureIn(
                X_[ind],        # integral image
                description[0], # haartype
                description[1], # x
                description[2], # y
                description[3], # w
                description[4]  # h
            )

        h = np.ones(len(X_))
        h[feature*wcself.bestd < wcself.bestp*wcself.bestd] = -1
        return h

    def _strongPredict(self, scself, X_):
        """Predict function for the strong classifier (AdaBoostClassifier).

        Parameters
        ----------
        scself : instance of self.SCClass
            The strong classifier (AdaBoostClassifier).
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        CI : np.array of shape = [n_samples]
            The confidence of each predict result.
        """
        hsum = 0
        for i in range(scself.nWC):
            hsum = hsum + scself.alpha[i] * self._weakPredict(scself.WCs[i], X_)

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0
        CI = abs(hsum) / np.sum(scself.alpha)

        return yPred, CI

    def predict(self, test_set_):
        """Predict whether it's a face or not.

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, height, width]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        """
        X = np.zeros((len(test_set_), self.detectWndH, self.detecWndW))
        for i in len(test_set_):
            X[i] = self.Haarlike._getIntegralImage(test_set_[i])

        yPred = np.ones(len(X))
        for ind in range(len(self.thresholds)):
            yiPred, CI = self._strongPredict(self.SCs[ind], X[yPred == 1])
            yiPred[yiPred == 1][CI[yiPred == 1] < self.thresholds[ind]] = 0
            yPred[yPred == 1][yiPred == 0] = 0 # Exclude all rejected
        
        return yPred
