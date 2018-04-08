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
from . import HaarlikeFeature, HaarlikeType

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
        self.detectWndW, self.detectWndH = (-1, -1)

        self.P = self.N = []
        self.features_cnt = -1
        self.features_descriptions = []

        # self.SCs : list of self.SCClass
        #   The strong classifiers.
        self.SCs = []
        # self.thresholds : list of float
        #   The thresholds of each strong classifier.
        self.thresholds = []
        # self.SCn : list of int
        #   The number of features used.
        self.SCn = []

    def savefeaturesdata(self, filename):
        np.save(filename+'-variables', [self.detectWndH, self.detectWndW, self.features_cnt])
        np.save(filename+'-P', self.P)
        np.save(filename+'-N', self.N)
        np.save(filename+'-features_descriptions', self.features_descriptions)

    def loadfeaturesdata(self, filename):
        self.detectWndH, self.detectWndW, self.features_cnt = np.load(filename+'-variables.npy')
        self.P = np.load(filename+'-P.npy')
        self.N = np.load(filename+'-N.npy')
        self.features_descriptions = np.load(filename+'-features_descriptions.npy')

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
        assert np.shape(P_)[1:3] == np.shape(N_)[1:3], "Window sizes mismatch."
        _, self.detectWndH, self.detectWndW = np.shape(P_)

        features_cnt, descriptions = \
            self.Haarlike.determineFeatures(self.detectWndW, self.detectWndH)
        descriptions = descriptions[::-1]

        if shuffle:
            # If P_ is a list, this is faster than
            # P_ = np.array(skshuffle(P_, random_state=1))
            P_ = skshuffle(np.array(P_), random_state=1)
            N_ = skshuffle(np.array(N_), random_state=1)

        P = np.zeros((len(P_), features_cnt))
        N = np.zeros((len(N_), features_cnt))
        for i in range(len(P_)):
            if verbose: print('Preparing positive data NO.%d.' % i)
            P[i] = self.Haarlike.extractFeatures(P_[i])[::-1]
        for j in range(len(N_)):
            if verbose: print('Preparing negative data NO.%d.' % j)
            N[j] = self.Haarlike.extractFeatures(N_[i])[::-1]

        self.P = P
        self.N = N
        self.features_cnt = features_cnt
        self.features_descriptions = descriptions

    def train(self, verbose=False):
        """Train the boosted cascade model.
        """
        assert self.detectWndW != -1 and self.detectWndH != -1 and \
               len(self.P) != 0 and len(self.N) != 0 and \
               self.features_cnt != -1 and \
               len(self.features_descriptions) != 0, \
               "Please call prepare first."

        P = self.P
        N = self.N

        divlineP = int(len(P)*self.validset_rate)
        validP = P[0:divlineP]
        P = P[divlineP:len(P)]

        divlineN = int(len(N)*self.validset_rate)
        validN = N[0:divlineN]
        N = N[divlineN:len(N)]

        validset_X = np.concatenate(( validP, validN ))
        validset_y = np.concatenate(( np.ones(len(validP)), np.zeros(len(validN)) ))
        validset_X, validset_y = skshuffle(validset_X, validset_y, random_state=1)
        
        self.SCs = []
        self.thresholds = []
        self.SCn = []
        self._initEvaluate(validset_y)

        f0 = f1 = 1.0
        D0 = D1 = 1.0
        n_step = int(self.features_cnt/1000)
        if n_step == 0: n_step = 1
        i = 0
        while f1 > self.Ftarget:
            i += 1
            f0 = f1
            D0 = D1
            # n = 0
            n = int(self.features_cnt/1000)

            print('Training iteration %d, false positive rate = %f' % (i, f1))
            
            while f1 > self.f * f0:
                n += n_step
                
                print('Features count %d, detection rate = %f, false positive rate = %f'
                    % (n, D1, f1))
        
                training_X = np.concatenate(( P, N ))
                training_X = training_X[:, 0:n]
                training_y = np.concatenate(( np.ones(len(P)), np.zeros(len(N)) ))
                training_X, training_y = skshuffle(training_X, training_y, random_state=1)
        
                classifier = copy.deepcopy(self.SCClass)
                classifier.train(training_X, training_y, verbose)
                self.SCs.append(classifier)
                self.thresholds.append(1.0)
                self.SCn.append(n)
                
                ySync, f1, D1, _ = self._evaluate(validset_X, validset_y)
                ind = len(self.thresholds)-1
                while D1 < self.d * D0:
                    print('Adjusting threshold to %f, detection rate = %f, false positive rate = %f'
                        % (self.thresholds[ind], D1, f1))

                    self.thresholds[ind] -= self.CIsteps # guess so
                    if self.thresholds[ind] < 0.0: self.thresholds[ind] = 0.0
                    ySync, f1, D1, _ = self._evaluate(validset_X, validset_y)

                self._updateEvaluate(ySync)

            if f1 > self.Ftarget:
                _, _, _, fp = self,_evaluate(self.N, np.zeros(len(self.N)))
                N = self.N[fp]
        
        print('%d cascaded classifiers, detection rate = %f, false positive rate = %f'
            % (len(self.SCs), D1, f1))

    def _initEvaluate(self, validset_y):
        """Initialize before evaluating the model.

        Parameters
        ----------
        validset_y : np.array of shape = [n_samples]
            The ground truth of the valid set.
        """
        class Eval: pass
        self._eval = Eval()
        self._eval.PySelector = (validset_y == 1)
        self._eval.NySelector = (validset_y == 0)
        self._eval.cP = len(validset_y[self._eval.PySelector])
        self._eval.cN = len(validset_y[self._eval.NySelector])
        self._eval.ySync = np.ones(len(validset_y)) # All exist possible positive
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
        ySync = self._eval.ySync.copy()

        yiPred, CI = self.SCs[ind].predict(X_[ySync == 1][:, 0:self.SCn[ind]])
        # Reject those whose confidences are less that thresholds
        yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
        ySync[ySync == 1] = yiPred # Exclude all rejected

        fp = (ySync[self._eval.NySelector] == 1)
        dp = (ySync[self._eval.PySelector] == 1)
        f = (np.sum(fp) / self._eval.cN) if self._eval.cN != 0.0 else 0.0
        d = (np.sum(dp) / self._eval.cP) if self._eval.cP != 0.0 else 0.0

        return ySync, f, d, fp

    def _updateEvaluate(self, ySync):
        """Update the parameter of the evaluating model.

        Parameters
        ----------
        ySync : np.array of shape = [n_samples]
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
        for ind in range(len(X_)):
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
        X = np.zeros((len(test_set_), self.detectWndH+1, self.detectWndW+1))
        for i in range(len(test_set_)):
            X[i] = self.Haarlike._getIntegralImage(test_set_[i])

        yPred = np.ones(len(X))
        for ind in range(len(self.thresholds)):
            yiPred, CI = self._strongPredict(self.SCs[ind], X[yPred == 1])
            yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred

    def preparePredictRaw(self, P_, N_, verbose=False):
        P = np.zeros((len(P_), self.features_cnt))
        N = np.zeros((len(N_), self.features_cnt))
        for i in range(len(P_)):
            if verbose: print('Preparing positive data NO.%d.' % i)
            P[i] = self.Haarlike.extractFeatures(P_[i])[::-1]
        for j in range(len(N_)):
            if verbose: print('Preparing negative data NO.%d.' % j)
            N[j] = self.Haarlike.extractFeatures(N_[i])[::-1]

        self.P = P
        self.N = N
            
    def predictRaw(self):
        X = np.concatenate((self.P, self.N))
        yPred = np.ones(len(X))
        for ind in range(len(self.thresholds)):
            yiPred, CI = self.SCs[ind].predict(X[yPred == 1][:, 0:self.SCn[ind]])
            yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred[0:len(self.P)], yPred[len(self.P):len(self.N)]

    def saveModel(self, filename):
        np.save(filename+'-variables', [
            self.detectWndH, self.detectWndW, self.features_cnt
        ])
        np.save(filename+'-features_descriptions',
            self.features_descriptions)
        np.save(filename+'-thresholds', self.thresholds)
        np.save(filename+'-SCs', self.SCs)
        np.save(filename+'-SCn', self.SCn)

    def loadModel(filename):
        model = BoostedCascade(-1.0, -1.0, -1.0)
        model.detectWndH, model.detectWndW, model.features_cnt = \
            np.load(filename+'-variables.npy')
        model.features_descriptions = \
            np.load(filename+'-features_descriptions.npy')
        model.thresholds = np.load(filename+'-thresholds.npy')
        model.SCs = np.load(filename+'-SCs.npy')
        model.SCn = np.load(filename+'-SCn.npy')
        return model
