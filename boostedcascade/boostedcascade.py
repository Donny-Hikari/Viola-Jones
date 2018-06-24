# 
# boostedcascade.py
#   Boosted cascade proposed by Viola & Jones
# 
# Author : Donny
# 

import os
import copy
import math
import time
import sys
import multiprocessing as mp
import numpy as np
from sklearn.utils import shuffle as skshuffle

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
    validset_rate : float, optional
        The ratio of valid set in the whole training set.
    CIsteps : float, optional.
        The steps of decreasing confidence threshold in each cascaded classifier.
    """

    # Haar-like features
    Haarlike = HaarlikeFeature()
    # Class of strong classifier, usu. AdaBoostClassifier
    SCClass = AdaBoostClassifier(weak_classifier_ = DecisionStumpClassifier(100))

    def __init__(self,
                 Ftarget, f, d,
                 validset_rate = 0.3,
                 CIsteps = 0.05):
        # Target false positive rate.
        self.Ftarget = Ftarget
        # The maximum acceptable false positive rate per layer.
        self.f = f
        # The minimum acceptable detection rate per layer
        self.d = d
        # The ratio of valid set in the whole training set.
        self.validset_rate = validset_rate
        # The steps of decreasing confidence threshold in each cascaded classifier.        
        self.CIsteps = CIsteps

        # The window size of detector
        self.detectWndW, self.detectWndH = (-1, -1)

        self.P = np.array([]); self.N = np.array([])
        self.validX = np.array([]); self.validy = np.array([])
        self.features_cnt = -1
        self.features_descriptions = np.array([])

        # self.SCs : list of self.SCClass
        #   The strong classifiers.
        self.SCs = []
        # self.thresholds : list of float
        #   The thresholds of each strong classifier.
        self.thresholds = []
        # self.SCn : list of int
        #   The number of features used.
        self.SCn = []

    def getDetectWnd(self):
        return (self.detectWndH, self.detectWndW)

    def architecture(self):
        archi=""
        for ind, SC in enumerate(self.SCs):
            archi = archi + ("layer %d: %d weak classifier\n" % (ind, SC.nWC))
        return archi

    def __str__(self):
        return ("BoostedCascade(Ftarget=%f, "
                "f=%f, "
                "d=%f, "
                "validset_rate=%f, "
                "CIsteps=%f, "
                "detectWnd=%s, "
                "features_cnt=%d, "
                "n_strong_classifier=%d)") % (
                self.Ftarget,
                self.f,
                self.d,
                self.validset_rate,
                self.CIsteps,
                (self.detectWndH, self.detectWndW),
                self.features_cnt,
                len(self.SCs))

    def savefeaturesdata(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        np.save(filename+'-variables', [self.detectWndH, self.detectWndW, self.features_cnt])
        np.save(filename+'-features_descriptions', self.features_descriptions)
        np.save(filename+'-P', self.P)
        np.save(filename+'-N', self.N)
        np.save(filename+'-validX', self.validX)
        np.save(filename+'-validy', self.validy)

    def loadfeaturesdata(self, filename):
        detectWndH, detectWndW, features_cnt = \
            np.load(filename+'-variables.npy')
        self.detectWndH, self.detectWndW = int(detectWndH), int(detectWndW)
        self.features_cnt = int(features_cnt)
        self.features_descriptions = \
            np.load(filename+'-features_descriptions.npy')
        self.P = np.load(filename+'-P.npy')
        self.N = np.load(filename+'-N.npy')
        self.validX = np.load(filename+'-validX.npy')
        self.validy = np.load(filename+'-validy.npy')

    def _parallel_translate(self, tid, range_, raw_data, result_output, schedule_output):
        assert type(range_) == tuple
        
        for n in range(range_[0], range_[1]):
            schedule_output.value = (n - range_[0]) / (range_[1] - range_[0])
            x = self.Haarlike.extractFeatures(raw_data[n], self.features_descriptions)
            result_output.put((n, x))

    def _translate(self, raw_data, verbose=False, max_parallel_process=8):
        n_samples, height, width = np.shape(raw_data)

        assert height == self.detectWndH and width == self.detectWndW, \
               "Height and width mismatch with current data."

        processes = [None] * max_parallel_process
        schedules = [None] * max_parallel_process
        results = mp.Queue()

        blocksize = math.ceil(n_samples / max_parallel_process)
        if blocksize <= 0: blocksize = 1
        for tid in range(max_parallel_process):
            schedules[tid] = mp.Value('f', 0.0)
            blockbegin = blocksize * tid
            if blockbegin >= n_samples: break
            blockend = blocksize * (tid+1)
            if blockend > n_samples: blockend = n_samples
            processes[tid] = mp.Process(target=__class__._parallel_translate,
                args=(self, tid, (blockbegin, blockend), raw_data, results, schedules[tid]))
            processes[tid].start()

        X = np.zeros((n_samples, self.features_cnt))
        while True:
            alive_processes = [None] * max_parallel_process
            for tid in range(max_parallel_process):
                alive_processes[tid] = processes[tid].is_alive()
            if sum(alive_processes) == 0:
                break

            while not results.empty():
                ind, x = results.get()
                X[ind] = x

            if verbose:
                for tid in range(max_parallel_process):
                    schedule = schedules[tid].value
                    print('% 7.1f%%' % (schedule * 100), end='')
                print('\r', end='', flush=True)

            time.sleep(0.2)

        sys.stdout.write("\033[K")
        
        return X

    def prepare(self, P_, N_, shuffle=True, verbose=False, max_parallel_process=8):
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

        self.features_cnt, descriptions = \
            self.Haarlike.determineFeatures(self.detectWndW, self.detectWndH)
        self.features_descriptions = descriptions[::-1]

        if shuffle:
            # If P_ is a list, this is faster than
            # P_ = np.array(skshuffle(P_, random_state=1))
            P_ = skshuffle(np.array(P_), random_state=1)
            N_ = skshuffle(np.array(N_), random_state=1)

        if verbose: print('Preparing positive data.')
        P = self._translate(P_, verbose=verbose, max_parallel_process=max_parallel_process)
        if verbose: print('Preparing negative data.')
        N = self._translate(N_, verbose=verbose, max_parallel_process=max_parallel_process)

        divlineP = int(len(P)*self.validset_rate)
        divlineN = int(len(N)*self.validset_rate)

        validset_X = np.concatenate(( P[0:divlineP], N[0:divlineN] ))
        validset_y = np.concatenate(( np.ones(len(P[0:divlineP])), np.zeros(len(N[0:divlineN])) ))
        # validset_X, validset_y = skshuffle(validset_X, validset_y, random_state=1)

        P = P[divlineP:len(P)]
        N = N[divlineN:len(N)]

        self.P = P
        self.N = N
        self.validX = validset_X
        self.validy = validset_y

    def train(self, is_continue=False, autosnap_filename=None, verbose=False):
        """Train the boosted cascade model.
        """
        assert self.detectWndW != -1 and self.detectWndH != -1 and \
               len(self.P) != 0 and len(self.N) != 0 and \
               self.features_cnt != -1 and \
               len(self.features_descriptions) != 0, \
               "Please call prepare first."

        P = self.P
        N = self.N

        self._initEvaluate(self.validX, self.validy)

        self.P = np.array([]); self.N = np.array([])
        self.validX = np.array([]); self.validy = np.array([])

        f1 = 1.0
        D1 = 1.0

        if not is_continue:
            self.SCs = []
            self.thresholds = []
            self.SCn = []
        else:
            yPred = self._predictRaw(N)
            N = N[yPred == 1]

            for ind in range(len(self.SCs)):
                ySync, f1, D1, _ = self._evaluate(ind)
                self._updateEvaluate(ySync)

        features_used = self.features_cnt
        n_step = 1
        # n_step = int(self.features_cnt/400)
        # if n_step == 0: n_step = 1

        print('Begin training, with n_classes += %d, n_step = %d, Ftarget = %f, f = %f, d = %f'
            % (1, n_step, self.Ftarget, self.f, self.d))

        itr = 0
        while f1 > self.Ftarget:
            itr += 1
            f0 = f1
            D0 = D1
            n = 0
            # n = int(self.features_cnt/400)
            # n = self.features_cnt

            print('Training iteration %d, false positive rate = %f' % (itr, f1))

            training_X = np.concatenate(( P, N ))
            training_y = np.concatenate(( np.ones(len(P)), np.zeros(len(N)) ))
            training_X, training_y = skshuffle(training_X, training_y, random_state=1)

            classifier = copy.deepcopy(self.SCClass)

            self.SCs.append(copy.deepcopy(classifier))
            self.thresholds.append(1.0)
            self.SCn.append(features_used)

            while f1 > self.f * f0:
                n = n_step # n += n_step
                # if n > self.features_cnt: n = self.features_cnt
                ind = len(self.SCs) - 1
                
                print('Itr-%d: Training %d-th AdaBoostClassifier, features count + %d, detection rate = %f, false positive rate = %f'
                    % (itr, ind, n, D1, f1))
                if verbose:
                    print('Aim detection rate : >=%f; Aim false positive rate : <=%f'
                        % (self.d * D0, self.f * f0))
                    print('Positive samples : %s; Negative samples : %s'
                        % (str(P.shape), str(N.shape)))

                # classifier = copy.deepcopy(self.SCClass)
                # classifier.mxWC = n
                classifier.train(training_X[:, 0:features_used], training_y, n, is_continue=True, verbose=verbose)
                if verbose:
                    for a in range(classifier.nWC):
                        print('%d-th weak classifier select %s as its feature.'
                            % (a, str(self.features_descriptions[classifier.WCs[a].bestn])))
                
                self.SCs[ind] = copy.deepcopy(classifier)
                self.thresholds[ind] = 1.0
                self.SCn[ind] = features_used
                
                ySync, f1, D1, _ = self._evaluate(ind)
                print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f'
                    % (self.thresholds[ind], D1, f1))
                
                while D1 < self.d * D0:
                    self.thresholds[ind] -= self.CIsteps
                    if self.thresholds[ind] < -1.0: self.thresholds[ind] = -1.0
                    ySync, f1, D1, _ = self._evaluate(ind)
                    print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f'
                        % (self.thresholds[ind], D1, f1))
                
            self._updateEvaluate(ySync)

            if f1 > self.Ftarget:
                yPred = self._predictRaw(N)
                N = N[yPred == 1]

            if autosnap_filename:
                self.saveModel(autosnap_filename)

        print('%d cascaded classifiers, detection rate = %f, false positive rate = %f'
            % (len(self.SCs), D1, f1))

    def _initEvaluate(self, validset_X, validset_y):
        """Initialize before evaluating the model.

        Parameters
        ----------
        validset_X : np.array of shape = [n_samples, n_features]
            The samples of the valid set.
        validset_y : np.array of shape = [n_samples]
            The ground truth of the valid set.
        """
        class Eval: pass
        self._eval = Eval()
        self._eval.validset_X = validset_X
        self._eval.validset_y = validset_y
        self._eval.PySelector = (validset_y == 1)
        self._eval.NySelector = (validset_y == 0)
        self._eval.cP = len(validset_y[self._eval.PySelector])
        self._eval.cN = len(validset_y[self._eval.NySelector])
        self._eval.ySync = np.ones(len(validset_y)) # All exist possible positive
        pass

    def _evaluate(self, ind):
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
        X_ = self._eval.validset_X
        y_ = self._eval.validset_y

        ySync = self._eval.ySync.copy()

        # yiPred, CI = self.SCs[ind].predict(X_[ySync == 1][:, 0:self.SCn[ind]])
        yiPred, CI = self.SCs[ind].predict(X_[:, 0:self.SCn[ind]])
        CI[yiPred != 1] = -CI[yiPred != 1]
        # Reject those whose confidences are less that thresholds
        yiPred = (CI >= self.thresholds[ind]).astype(int)
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
        self._eval.validset_X = self._eval.validset_X[ySync[self._eval.ySync == 1] == 1]
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

    def translateToIntegralImage(self, image):
        return self.Haarlike._getIntegralImage(image)

    def predictIntegralImage(self, test_set_integral_images):
        X = test_set_integral_images
        yPred = np.ones(len(X))
        for ind in range(len(self.SCs)):
            yiPred, CI = self._strongPredict(self.SCs[ind], X[yPred == 1])
            CI[yiPred != 1] = -CI[yiPred != 1]
            yiPred = (CI >= self.thresholds[ind]).astype(int)
            # yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred

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
        return self.predictIntegralImage(X)
        
    def preparePredictRaw(self, P_, N_, verbose=False):
        P = np.zeros((len(P_), self.features_cnt))
        N = np.zeros((len(N_), self.features_cnt))
        for i in range(len(P_)):
            if verbose: print('Preparing positive data NO.%d.' % i)
            P[i] = self.Haarlike.extractFeatures(P_[i], self.features_descriptions)
        for j in range(len(N_)):
            if verbose: print('Preparing negative data NO.%d.' % j)
            N[j] = self.Haarlike.extractFeatures(N_[i], self.featuers_descriptions)

        self.P = P
        self.N = N

    def _predictRaw(self, test_set_):
        yPred = np.ones(len(test_set_))
        for ind in range(len(self.SCs)):
            yiPred, CI = self.SCs[ind].predict(test_set_[yPred == 1][:, 0:self.SCn[ind]])
            CI[yiPred != 1] = -CI[yiPred != 1]
            yiPred = (CI >= self.thresholds[ind]).astype(int)
            # yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred
            
    def predictRaw(self):
        X = np.concatenate((self.P, self.N))
        yPred = self._predictRaw(X)
        return yPred[0:len(self.P)], yPred[len(self.P):len(X)]

    def saveModel(self, filename):
        np.save(filename+'-variables', [
            self.Ftarget, self.f, self.d, self.validset_rate, self.CIsteps,
            self.detectWndH, self.detectWndW, self.features_cnt,
        ])
        np.save(filename+'-features_descriptions',
            self.features_descriptions)
        np.save(filename+'-thresholds', self.thresholds)
        np.save(filename+'-SCs', self.SCs)
        np.save(filename+'-SCn', self.SCn)

    def loadModel(filename):
        Ftarget, f, d, validset_rate, CIsteps, \
        detectWndH, detectWndW, features_cnt   \
            = np.load(filename+'-variables.npy')
        model = BoostedCascade(Ftarget, f, d, validset_rate, CIsteps)
        model.detectWndH, model.detectWndW = int(detectWndH), int(detectWndW)
        model.features_cnt = int(features_cnt)

        model.features_descriptions = \
            np.load(filename+'-features_descriptions.npy')
        model.thresholds = list(np.load(filename+'-thresholds.npy'))
        model.SCs = list(np.load(filename+'-SCs.npy'))
        model.SCn = list(np.load(filename+'-SCn.npy'))
        return model
