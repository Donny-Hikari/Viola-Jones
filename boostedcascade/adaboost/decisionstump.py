
# Author: Donny

import math
import time
import sys
import multiprocessing as mp
import numpy as np

class DecisionStumpClassifier:
    """A decision stump classifier

    Parameters
    ----------
    steps_ : int, optional
        The steps to train on each feature.
    """

    def __init__(self, steps_=400, max_parallel_processes_=8):
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
        self.max_parallel_processes = max_parallel_processes_

    def train(self, X_, y_, W_, verbose=False):
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

        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = y_ if type(y_) == np.ndarray else np.array(y_)
        W = W_ if type(W_) == np.ndarray else np.array(W_)
        steps = self.steps

        n_samples, n_features = X.shape
        assert n_samples == y.size

        processes = [None] * self.max_parallel_processes
        schedules = [None] * self.max_parallel_processes
        results = [None] * self.max_parallel_processes

        blocksize = math.ceil(n_features / self.max_parallel_processes)
        if blocksize <= 0: blocksize = 1
        for tid in range(self.max_parallel_processes):
            schedules[tid] = mp.Value('f', 0.0)
            results[tid] = mp.Queue()
            blockbegin = blocksize * tid
            if blockbegin >= n_features: break; # Has got enough processes
            blockend = blocksize * (tid+1)
            if blockend > n_features: blockend = n_features
            processes[tid] = mp.Process(target=__class__._parallel_optimize,
                args=(self, tid, (blockbegin, blockend), results[tid], schedules[tid], X, y, W, steps))
            processes[tid].start()
        
        if verbose:
            while True:
                alive_processes = [None] * self.max_parallel_processes
                for tid in range(self.max_parallel_processes):
                    alive_processes[tid] = processes[tid].is_alive()
                if sum(alive_processes) == 0:
                    break

                for tid in range(self.max_parallel_processes):
                    schedule = schedules[tid].value
                    print('% 7.1f%%' % (schedule*100), end='')
                print('\r', end='', flush=True)

                time.sleep(0.2)

            sys.stdout.write("\033[K") # Clear line

        bestn = 0
        bestd = 1
        bestp = 0
        minerr = W.sum()

        for tid in range(self.max_parallel_processes):
            processes[tid].join()
            result = results[tid].get()
            if result['minerr'] < minerr:
                minerr = result['minerr']
                bestn = result['bestn']
                bestd = result['bestd']
                bestp = result['bestp']

        self.features = n_features
        self.bestn = bestn
        self.bestd = bestd
        self.bestp = bestp

        return minerr

    def _parallel_optimize(self, tid, range_, result_output, schedule, X, y, W, steps):
        assert type(range_) == tuple

        bestn = 0
        bestd = 1
        bestp = 0
        minerr = W.sum()
        
        for n in range(range_[0], range_[1]):
            # setting and getting a float value is thread-safe
            schedule.value = (n - range_[0]) / (range_[1] - range_[0])
            err, d, p = self._optimize(X[:, n], y, W, steps)
            if err < minerr:
                minerr = err
                bestn = n
                bestd = d
                bestp = p
        
        result = dict()
        result['bestn'] = bestn
        result['bestd'] = bestd
        result['bestp'] = bestp
        result['minerr'] = minerr
        result_output.put(result)

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

        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)
        n_samples, n_features = test_set.shape

        assert n_features == self.features

        single_feature = test_set[:, self.bestn]
        h = np.ones((n_samples))
        h[single_feature*self.bestd < self.bestp*self.bestd] = -1
        return h
