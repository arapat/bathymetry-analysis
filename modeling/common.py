#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import lightgbm as lgb
import numpy as np
from time import time


DATA_BASE_DIR = "/geosat2/julaiti/tsv_all"
TRAINING_FILES_DESC = os.path.join(DATA_BASE_DIR, "training_files_desc.txt")
VALIDATION_FILES_DESC = os.path.join(DATA_BASE_DIR, "validation_files_desc.txt")
TESTING_FILES_DESC = os.path.join(DATA_BASE_DIR, "testing_files_desc.txt")


class Logger:
    def __init__(self):
        self.file_handle = None
        self.starting_time = time()

    def set_file_handle(self, log_file_name):
        self.file_handle = open(log_file_name, 'w')

    def log(self, msg, show_time=False):
        if show_time:
            msg = "Current time: %.2f" % time()
            print(msg)
            self.file_handle.write(msg + '\n')

        msg = "[%.5f] %s" % (time() - self.starting_time, msg)
        print(msg)
        sys.stdout.flush()
        self.file_handle.write(msg + '\n')
        self.file_handle.flush()


def print_ts(logger, period=1):
    """Create a callback that prints the tree is generated.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to print the evaluation results.

    Returns
    -------
    callback : function
        The callback that prints the evaluation results every ``period`` iteration(s).
    """
    def callback(env):
        """internal function"""
        if period > 0 and (env.iteration + 1) % period == 0:
            logger.log('Tree %d' % (env.iteration + 1))
    callback.order = 10
    return callback


# AdaBoost potential function
"""
def expobj(preds, dtrain):
    labels = ((dtrain.get_label() > 0) * 2 - 1).astype("float16")
    hess = np.exp(-labels * preds)  # exp(-margin)
    return -labels * hess, hess     # grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
def exp_eval(preds, data):
    labels = ((data.get_label() > 0) * 2 - 1).astype("float16")
    return 'potential', np.mean(np.exp(-labels * preds)), False
"""
