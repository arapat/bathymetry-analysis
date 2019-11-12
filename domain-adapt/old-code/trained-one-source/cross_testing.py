#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import sys
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from time import time

def get_model_and_testing():
    pattern = r"[a-zA-Z]+"
    model_pkl = "model.pkl"
    testing_file = "testing.libsvm"
    ret = []
    for root, subdirs, files in os.walk("./"):
        if "backup" not in root and root.endswith("libsvm"):
            files = (
                re.findall(pattern, root)[0],
                (
                    root,
                    os.path.join(root, model_pkl),
                    os.path.join(root, testing_file),
                ),
            )
            for filename in files[1]:
                assert(os.path.exists(filename))
            ret.append(files)
    return ret


# for performance
t0 = time()

def logger(s, show_time=False):
    if show_time:
        print("Current time: %.2f" % time())
    print("[%.5f] %s" % (time() - t0, s))
    sys.stdout.flush()


def run_test(testing_path, pkl_model_path, max_bin):
    # get true labels
    testing = lgb.Dataset(testing_path, params={'max_bin': max_bin})
    testing.construct()
    true = (testing.get_label() > 0) * 2 - 1
    testing = lgb.Dataset(testing_path, params={'max_bin': max_bin})

    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        pkl_bst = pickle.load(fin)

    # Prediction
    preds = pkl_bst.predict(testing_path)
    logger('finished prediction')

    # compute auprc
    scores = preds
    loss = np.mean(np.exp(-true * scores))
    precision, recall, _ = precision_recall_curve(true, scores, pos_label=1)
    # precision[-1] = np.sum(true > 0) / true.size
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    auroc = auc(fpr, tpr)

    logger("eval, {}, {}, {}, {}, {}".format(
        testing_path, pkl_bst.num_trees(), loss, auprc, auroc))
    return scores


def persist_score(root, model_id, test_id, scores):
    filename = os.path.join(root, "scores_ms_{}_{}".format(model_id, test_id))
    with open(filename, 'wb') as fout:
        pickle.dump(scores, fout)



ret = get_model_and_testing()
max_bin = 63
for model_id, (_, model, _) in ret:
    for test_id, (test_root, _, test) in ret:
        if test_id == "JAMSTEC" or test_id == "NGDC":
            continue
        logger("Using model {} to predict {}".format(model_id, test_id))
        scores = run_test(test, model, max_bin)
        persist_score(test_root, model_id, test_id, scores)

