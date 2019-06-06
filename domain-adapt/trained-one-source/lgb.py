#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys
import pickle
import yaml
import lightgbm as lgb
import multiprocessing
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from time import time


# In[12]:


# for performance
if os.path.exists("./log_file.txt"):
    print("Log file exists. Quit.")
    sys.exit(1)
flog = open("./log_file.txt", 'w')
t0 = time()


def logger(s, show_time=False):
    if show_time:
        msg = "Current time: %.2f" % time()
        print(msg)
        flog.write(msg + '\n')
    msg = "[%.5f] %s" % (time() - t0, s)
    print(msg)
    flog.write(msg + '\n')
    sys.stdout.flush()
    flog.flush()


def print_ts(period=1):
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
            logger('Tree %d' % (env.iteration + 1))
    callback.order = 10
    return callback


# In[ ]:


def train_lgb(train_data, valid_data, scale_pos_weight, num_leaves, rounds, early_stopping_rounds, max_bin):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',  # 'goss',
        'num_leaves': num_leaves,
        'learning_rate': 0.3,
        'tree_learner': 'voting',
        'task': 'train',
        'num_thread': multiprocessing.cpu_count(),
        'min_data_in_leaf': 5000,  # This is min bound for stopping rule in Sparrow
        'two_round': True,
        'scale_pos_weight': scale_pos_weight,
        'max_bin': max_bin,
    }

    logger('Start training...')
    gbm = lgb.train(
        params, train_data, num_boost_round=rounds,
        early_stopping_rounds=early_stopping_rounds,
        fobj=expobj, feval=exp_eval,
        valid_sets=[valid_data], callbacks=[print_ts()],
    )
    logger('Training completed.')
    return gbm


def persist_model(base_dir, gbm):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    txt_model = os.path.join(base_dir, 'model.txt')
    pkl_model = os.path.join(base_dir, 'model.pkl')
    gbm.save_model(txt_model)
    with open(pkl_model, 'wb') as fout:
        pickle.dump(gbm, fout)
    return (txt_model, pkl_model)


# In[ ]:


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

    with open(testing_path + "_score_self", 'wb') as fout:
        pickle.dump(scores, fout)

    logger("eval, {}, {}, {}, {}, {}".format(
        testing_path, pkl_bst.num_trees(), loss, auprc, auroc))


# In[ ]:


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


# In[11]:


def get_datasets():
    train = 'training.libsvm'
    test = 'testing.libsvm'
    ret = []
    logger("Listing directories:")
    for root, subdirs, files in os.walk("./"):
        if root.endswith("libsvm"):
            # if os.path.exists(os.path.join(root, "testing.libsvm_score_self")):
            #     print("skipping trained file " + root.strip())
            #     continue
            if root.strip().endswith("NGDC_libsvm") or root.strip().endswith("JAMSTEC_libsvm"):
                logger("skipping large file " + root.strip())
                continue
            logger(root)
            ret.append((
                root,
                os.path.join(root, train),
                os.path.join(root, test),
            ))
    print
    return ret


def construct_data(rtrain, max_bin):
    all_train = lgb.Dataset(rtrain, params={'max_bin': max_bin})
    all_train.construct()
    size = all_train.num_data()
    # Split training and validating
    thr = int(size * 0.1)
    training = all_train.subset(list(range(0, thr)))
    validating = all_train.subset(list(range(thr, size)))
    # scale pos weight
    labels = all_train.get_label()
    positive = labels.sum()
    negative = size - positive
    scale_pos_weight = 1.0 * negative / positive
    return ((training, validating), scale_pos_weight)


# In[ ]:


def main(config):
    datasets = get_datasets()
    for dataset in datasets:
        logger("start, {}".format(dataset[0]))
        root, rtrain, rtest = dataset
        (train, valid), scale_pos_weight = construct_data(rtrain, config["max_bin"])
        model = train_lgb(
            train,
            valid,
            scale_pos_weight,
            config["num_leaves"],
            config["rounds"],
            config["early_stopping_rounds"],
            config["max_bin"],
        )
        (_, pkl_model_path) = persist_model(root, model)
        run_test(rtest, pkl_model_path, config["max_bin"])
        logger("finish, {}".format(root))


# In[ ]:


config = {
    "num_leaves": 31,
    "rounds": 400,
    "max_bin": 63,
    "early_stopping_rounds": 16,
}


if __name__ == '__main__':
    main(config)
    flog.close()

