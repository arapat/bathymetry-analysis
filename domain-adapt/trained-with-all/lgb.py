#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys
import pickle
import lightgbm as lgb
import multiprocessing
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from time import time


# In[12]:
DATA_TYPE = {
    "M": 1,  # - multibeam
    "G": 2,  # - grid
    "S": 3,  # - single beam
    "P": 4,  # - point measurement
}


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
    txt_model_path = os.path.join(base_dir, 'model.txt')
    pkl_model_path = os.path.join(base_dir, 'model.pkl')
    gbm.save_model(txt_model_path)
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    return (txt_model_path, pkl_model_path)


# In[ ]:


def run_test(features_test, labels_test, pkl_model_path):
    # format raw testing input
    features = np.concatenate(features_test, axis=0)
    true = np.array(labels_test)
    true = (true > 0) * 2 - 1

    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        model = pickle.load(fin)

    # Prediction
    preds = model.predict(features)
    logger('finished prediction')

    # compute auprc
    scores = preds
    loss = np.mean(np.exp(-true * scores))
    precision, recall, _ = precision_recall_curve(true, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum((true > 0) == (scores > 0)) / true.shape[0]

    with open("testing_result.pkl", 'wb') as fout:
        pickle.dump((true, scores), fout)

    logger("eval, {}, {}, {}, {}, {}".format(model.num_trees(), loss, auprc, auroc, acc))


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
def construct_data(features, labels, max_bin):
    all_train = lgb.Dataset(features, label=labels, params={'max_bin': max_bin})
    all_train.construct()
    size = all_train.num_data()
    # Split training and validating
    thr = int(size * 0.1)
    training = all_train.subset(list(range(0, thr)))
    validating = all_train.subset(list(range(thr, size)))
    # scale pos weight
    labels = all_train.get_label()
    positive = np.array(labels).sum()
    negative = size - positive
    scale_pos_weight = 1.0 * negative / positive
    return ((training, validating), scale_pos_weight)


# In[ ]:


def get_datasets(filepaths, get_label=lambda cols: cols[4] == '9999'):
    removed_features = [0, 1, 3, 4, 5, 7]
    features_list = []
    labels = []
    for filename in filepaths:
        filename = filename.strip()
        if not filename:
            continue
        features = []
        with open(filename) as fread:
            for line in fread:
                cols = line.strip().split()
                if not cols:
                    continue
                cols[29] = DATA_TYPE[cols[29]]
                labels.append(get_label(cols))
                features.append(np.array(
                    [float(cols[i]) for i in range(len(cols)) if i not in removed_features]
                ))
        features_list.append(np.array(features))
        # if len(features_list) > 200:
        #     break
    return (features_list, np.array(labels))


def main(config):
    base_dir = config["base_dir"]
    with open(config["training_files"]) as f:
        training_files = f.readlines()

    logger("start constructing datasets")
    (features_train, labels_train) = get_datasets(training_files)
    (train, valid), scale_pos_weight = construct_data(
        features_train, labels_train, config["max_bin"])
    logger("start training")
    model = train_lgb(
        train,
        valid,
        scale_pos_weight,
        config["num_leaves"],
        config["rounds"],
        config["early_stopping_rounds"],
        config["max_bin"],
    )
    logger("finished training")
    (_, pkl_model_path) = persist_model(base_dir, model)
    logger("model is persisted at {}".format(pkl_model_path))

    logger("start testing")
    with open(config["testing_files"]) as f:
        testing_files = f.readlines()
    (features_test, labels_test) = get_datasets(testing_files)
    logger("finished loading testing data")
    pkl_model_path = os.path.join(base_dir, 'model.pkl')
    run_test(features_test, labels_test, pkl_model_path)
    logger("finished testing")


# In[ ]:


DATA_BASE_DIR = "/geosat2/julaiti/tsv_all"
TRAINING_FILES_DESC = os.path.join(DATA_BASE_DIR, "training_files_desc.txt")
TESTING_FILES_DESC = os.path.join(DATA_BASE_DIR, "testing_files_desc.txt")
config = {
    "base_dir": "./",
    "training_files": TRAINING_FILES_DESC,
    "testing_files": TESTING_FILES_DESC,
    "num_leaves": 31,
    "rounds": 400,
    "max_bin": 63,
    "early_stopping_rounds": 16,
}


if __name__ == '__main__':
    main(config)
    flog.close()

