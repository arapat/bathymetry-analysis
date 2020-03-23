import lightgbm as lgb
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from .common import print_ts
from .load_data import persist_model

def train(config, train_dataset, valid_dataset, region, logger):
    logger.log("start training...")
    # Strange bug exists that prevents saving all iterations if `early_stopping_rounds` is enabled
    config["early_stopping_rounds"] = None
    valid_sets = [train_dataset]
    if valid_dataset is not None:
        valid_sets.append(valid_dataset)
    try:
        gbm = lgb.train(
            config,
            train_dataset,
            num_boost_round=config["rounds"],
            valid_sets=valid_sets,
            callbacks=[print_ts(logger)],
            # early_stopping_rounds=config["early_stopping_rounds"],
            # fobj=expobj, feval=exp_eval,
        )
    except:
        logger.log("Failed to train, {}".format(region))
        return
    logger.log("training completed.")
    persist_model(config["base_dir"], region, gbm)
    logger.log("Model for {} is persisted".format(region))


def get_scores(region, test_region, features, labels, pkl_model_path, logger):
    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        model = pickle.load(fin)

    # Prediction
    preds = model.predict(features)
    scores = np.clip(preds, 1e-15, 1.0 - 1e-15)
    logger.log('finished prediction')

    # compute auprc
    loss = np.mean(labels * -np.log(scores) + (1 - labels) * -np.log(1.0 - scores))
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(labels == (scores > 0.5)) / labels.shape[0]

    logger.log("eval, {}, {}, {}, {}, {}, {}, {}".format(
        region, test_region, model.num_trees(), loss, auprc, auroc, acc))
    return scores
