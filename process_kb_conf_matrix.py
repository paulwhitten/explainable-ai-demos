#!/usr/bin/env python

import pickle
import argparse
from multiprocessing import Process
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_recall_fscore_support, accuracy_score, matthews_corrcoef, f1_score, cohen_kappa_score
from load_mnist_data import load_mnist_float, load_mnist_labels
import json

"""
This program reprocesses all of the transform data in the knowledgebase to build confusion matrices
"""

te_f = open("trans_explainability.json")
TRANS_EXPLAINABILITY = json.load(te_f)
te_f.close()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# calculates the sensitivity, if zero denominator, returns 0.0
def calc_sensitivity(tp, fn):
    result = 0.0
    if tp + fn > 0:
        result = tp / (tp + fn)
    return result

# calculates the specificity, if zero denominator, returns 0.0
def calc_specificity(tn, fp):
    result = 0.0
    if tn + fp > 0:
        result = tn / (tn + fp)
    return result

# get one versus others labels
def get_ovo_labels(tl, tp, one):
    otl = []
    otp = []
    for i, l in enumerate(tl):
        if l == one:
            otl.append(1)
        else:
            otl.append(0)
        if tp[i] == one:
            otp.append(1)
        else:
            otp.append(0)
    return otl, otp

def eval_transforms(batch, kb_folder):
    print("====", batch, "====", flush=True)
    batch_start_time = datetime.now()

    trans_stat = {
        "name": batch,
        "explainability": TRANS_EXPLAINABILITY[batch],
        "classes": []
    }
    # "train_class_stats" : [],
    # "test_class_stats" : [],

    # load the kb data to process

    ## load the training data
    tr_label_f = open(kb_folder + "/" + batch + ".train-labels.json")
    train_labels = json.load(tr_label_f)
    tr_label_f.close()
    tr_pred_f = open(kb_folder + "/" + batch + ".train.json")
    train_pred = json.load(tr_pred_f)
    tr_pred_f.close()
    ## load the test data
    te_label_f = open(kb_folder + "/" + batch + ".test-labels.json")
    test_labels = json.load(te_label_f)
    te_label_f.close()
    te_pred_f = open(kb_folder + "/" + batch + ".test.json")
    test_pred = json.load(te_pred_f)
    te_pred_f.close()
    ## load the auc data
    #auc_f = open(kb_folder + "/" + batch + "_auc.json")
    #aucs = json.load(auc_f)
    #auc_f.close()

    min_label = min(train_labels)
    max_label = max(train_labels)
    labels = range(min_label, max_label+1)
    
    train_matrix = confusion_matrix(train_labels, train_pred)
    test_matrix = confusion_matrix(test_labels, test_pred)
    trans_stat["train_cm"] = train_matrix
    trans_stat["test_cm"] = test_matrix

    ml_cm_train = multilabel_confusion_matrix(train_labels, train_pred, labels=labels)
    ml_cm_test = multilabel_confusion_matrix(test_labels, test_pred, labels=labels)

    prfs_train = precision_recall_fscore_support(train_labels, train_pred)
    prfs_test = precision_recall_fscore_support(test_labels, test_pred)

    trans_stat["acc"] = accuracy_score(test_labels, test_pred)
    trans_stat["f1"] = f1_score(test_labels, test_pred, average='macro')
    trans_stat["mcc"] = matthews_corrcoef(test_labels, test_pred)
    if trans_stat["mcc"] < 0.0:
        print("mcc was less than zero", trans_stat["mcc"], flush=True) 
    trans_stat["cohen_kappa"] = cohen_kappa_score(test_labels, test_pred)

    for i, l in enumerate(labels):
        train_tn, train_fp, train_fn, train_tp = ml_cm_train[i].ravel()
        test_tn, test_fp, test_fn, test_tp = ml_cm_test[i].ravel()
        ov_train_labels, ov_train_preds = get_ovo_labels(train_labels, train_pred, l)
        label_stat = {
            "class_label": l,
            "train_tn": train_tn, "train_fp": train_fp, "train_fn": train_fn, "train_tp": train_tp,
            "test_tn": test_tn, "test_fp": test_fp, "test_fn": test_fn, "test_tp": test_tp,
            "train_acc": (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn),
            "test_acc": (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn),
            "train_balanced_acc": (prfs_train[1][i] + calc_specificity(train_tn, train_fp) ) / 2.0,
            "test_balanced_acc": (prfs_test[1][i] + calc_specificity(test_tn, test_fp) ) / 2.0,
            "train_sensitivity": calc_sensitivity(train_tp, train_fn),
            "test_sensitivity": calc_sensitivity(test_tp, test_fn),
            "train_specificity": calc_specificity(train_tn, train_fp),
            "test_specificity": calc_specificity(test_tn, test_fp),
            "train_precision": prfs_train[0][i], "train_recall": prfs_train[1][i], "train_f_score": prfs_train[2][i], "train_support": prfs_train[3][i],
            #"train_auc": aucs[i],
            "mcc": matthews_corrcoef(ov_train_labels, ov_train_preds),
            "cohen_kappa": cohen_kappa_score(ov_train_labels, ov_train_preds)
        }
        label_stat["product"] = label_stat["train_sensitivity"] * label_stat["train_specificity"] * label_stat["train_precision"] * label_stat["train_acc"]
        trans_stat["classes"].append(label_stat)
        #TODO add accuracy, balanced accuracy, sensitivity, specificity

    # print training stats
    print("Training CM", flush=True)
    print(train_matrix, flush=True)

    print("Multi-label cm training", flush=True)
    for i, s in enumerate(ml_cm_train):
        tn, fp, fn, tp = s.ravel()
        #print("type:", type(tn))
        print("label:", min_label + i, "tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn, flush=True)
    
    print("PRFS train - precision, recall, f, support", flush=True)
    print(prfs_train, flush=True)

    # print test stats
    print("Test CM", flush=True)
    print(test_matrix, flush=True)

    print("Multi-label cm test", flush=True)
    for i, s in enumerate(ml_cm_test):
        tn, fp, fn, tp = s.ravel()
        print("label:", min_label + i, "tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn, flush=True)
    
    print("PRFS test - precision, recall, f, support", flush=True)
    print(prfs_test, flush=True)

    # with open(batch[5] + "/" + batch[0] + ".train.json", 'w') as outfile:
    #     json.dump(train_predictions.tolist(), outfile)
    # with open(batch[5] + "/" + batch[0] + ".test.json", 'w') as outfile:
    #     json.dump(test_predictions.tolist(), outfile)
    # with open(batch[5] + "/" + batch[0] + ".train-labels.json", 'w') as outfile:
    #     json.dump(train_label_list, outfile)
    # with open(batch[5] + "/" + batch[0] + ".test-labels.json", 'w') as outfile:
    #     json.dump(test_label_list, outfile)
    with open(kb_folder + "/" + batch + "-stat.json", "w") as outfile:
        json.dump(trans_stat, outfile, cls=NpEncoder, indent=4)

    batch_end_time = datetime.now()
    print(batch, "done in:", batch_end_time - batch_start_time, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trains transforms for a set')
    parser.add_argument('-k', '--knowledgebase_folder', 
                        help='The knowledgebase folder to load and output data')
    args = parser.parse_args()

    batches = [
        "raw", "crossing", "endpoint", "fill", "skel-fill", "skel", "thresh", "line", "ellipse",
        "circle", "ellipse-circle", "chull", "corner", "pca"
    ]

    start_time = datetime.now()

    processes = []
    for batch in batches:
        eval_transforms(batch, args.knowledgebase_folder)

    end_time = datetime.now()
    print("Program done in:", end_time - start_time, flush=True)
        
