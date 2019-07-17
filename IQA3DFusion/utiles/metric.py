import numpy as np
from scipy.optimize import leastsq
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch import nn
import json


class Metric(nn.Module):

    def __init__(self):
        super(Metric, self).__init__()
        self.results = {
            'ep': [],
            'train_loss': [],
            'test_loss': [],
            'test_plcc': [],
            'test_srocc': [],
            'test_rmse': [],

            'max_plcc': -1,
            'max_srocc': -1,
            'min_rmse': 10e8
        }

    def forward(self, dmos, predict_dmos):
        dmos = np.asarray(dmos, dtype=np.float32)
        predict_dmos = np.asarray(predict_dmos, dtype=np.float32)

        def logistic(bayta, X):
            bayta1 = bayta[0]
            bayta2 = bayta[1]
            bayta3 = bayta[2]
            bayta4 = bayta[3]
            bayta5 = bayta[4]
            logisticPart = 0.5 - 1 / (1 + np.exp(bayta2 * (X - bayta3)))
            return bayta1 * logisticPart + bayta4 * X + bayta5

        def err(bayta, x, y):
            return logistic(bayta, x) - y

        predict_dmos = predict_dmos.round(4)
        dmos = dmos.round(4)

        beta = [10, 0, predict_dmos.mean(), 0.1, 0.1]
        para = leastsq(err, beta, args=(predict_dmos, dmos))

        fit_predict_dmos = logistic(para[0], predict_dmos)

        plcc = pearsonr(dmos, fit_predict_dmos)
        srocc = spearmanr(dmos, predict_dmos)
        krocc = kendalltau(dmos, predict_dmos)
        rmse = np.sqrt(np.sum((fit_predict_dmos - dmos) ** 2) / len(dmos))

        return np.abs(plcc[0]), np.abs(srocc[0]), np.abs(rmse), np.abs(krocc[0])

    def update_train(self, train_loss):
        self.results['train_loss'].append(train_loss)
        return 0

    def update_test(self, ep, test_loss, test_plcc, test_srocc, test_rmse, max_plcc=None, max_srocc=None, min_rmse=None):
        self.results['ep'].append(ep)
        self.results['test_loss'].append(test_loss)
        self.results['test_plcc'].append(test_plcc)
        self.results['test_srocc'].append(test_srocc)
        self.results['test_rmse'].append(test_rmse)

        if max_plcc is not None:
            self.results['max_plcc'] = max_plcc
        if max_srocc is not None:
            self.results['max_srocc'] = max_srocc
        if min_rmse is not None:
            self.results['min_rmse'] = min_rmse
        return 0

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False)
        return 0

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.result = json.load(f)
        return self.results


def eval4metric(dmos, predict_dmos):
    dmos = np.asarray(dmos, dtype=np.float32)
    predict_dmos = np.asarray(predict_dmos, dtype=np.float32)

    def logistic(bayta, X):
        bayta1 = bayta[0]
        bayta2 = bayta[1]
        bayta3 = bayta[2]
        bayta4 = bayta[3]
        bayta5 = bayta[4]
        logisticPart = 0.5 - 1 / (1 + np.exp(bayta2 * (X - bayta3)))
        return bayta1 * logisticPart + bayta4 * X + bayta5

    def err(bayta, x, y):
        return logistic(bayta, x) - y

    predict_dmos = predict_dmos.round(4)
    dmos = dmos.round(4)

    beta = [10, 0, predict_dmos.mean(), 0.1, 0.1]
    para = leastsq(err, beta, args=(predict_dmos, dmos))

    fit_predict_dmos = logistic(para[0], predict_dmos)

    plcc = pearsonr(dmos, fit_predict_dmos)
    srocc = spearmanr(dmos, predict_dmos)
    krocc = kendalltau(dmos, predict_dmos)
    rmse = np.sqrt(np.sum((fit_predict_dmos - dmos) ** 2) / len(dmos))

    return np.abs(plcc[0]), np.abs(srocc[0]), np.abs(rmse), np.abs(krocc[0])






