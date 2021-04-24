# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import pickle
# import sys


class Policy(object):
    def __init__(self, pre_trained=None):
        if pre_trained:
            self.mlp = load(pre_trained)
        else:
            self.mlp = MLPClassifier()

    def save_model(self, filename):
        dump(self.mlp, filename)

    def train(self, x, y):
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)
        y = np.array(y)

        self.mlp.fit(x, y)

    def test(self, x, y):
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)
        y = np.array(y)
        return self.mlp.score(x, y)

    def predict(self, x):
        x = np.array(x).reshape(1, -1)

        return self.mlp.predict(x)


if __name__ == '__main__':
    training_data = pickle.load(open('datasets/train.pkl', 'rb'))
    policy = Policy()
    policy.train(training_data['x'], training_data['y'])

    test = pickle.load(open('datasets/test.pkl', 'rb'))
    score = policy.test(test['x'], test['y'])

    policy.save_model(
        filename=os.path.join('pretrained', '{:.3f}'.format(score)))

    print(score)
