import argparse
import itertools
import os
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from surrogates.base_predictor import BasePredictor


def mle_loss(y_true, y_pred):
    # Minimum likelihood estimate loss function
    mean = tf.slice(y_pred, [0, 0], [-1, 1])
    var = tf.slice(y_pred, [0, 1], [-1, 1])
    return 0.5 * tf.log(2 * np.pi * var) + tf.square(y_true - mean) / (2 * var)


def mape_loss(y_true, y_pred):
    # Minimum absolute percentage error loss function
    lower_bound = 4.5
    fraction = tf.math.divide(tf.subtract(y_pred, lower_bound),
                              tf.subtract(y_true, lower_bound))
    return tf.abs(tf.subtract(fraction, 1))


class MetaNeuralEnsemble(BasePredictor):
    def __init__(self, n_ensemble=5, ):
        super(MetaNeuralEnsemble, self).__init__()
        self.n_ensembles = n_ensemble
        self.metanns = [MetaNeuralnet() for _ in range(self.n_ensembles)]

    def fit(self, xtrain, ytrain, **kwargs):
        for i, metann in enumerate(self.metanns):
            self.metanns[i].fit(xtrain, ytrain, **kwargs)

    def predict(self, x_eval, **kwargs):
        all_predicts = []
        for metann in self.metanns:
            all_predicts.append(
                np.squeeze(metann.predict(x_eval))
            )
        mean = np.mean(all_predicts, axis=0)
        variance = np.var(all_predicts, axis=0)
        return mean, variance

    def save(self, save_path):
        if not os.path.exists(f'{save_path}'):
            os.makedirs(f'{save_path}')
        for i in range(self.n_ensembles):
            self.metanns[i].model.save(f'{save_path}/model_{i}')

    def load(self, save_path):
        self.metanns = []
        for i in range(self.n_ensembles):
            model = keras.models.load_model(f'{save_path}/model_{i}')
            meta_neural_net = MetaNeuralnet()
            meta_neural_net.model = model
            self.metanns.append(meta_neural_net)


class MetaNeuralnet:

    def get_dense_model(self,
                        input_dims,
                        num_layers,
                        layer_width,
                        loss,
                        regularization):
        input_layer = keras.layers.Input(input_dims)
        model = keras.models.Sequential()

        for _ in range(num_layers):
            model.add(keras.layers.Dense(layer_width, activation='relu'))

        model = model(input_layer)
        if loss == 'mle':
            mean = keras.layers.Dense(1)(model)
            var = keras.layers.Dense(1)(model)
            var = keras.layers.Activation(tf.math.softplus)(var)
            output = keras.layers.concatenate([mean, var])
        else:
            if regularization == 0:
                output = keras.layers.Dense(1)(model)
            else:
                reg = keras.regularizers.l1(regularization)
                output = keras.layers.Dense(1, kernel_regularizer=reg)(model)

        dense_net = keras.models.Model(inputs=input_layer, outputs=output)
        return dense_net

    def fit(self, xtrain, ytrain,
            num_layers=10,
            layer_width=20,
            loss='mape',
            epochs=500,
            batch_size=32,
            lr=.01,
            verbose=0,
            regularization=0,
            **kwargs):

        if isinstance(xtrain, torch.Tensor): xtrain = xtrain.detach().cpu().numpy()
        if isinstance(ytrain, torch.Tensor): ytrain = ytrain.detach().cpu().numpy()

        if loss == 'mle':
            loss_fn = mle_loss
        elif loss == 'mape':
            loss_fn = mape_loss
        else:
            loss_fn = 'mae'

        self.model = self.get_dense_model((xtrain.shape[1],),
                                          loss=loss_fn,
                                          num_layers=num_layers,
                                          layer_width=layer_width,
                                          regularization=regularization)
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=.9, beta_2=.99)

        self.model.compile(optimizer=optimizer, loss=loss_fn)
        # print(self.model.summary())
        self.model.fit(xtrain, ytrain,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose)

        train_pred = np.squeeze(self.model.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def predict(self, xtest):
        return self.model.predict(xtest)
