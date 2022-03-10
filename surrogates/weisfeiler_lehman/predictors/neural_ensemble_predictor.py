import numpy as np
from tensorflow import keras
import tensorflow as tf
import torch
from surrogates.weisfeiler_lehman.wl_extractor import WeisfeilerLehmanExtractor
from surrogates.weisfeiler_lehman.predictors.base_wl_predictor import BaseWLPredictor
from copy import deepcopy
from surrogates.weisfeiler_lehman.utils import to_unit_cube, from_unit_normal, to_unit_normal


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


class NeuralEnsemble(BaseWLPredictor):

    def __init__(self, num_ensemble=3, h=1, extractor_mode: str = 'categorical',
                 node_attr='op_name',
                 **meta_nn_kwargs):
        """
        Implementation of the Neural Ensemble predictor which uses an ensemble of neural network to predict mean and
            predictive variance for Bayesian optimisation.
        :param num_ensemble: number of ensembles. In general, a higher number of ensembles lead to more diverse pred
            which usually lead to better-calibrated uncertainty estimates; however, this also leads to longer training
            time
        :param h: the number of Weisfeiler-Lehman iterations
        :param extractor_mode: see the extractor documentation
        :param meta_nn_kwargs: any optional keyword arguments to be passed to MetaNN (the member class of the ensemble)
        """
        super().__init__(h=h, )
        self.meta_nn_kwargs = meta_nn_kwargs
        self.num_ensemble = num_ensemble
        self.metanns = [MetaNeuralnet(**meta_nn_kwargs) for _ in range(num_ensemble)]
        self.extractor = WeisfeilerLehmanExtractor(h=h, mode=extractor_mode, node_attr=node_attr)

    def fit(self, x_train: list, y_train: torch.Tensor):
        """Same as GPWL"""
        if len(y_train.shape) == 0:  # y_train is a scalar
            y_train = y_train.reshape(1)
        assert len(x_train) == y_train.shape[0]
        assert y_train.ndim == 1
        # Fit the feature extractor with the graph input
        self.extractor.fit(x_train)
        self.X = deepcopy(x_train)
        self.y = deepcopy(y_train)
        x_feat_vector = self.extractor.get_train_features()

        self.lb, self.ub = np.min(x_feat_vector, axis=0), np.max(x_feat_vector, axis=0)

        x_feat_vector_normalized = to_unit_cube(x_feat_vector, self.lb, self.ub)

        y_train_local = y_train.detach().numpy()
        self.ymean, self.ystd = np.mean(y_train_local), np.std(y_train_local)
        y_train_normal = to_unit_normal(y_train_local, self.ymean, self.ystd)

        for metann in self.metanns:
            metann.fit(x_feat_vector_normalized, y_train_normal)

    def update(self, x_update: list, y_update: torch.Tensor):
        """Same as GPWL"""
        if len(y_update.shape) == 0:  # y_train is a scalar
            y_update = y_update.reshape(1)
        assert len(x_update) == y_update.shape[0]
        assert y_update.ndim == 1
        self.extractor.update(x_update)
        x_feat_vector = self.extractor.get_train_features()
        self.lb, self.ub = np.min(x_feat_vector, axis=0), np.max(x_feat_vector, axis=0)
        x_feat_vector_normalized = to_unit_cube(x_feat_vector, self.lb, self.ub)

        self.X += deepcopy(x_update)
        self.y = torch.cat((self.y, y_update))
        y = to_unit_normal(deepcopy(self.y).numpy(), self.ymean, self.ystd)

        for metann in self.metanns:
            metann.fit(x_feat_vector_normalized, y)

    def predict(self, x_eval: list, include_noise_variance=False, **kwargs):
        """Same as GPWL but the MetaNN cannot output full covariance so the that argument is omitted.
        Note that MetaNN also does not explicitly separate the noise variances from the posterior variance, so if
            include noise_variance is False, we simply empirically substract the variance by the minimum variance
            of all samples in x_eval.
        """
        x_feat_vector = self.extractor.transform(x_eval)
        x_feat_vector = to_unit_cube(x_feat_vector, self.lb, self.ub)

        all_predicts = []
        for metann in self.metanns:
            all_predicts.append(
                np.squeeze(metann.predict(x_feat_vector))
            )
        all_predicts = np.array(all_predicts)
        mean = np.mean(all_predicts, axis=0)
        variance = np.var(all_predicts, axis=0)
        mean = from_unit_normal(mean, self.ymean, self.ystd)
        variance = from_unit_normal(variance, self.ymean, self.ystd, scale_variance=True)
        if not include_noise_variance:
            variance -= np.min(variance)
        std = np.sqrt(variance)
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class MetaNeuralnet:

    def __init__(self, num_layers=10, layer_width=20, loss='mae', epochs=200, batch_size=32,
                 lr=0.01, verbose=False, regularization=0):
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.regularization = regularization

    def get_dense_model(self,
                        input_dims):
        input_layer = keras.layers.Input(input_dims)
        model = keras.models.Sequential()

        for _ in range(self.num_layers):
            model.add(keras.layers.Dense(self.layer_width, activation='relu'))

        model = model(input_layer)
        if self.loss == 'mle':
            mean = keras.layers.Dense(1)(model)
            var = keras.layers.Dense(1)(model)
            var = keras.layers.Activation(tf.math.softplus)(var)
            output = keras.layers.concatenate([mean, var])
        else:
            if self.regularization == 0:
                output = keras.layers.Dense(1)(model)
            else:
                reg = keras.regularizers.l1(self.regularization)
                output = keras.layers.Dense(1, kernel_regularizer=reg)(model)

        dense_net = keras.models.Model(inputs=input_layer, outputs=output)
        return dense_net

    def fit(self, xtrain, ytrain):

        if self.loss == 'mle':
            loss_fn = mle_loss
        elif self.loss == 'mape':
            loss_fn = mape_loss
        else:
            loss_fn = 'mae'

        # print(xtrain, ytrain)

        self.model = self.get_dense_model((xtrain.shape[1],),)
        optimizer = keras.optimizers.Adam(lr=self.lr, beta_1=.9, beta_2=.99)

        self.model.compile(optimizer=optimizer, loss=loss_fn)
        if self.verbose:
            print(self.model.summary())
        self.model.fit(xtrain, ytrain,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verbose)

        train_pred = np.squeeze(self.model.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def predict(self, xtest):
        return self.model.predict(xtest)