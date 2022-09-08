from tensorflow.python.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from .configs_cnn import (
    n_features,
    n_outcomes,
    dropout
)
import tensorflow as tf
import random
import numpy as np


class NeuralNets:
    def __init__(self, args):
        self.model = None
        self.args = args
        self.seed = args.seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

    def build(self):
        self.model.compile(
            optimizer='adam',
            loss="mean_squared_error",
            metrics=["mse"])
        print(self.model.summary())

    def train(self, x_train, y_train):
        if self.args.model == 'mlp':
            x_train, y_train = np.squeeze(x_train), np.squeeze(y_train)
        if self.args.k_fold:
            return self.train_with_kfold(
                x_train=x_train,
                y_train=y_train
            )
        else:
            return self.train_without_kfold(
                x_train=x_train,
                y_train=y_train
            )

    def train_with_kfold(self, x_train, y_train):
        kfold = KFold(n_splits=self.args.k_split, shuffle=True, random_state=self.seed)
        keys = ['loss', 'mse', 'val_loss', 'val_mse']
        res = {}
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-8,
                                   mode="auto", restore_best_weights=True, patience=1000)]
        for key in keys:
            res[key] = []
        if self.args.epochs // self.args.k_split <= 0:
            ValueError("The args.epochs should be larger than args.k_split")
        for train, val in kfold.split(x_train, y_train):
            x_train_ds, y_train_ds = x_train[train], y_train[train]
            x_val_ds, y_val_ds = x_train[val], y_train[val]
            hist = self.model.fit(
                x_train_ds,
                y_train_ds,
                epochs=self.args.epochs // self.args.k_split,
                batch_size=self.args.batch_size,
                validation_data=(x_val_ds, y_val_ds),
                verbose=1,
                shuffle=True,
                callbacks=[callbacks]
            )
            for key in res:
                res[key].extend(hist.history[key])
        return res

    def train_without_kfold(self, x_train, y_train):
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-8,
                                   mode="auto", restore_best_weights=True, patience=1000)]
        hist = self.model.fit(
            x_train,
            y_train,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            validation_split=0.2,
            verbose=1,
            shuffle=True,
            callbacks=[callbacks]
        )
        return hist.history

    def test(self, x_test, y_test):
        # Evaluate the model on the test data using `evaluate`
        if self.args.model == 'mlp':
            x_test, y_test = np.squeeze(x_test), np.squeeze(y_test)
        print("Evaluate on test data")
        results = self.model.evaluate(x_test, y_test)
        return results

    def predict(self, x_train):
        return self.model.predict(x_train)

# deep learning models
class MLP(NeuralNets):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = self._initModel()

    @staticmethod
    def _initModel():
        model = Sequential()
        model.add(Dense(32, input_dim=n_features))
        model.add(Activation('relu'))
        model.add(Dense(64, input_shape=(n_features,), activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, input_shape=(n_features,), activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(n_outcomes, activation='linear'))
        return model


class CNN1d(NeuralNets):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = self._initModel()
        self.seed = args.seed

    @staticmethod
    def _initModel():
        model = Sequential()
        model.add(Convolution1D(512, 1, input_shape=(n_features, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dropout(dropout))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(n_outcomes, activation='linear'))
        return model
