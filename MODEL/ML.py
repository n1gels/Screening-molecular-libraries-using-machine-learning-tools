from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class ML:
    def __init__(self, args):
        self.model = None
        self.args = args
        self.seed = args.seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

    def train(self, x_train, y_train):
        x_train = np.squeeze(x_train)
        if self.args.k_fold:
            print('Using k-fold cross validation')
            return self.train_with_kfold(
                x_train=x_train,
                y_train=y_train
            )
        else:
            print('No-using k-fold cross validation')
            return self.train_without_kfold(
                x_train=x_train,
                y_train=y_train
            )

    def train_without_kfold(self, x_train, y_train):
        x_train_ds, x_val_ds, y_train_ds, y_val_ds = \
            train_test_split(x_train, y_train, test_size=0.1, random_state=self.args.seed)
        self.model.fit(
            x_train_ds,
            y_train_ds,
        )
        train_loss = mean_squared_error(y_train_ds, y_pred=self.model.predict(x_train_ds))
        val_loss = mean_squared_error(y_val_ds, y_pred=self.model.predict(x_val_ds))
        res = {
            "loss": [train_loss],
            "mse": [train_loss],
            "val_loss": [val_loss],
            "val_mse": [val_loss]
        }
        # print(res)
        return res

    def train_with_kfold(self, x_train, y_train):
        kfold = KFold(n_splits=self.args.k_split, shuffle=True, random_state=100)
        keys = ['loss', 'mse', 'val_loss', 'val_mse']
        res = {}
        for key in keys:
            res[key] = []
        x_train, y_train = np.squeeze(x_train), np.squeeze(y_train)
        for train, val in kfold.split(x_train, y_train):
            x_train_ds, y_train_ds = x_train[train], y_train[train]
            x_val_ds, y_val_ds = x_train[val], y_train[val]
            self.model.fit(
                x_train_ds,
                y_train_ds,
            )
            # loss, val loss == mse, val mse == train mse, val mse
            train_loss = tf.losses.mean_squared_error(
                y_pred=self.model.predict(x_train_ds),
                y_true=y_train_ds).numpy()
            res['loss'].append(train_loss)
            res['mse'].append(train_loss)

            val_loss = tf.losses.mean_squared_error(self.model.predict(x_val_ds), y_val_ds).numpy()
            res['val_loss'].append(val_loss)
            res['val_mse'].append(val_loss)
        return res

    def test(self, x_test, y_test):
        # Evaluate the model on the test data using `evaluate`
        x_test, y_test = np.squeeze(x_test), np.squeeze(y_test)
        test_loss = tf.losses.mean_squared_error(self.model.predict(x_test), y_test).numpy()
        print(r"Adjusted R square: {}".format(self.model.score(x_test, y_test)))
        print("Evaluate on test data")
        results = [test_loss, test_loss]
        return results

    def predict(self, x_train):
        return self.model.predict(x_train)


class LARS(ML):
    # Least-angle regression, LARS is used to high dimention data
    def __init__(self, args):
        super().__init__(args)
        self.model = linear_model.Lars(
            n_nonzero_coefs=np.inf,
            normalize=False
        )


class SVR(ML):
    def __init__(self, args):
        super().__init__(args)
        self.model = make_pipeline(StandardScaler(), svm.SVR(C=3))

    def test(self, x_test, y_test):
        # Evaluate the model on the test data using `evaluate`
        x_test, y_test = np.squeeze(x_test), np.squeeze(y_test)
        test_loss = tf.losses.mean_squared_error(self.model.predict(x_test), y_test).numpy()
        print("Evaluate on test data")
        results = [test_loss, test_loss]
        x_test = (x_test - np.min(x_test)) / np.max(x_test)
        print(r"Adjusted R square: {}".format(self.model.score(x_test, y_test)))
        return results


class KNN(ML):
    def __init__(self, args):
        super().__init__(args)
        self.model = KNeighborsRegressor(n_neighbors=5)


class RF(ML):
    def __init__(self, args):
        super().__init__(args)
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=args.seed
        )


class AdaBoost(ML):
    def __init__(self, args):
        super().__init__(args)
        self.model = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=5),
            random_state=args.seed,
            n_estimators=150
        )


# Gradient Boosting regression
class XGBoost(ML):
    def __init__(self, args):
        super().__init__(args)
        params = {
            "n_estimators": 1000,
            "eta": 0.01,  # similar to learning rate
            "max_depth": 4,
            "reg_alpha": 0.5,  # L1 used in high dimension
            "subsample ": 0.7,
            "colsample_bytree": 0.8,
            "eval_metric": mean_squared_error
        }
        self.model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            reg_alpha=params["reg_alpha"],
            subsample=params["subsample "],
            colsample_bytree=params["colsample_bytree"],
            eval_metric=params["eval_metric"],
            learning_rate=params["eta"]
        )

    def train_with_kfold(self, x_train, y_train):
        kfold = KFold(n_splits=self.args.k_split, shuffle=True, random_state=100)
        keys = ['loss', 'mse', 'val_loss', 'val_mse']
        res = {}
        for key in keys:
            res[key] = []
        x_train, y_train = np.squeeze(x_train), np.squeeze(y_train)
        for train, val in kfold.split(x_train, y_train):
            x_train_ds, y_train_ds = x_train[train], y_train[train]
            x_val_ds, y_val_ds = x_train[val], y_train[val]
            self.model.fit(
                x_train_ds,
                y_train_ds,
                eval_set=[(x_train_ds, y_train_ds), (x_val_ds, y_val_ds)],
                verbose=False
            )
            # loss, val loss == mse, val mse == train mse, val mse
            train_loss = tf.losses.mean_squared_error(self.model.predict(x_train_ds), y_train_ds).numpy()
            res['loss'].append(train_loss)
            res['mse'].append(train_loss)

            val_loss = tf.losses.mean_squared_error(self.model.predict(x_val_ds), y_val_ds).numpy()
            res['val_loss'].append(val_loss)
            res['val_mse'].append(val_loss)
        return res

    def train_without_kfold(self, x_train, y_train):
        x_train_ds, x_val_ds, y_train_ds, y_val_ds = \
            train_test_split(x_train, y_train, test_size=0.1, random_state=self.args.seed)
        self.model.fit(
            x_train_ds,
            y_train_ds,
            eval_set=[(x_train_ds, y_train_ds), (x_val_ds, y_val_ds)],
            verbose=False
        )
        train_loss = mean_squared_error(y_train_ds, y_pred=self.model.predict(x_train_ds))
        val_loss = mean_squared_error(y_val_ds, y_pred=self.model.predict(x_val_ds))
        res = {
            "loss": [train_loss],
            "mse": [train_loss],
            "val_loss": [val_loss],
            "val_mse": [val_loss]
        }
        return res

    def vis(self):
        val_res = self.model.evals_result()
        train_loss = val_res['validation_0']['rmse']
        val_loss = val_res['validation_1']['rmse']
        print(len(train_loss), len(val_loss))
        plt.clf()
        plt.plot(train_loss, label='Train', c='r')
        plt.plot(val_loss,  label='Val', c='b')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(f'{self.args.out_dir}/XGBoost.jpg')


class Bayes(ML):
    def __init__(self, args):
        super().__init__(args)
        self.model = BayesianRidge(
            n_iter=300,
            tol=1e-6,
            fit_intercept=True,
            compute_score=True
        )
