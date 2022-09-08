# README

## 1. overview

This is a project using various of machine learning model to predict double mutant barrier.  The input data is 18 dimensions, it is a regression problem map 18  to 1.

The models used  in projects are：CNN, MLP, LARS, SVR, KNN, Random Forest, AdaBoost, XGBoost, Bayes Regression.

## 2.Parameters

```sh
{
  "epochs": 120,
  "batch_size": 16,
  "k_split": 30,
  "out_dir": "result",
  "model": "lars",
  "seed": 100,
  "k_fold": false
}
```

- epochs， training iteration times
- batch\_size,  used for CNN, MLP
- k\_split,   k-fold cross-validation number
- out\_dir,  the folder stores output plots, will create automatlly
- model,  assign a specific number to use

  - cnn, mlp, lars, svr, knn, rf, adaboost, xgboost, bayes
- seed, set random seeds (but there still seems to be a large gap between models)
- k\_fold, choose whether to use K -Fold,

  - Pllease modify it in `MAIN.py:55`.
  - ```python
    args.k_fold = False/True
    ```

## 3.Usage

    There are two methods using this script:

1. Modify parameters in configs.json, and run MAIN.py
2. Using command line to modify parameters and run.

    For example, we want build XGBoost.

```
# Decide whether using k_fold in MAIN.py:55 | args.k_fold = False
# Decide whether predict, uncomment in MAIN.py :45 |    #trainer.metrics()
# run
>>> python -u MAIN.py --model xgboost

=========output============
Training set:   380 ins with 380 outs
Validation set:   0 ins with   0 outs
Processing single data
E:\Research\Money\Reproduction\ML(multi-models)-VSR-5号\DATALOADER\read_data.py:169: UserWarning: genfromtxt: Empty input file: "DATA/singles.csv"
  sing = np.genfromtxt(r"DATA/singles.csv", dtype=str)
Processing double data
test for read data...Done
Namespace(batch_size=16, epochs=3, k_fold=False, k_split=30, model='xgboost', out_dir='result', seed=100)
<class 'numpy.ndarray'> float64
No-using k-fold cross validation
Adjusted R square: 0.8206504710697922
Evaluate on test data
test loss, test mse: [5.353282692517315, 5.353282692517315]
1000 1000
<class 'numpy.ndarray'> (144, 1)
ALA790GLU  LEU902LYS    3.02   2.72   2.49
THR785ASP  ALA790GLU    0.04   0.56  -1.09

。。。
LEU902GLU  ALA790VAL   -4.38  -4.46  -6.13
LEU902GLU  ALA790LEU   -4.32  -4.39  -6.98
LEU902GLN  ALA790LYS    0.39   0.06  -0.68
RMSD:      1.61
Pearson:   0.92

```


There RMSD and Pearson's correlation cofficient printed in output and plots will be stored in out_dir.

<img src="images%20in%20text/image-20220903133806046.png" alt="image-20220903133806046" style="zoom:50%;" />

    `<img src="images%20in%20text/image-20220903133855818.png" alt="image-20220903133855818" style="zoom:50%;" />`

## 4. structure

### 4.1.MODEL

    Implement of models，it is the main file of building models.

    It contains two part:

- NeuralNet： neural network/deep learning model : MLP, CNN
- ML： Machine learning models
- In the implementation, the parent class implements the model in the usual way, such as training, and the different models are implemented as subclasses. Therefore, if additional models need to be added, it is only necessary to rewrite new subclasses to inherit them.

  - ```python
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

    # Fllow LARS to DIY new models
    class LARS(ML):
        # Least-angle regression， LARS
        def __init__(self, args):
            super().__init__(args)
            self.model = linear_model.Lars(
                n_nonzero_coefs=np.inf,
                normalize=False
            )
    ```

### 4.2.TRAINER

    The process control module, which has three main branches: train, test, vis。

- train： trainmodel，
- test:    test model， criteria  are mse, Adjusted R square
- vis： visualise
- evaluate： model validating, criteria are  RMSD ，Pearson's correlation cofficience
