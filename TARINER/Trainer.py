from ast import arg
from pyexpat import model
from MODEL import (
    CNN1d,
    MLP,
    LARS,
    SVR,
    KNN,
    RF,
    AdaBoost,
    XGBoost,
    Bayes
)
import matplotlib.pyplot as plt
import os
import numpy as np
from DATALOADER.read_data import (
    v_names,
    v_outp,
    v_inp,
    doubles,
    singles,
    barr,
    double_mut,
    single_mut,
    todo,
    siteall
)

# integrate all models needed
class Trainer:
    def __init__(self, args, x_train, y_train, x_test, y_test):
        self.args = args
        self.x_train, self.y_train = np.expand_dims(x_train, axis=-1), y_train
        self.x_test, self.y_test = np.expand_dims(x_test, axis=-1), y_test
        if args.model == 'cnn':
            self.model = CNN1d(args=args)
            self.model.build()
        elif args.model == 'mlp':
            self.model = MLP(args=args)
            self.model.build()
        elif args.model == 'lars':
            self.model = LARS(args=args)
        elif args.model == 'svr':
            self.model = SVR(args=args)
        elif args.model == 'knn':
            self.model = KNN(args=args)
        elif args.model == 'rf':
            self.model = RF(args=args)
        elif args.model == 'adaboost':
            self.model = AdaBoost(args=args)
        elif args.model == 'xgboost':
            self.model = XGBoost(args=args)
        elif args.model == 'bayes':
            self.model = Bayes(args=args)
        else:
            ValueError(f'No implemented model {args.model}')
        self.history = self.results = None

    def train(self):
        print(type(self.y_train), self.y_train.dtype)
        # return a dict
        self.history = self.model.train(
            x_train=self.x_train,
            y_train=self.y_train
        )

    def test(self):
        self.results = self.model.test(x_test=self.x_test, y_test=self.y_test)
        print("test loss, test mse:", self.results)

    def vis(self):
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        # plot loss curve
        plt.plot(self.history['loss'], label='Train', c='r')
        plt.plot(self.history['val_loss'], label='Val', c='b')
        plt.title(f'{self.args.model} loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(f'{self.args.out_dir}/{str(self.args.model)}_Loss.jpg')

        # plot mse curve
        plt.cla()
        plt.plot(self.history['mse'], label='Train', c='r')
        plt.plot(self.history['val_mse'], label='Val', c='b')
        plt.title(f'{self.args.model} MSE')
        plt.ylabel('Mse')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(f'{self.args.out_dir}/{str(self.args.model)}_MSE.jpg')

        if self.args.model == 'xgboost':
            self.model.vis()
    
    # training and redict the output top10 and bottom10
    def metrics(self):
        names = np.empty((0, 2), dtype=str)
        value = np.empty((0, 18, 1), dtype=float)
        
        if doubles:
            for n in range(len(barr[0])-1):
                for m in range(len(barr)-1):
                    if n == m :
                        continue
                    for n2 in range(len(barr[0])-1):
                        if n <= n2 :
                            continue
                        for m2 in range(len(barr)-1):
                            if n2 == m2 :
                                continue
                            value = np.append(value,double_mut(barr[0][n+1] + barr[m+1][0],barr[0][n2+1] + barr[m2+1][0]),axis=0)
                            names = np.append(names,[[barr[0][n+1] + barr[m+1][0], barr[0][n2+1] + barr[m2+1][0]]],axis=0)
        
        if singles:
            for i in todo:
                for s in siteall(i):
                    names = np.append(names,[[s,'']],axis=0)
                    value = np.append(value,single_mut(s),axis=0)
        
        tmp = self.model.predict(value.reshape(-1, 18)).reshape(-1, 1)
        value = np.empty((len(names), 2))
        for n in range(len(names)):
            value[n][0] = tmp[2*n]
            value[n][1] = tmp[2*n+1]
            
        f = open("prediction_nativ.dat", "w")
        for n in range(len(names)):
            f.write("{0:10s} {1:10s} {2:6.2f} {3:6.2f}\n".format(names[n][0],names[n][1],value[n][0],value[n][1]))
        f.close()
        
        means = np.mean(value, axis=1)
        names = names[np.argsort(means)]
        value = value[np.argsort(means)]
            
        # print top 10 and bottom 10 for validation
        print("Top 10:")
        for n in range(10):
            print("{0:10s} {1:10s} {2:6.2f} {3:6.2f}".format(names[n][0],names[n][1],value[n][0],value[n][1]))
        
        print("Bottom 10:")
        for n in range(10):
            print("{0:10s} {1:10s} {2:6.2f} {3:6.2f}".format(names[-n-1][0],names[-n-1][1],value[-n-1][0],value[-n-1][1]))
        
        f = open("prediction.dat", "w")
        for n in range(len(names)):
            f.write("{0:10s} {1:10s} {2:6.2f} {3:6.2f}\n".format(names[n][0],names[n][1],value[n][0],value[n][1]))
        f.close()

    # validating the predict value and real value
    def evaluate(self):
        x = self.x_test
        if self.args.model != 'cnn':
            x = np.squeeze(self.x_test)
        pred = self.model.predict(x)
        length = len(pred)
        pred = pred.reshape(length, 1)
        print(type(pred), pred.shape)
        mse = 0
        a1 = 0
        b1 = 0
        b2 = 0
        for i in range(int(len(v_outp) / 2)):
            print(
                "{0:10s} {1:10s} {2:6.2f} {3:6.2f} {4:6.2f}"
                    .format(v_names[2 * i], v_names[2 * i + 1],
                            pred[2 * i][0], pred[2 * i + 1][0],
                            v_outp[2 * i]))
            mse += ((pred[2 * i][0] + pred[2 * i + 1][0]) / 2 - v_outp[2 * i]) ** 2
            a1 += (pred[2 * i][0] - np.mean(pred)) * (v_outp[2 * i] - np.mean(v_outp)) + (
                    pred[2 * i + 1][0] - np.mean(pred)) * (v_outp[2 * i] - np.mean(v_outp))
            b1 += (pred[2 * i][0] - np.mean(pred)) ** 2 + (pred[2 * i + 1][0] - np.mean(pred)) ** 2
            b2 += (v_outp[2 * i] - np.mean(v_outp)) ** 2 * 2
        print("RMSD:    {0:6.2f}".format(np.sqrt(mse / len(v_outp))))
        print("Pearson: {0:6.2f}".format(a1 / (np.sqrt(b1) * np.sqrt(b2))))
        plt.cla()
        plt.scatter(v_outp, pred, c='r')
        plt.plot((min(v_outp), max(v_outp)), (min(v_outp), max(v_outp)), label='refer', c='b')
        plt.legend('upper left')
        plt.xlabel("calc.")
        plt.ylabel("pred.")
        plt.title("Scatter plot of True vs Predictions({})".format(
            {str(self.args.model)}))
        plt.savefig(
            f'{self.args.out_dir}/{str(self.args.model)}_image.jpg')
        with open("validation.dat", "w") as f:
            for i in range(int(len(v_outp) / 2)):
                f.write(
                    "{0:10s} {1:10s} {2:6.2f} {3:6.2f}\n"
                        .format(v_names[2 * i], v_names[2 * i + 1],
                                (pred[2 * i][0] + pred[2 * i + 1][0]) / 2,
                                v_outp[2 * i])
                )
