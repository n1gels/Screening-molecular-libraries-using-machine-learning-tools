import json
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from TARINER import Trainer
from DATALOADER.read_data import (
    x_train,
    y_train,
    x_test,
    y_test
)
print('test for read data...Done')


def parse():
    """
    Return parse object, some parameters are pre-set
    command lineis updating config.json
    :return: parse 
    """
    with open('configs.json', 'r') as f: # get the command line from configs.json
        cfgs = json.load(f)
    parser = argparse.ArgumentParser() # update the parameters in command line
    for key in cfgs.keys():
        # print(type(cfgs[key]))
        parser.add_argument('--{}'.format(key), type=type(cfgs[key]), default=cfgs[key])
    configs = parser.parse_args()  # Return the parameter from command line
    return configs


def main(cfgs):
    print("\nconfigures ====\n", cfgs)
    trainer = Trainer(
        args=cfgs,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    trainer.train()
    trainer.test()
    trainer.vis()
    trainer.evaluate()
    #trainer.metrics()



if __name__ == '__main__':
    args = parse()
    # Manually set k_fold as False.
    args.k_fold = False
    main(cfgs=args)
