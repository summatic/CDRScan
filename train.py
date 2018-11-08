import os
import numpy as np

import tensorflow as tf
from keras.optimizers import RMSprop
import keras.backend as K

from Datasets import Datasets
from evaluation import eval_model
from models.master import MasterModel
from models.fully_connected import FullyConnectedModel
from models.shallow import ShallowModel
from models.tanh import TanhModel
from models.unified import UnifiedModel
from utils import get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

prefix = '/home/hanseok/sandbox/cdrscan/'
cell_line_size = 28087
drug_size = 3072
es_tolerance = 10
eval_steps = 1000
log_steps = 100


def tf_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    return sess


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def load_model(name):
    if name == 'master':
        return MasterModel()()
    elif name == 'fully_connected':
        return FullyConnectedModel()()
    elif name == 'shallow':
        return ShallowModel()()
    elif name == 'tanh':
        return TanhModel()()
    elif name == 'unified':
        return UnifiedModel()()
    else:
        raise TypeError('model name is not correct')


def kfold_cross_validation(datasets, optimizer, split, name):
    k = 5

    final_results = {'rmse': [], 'r2': [], 'auroc': []}
    logger = get_logger(logger_name=name, logger_path=prefix + '/results/%s/log' % name)

    for idx in range(1, k):
        model = load_model(name)
        model.compile(optimizer=optimizer, loss=root_mean_squared_error)
        try:
            eval_results = _train(idx, model, datasets, name, split, logger)

            for k, v in eval_results.items():
                final_results[k].append(v)
        except Exception as e:
            logger.warning(e)

    avgs = {k: str(sum(v) / len(v)) for k, v in final_results.items()}
    logger.warning('[Final Results] %s' % ' '.join(['%s: %s' % (k, v) for k, v in avgs.items()]))


def _train(idx, model, datasets, name, split, logger):
    es = es_tolerance
    best_eval = 100000
    _eval = 100001
    steps = 0
    for epoch in range(100000):
        train_gen, (val_X, val_y), steps_per_epoch = datasets.kfold(idx, split)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, verbose=1)
        train_loss = history.history['loss'][0]
        logger.warning('[%s_%d|Train|Epoch %d|Steps %d] Loss: %f' % (name, idx, epoch, steps, train_loss))

        eval_results = eval_model(model, val_X, val_y)
        logger.warning('####################')
        logger.warning('[%s_%d|Valid|Epoch %d|Steps %d]' % (name, idx, epoch, steps))
        for k, v in eval_results.items():
            logger.warning('%s: %f' % (k, v))
        logger.warning('####################')
        _eval = eval_results['rmse']
        steps += 256


        # _losses = []
        # for _ in range(steps_per_epoch):
        #     try:
        #         history = model.fit_generator(generator=train_gen, steps_per_epoch=10, verbose=2)
        #     except StopIteration:
        #         train_gen, (val_X, val_y), steps_per_epoch = datasets.kfold(idx, split)
        #         print(steps, _)
        #     _losses.append(history.history['loss'][0])
        #
        #     if steps % log_steps == 0:
        #         logger.warning('[%s_%d|Epoch %d|Steps %d] Loss: %f' % (name, idx, epoch, steps, np.mean(_losses)))
        #
        #     if steps % eval_steps == 0:
        #         eval_results = eval_model(model, val_X, val_y)
        #         logger.warning('####################')
        #         logger.warning('####################')
        #         logger.warning('Epoch: %d, Steps: %d' % (epoch, steps))
        #         for k, v in eval_results.items():
        #             logger.warning('%s: %f' % (k, v))
        #         logger.warning('####################')
        #         logger.warning('####################')
        #         _eval = eval_results['rmse']
        #
        #      steps += 1

        if _eval < best_eval:
            es = es_tolerance
            # model.save(prefix + 'results/%s/%d.model' % (name, epoch))
        elif es > 0:
            es -= 1
        else:
            break
    # eval_results = eval_model(model, val_X, val_y)
    return eval_results


if __name__ == '__main__':

    ds = Datasets(prefix=prefix, mode='real')
    rmsprop = RMSprop(lr=1e-3, clipnorm=1)
    sess = tf_config()

    kfold_cross_validation(ds, optimizer=rmsprop, split=True, name='master')
    kfold_cross_validation(ds, optimizer=rmsprop, split=True, name='shallow')
    kfold_cross_validation(ds, optimizer=rmsprop, split=True, name='fully_connected')
