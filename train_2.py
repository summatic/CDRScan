import os

import tensorflow as tf
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import keras.backend as K

from Datasets import Datasets
from evaluation import eval_model
from models.master import MasterModel
from models.fully_connected import FullyConnectedModel
from models.shallow import ShallowModel
from models.tanh import TanhModel
from models.unified import UnifiedModel
from utils import get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

prefix = '/home/hanseok/sandbox/cdrscan/'
batch_size = 256
cell_line_size = 28087
drug_size = 3072
es_tolerance = 10
eval_epoch = 20


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

    for idx in range(k):
        model = load_model(name)
        model.compile(optimizer=optimizer, loss=root_mean_squared_error)
        eval_results = _train(idx, model, datasets, name, split, logger)

        for k, v in eval_results.items():
            final_results[k].append(v)

    avgs = {k: str(sum(v) / len(v)) for k, v in final_results.items()}
    logger.warning('[Final Results] %s' % ' '.join(['%s: %s' % (k, v) for k, v in avgs.items()]))


def _train(idx, model, datasets, name, split, logger):
    es = es_tolerance
    best_eval = 100000
    _eval = 100001
    for epoch in range(100000):
        train_gen, (val_X, val_y), steps_per_epoch = datasets.kfold(idx, split)
        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=[TensorBoard(prefix + 'results/%s/tflog' % name)], verbose=0)
        logger.warning('[%s_%d: Epoch %d] %f' % (name, idx, epoch, history.history['loss'][0]))

        if epoch % eval_epoch == 0:
            eval_results = eval_model(model, val_X, val_y)
            logger.warning('####################')
            logger.warning('########eval########')
            for k, v in eval_results.items():
                logger.warning('%s: %f' % (k, v))
            logger.warning('####################')
            logger.warning('####################')
            _eval = eval_results['rmse']

        if _eval < best_eval:
            es = es_tolerance
            # model.save(prefix + 'results/%s/%d.model' % (name, epoch))
        elif es > 0:
            es -= 1
        else:
            break
    eval_results = eval_model(model, val_X, val_y)
    return eval_results


if __name__ == '__main__':

    ds = Datasets(prefix=prefix, mode='real')
    rmsprop = RMSprop(lr=1e-4)
    sess = tf_config()

    kfold_cross_validation(ds, optimizer=rmsprop, split=True, name='tanh')
    kfold_cross_validation(ds, optimizer=rmsprop, split=False, name='unified')
