import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error


def eval_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def eval_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def eval_auroc(y_true, y_pred):
    return roc_auc_score(y_true > -2, y_pred)


def eval_model(model, val_X, y_true):
    y_pred = model.predict(val_X)
    rmse = eval_rmse(y_true, y_pred)
    r2 = eval_r2(y_true, y_pred)
    auroc = eval_auroc(y_true, y_pred)

    return {'rmse': rmse, 'r2': r2, 'auroc': auroc}
