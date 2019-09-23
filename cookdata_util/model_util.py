__all__ = ('is_model_fit', 'default_param_on_model', 'load_data_into',
           'roc_on_data', 'roc_auc_test', 'roc_auc_train',
           'accuracy_on_data', 'accuracy_test', 'accuracy_train',
           'persistent', 'dump_to_json', 'smote_test_pack', 'load_from_json', 'BestParam')
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from .constants import CONSTANTS, check_constants


def roc_on_data(model, X, y):
    y_predict = model.predict_proba(X)[:, 1]
    return np.round(roc_auc_score(y, y_predict), 6)


def roc_auc_test(model, test_pack):
    if len(test_pack) != 4:
        raise ValueError("Length of pack must be 4")
    _, X_test, _, y_test = test_pack
    return roc_on_data(model, X_test, y_test)


def roc_auc_train(model, test_pack):
    if len(test_pack) != 4:
        raise ValueError("Length of pack must be 4")
    X_train, _, y_train, _ = test_pack
    return roc_on_data(model, X_train, y_train)


def accuracy_on_data(model, X, y):
    return np.round(accuracy_score(y, model.predict(X)), 6)


def accuracy_test(model, test_pack):
    _, X_test, _, y_test = test_pack
    return accuracy_on_data(model, X_test, y_test)


def accuracy_train(model, test_pack):
    X_train, _, y_train, _ = test_pack
    return accuracy_on_data(model, X_train, y_train)


def is_model_fit(model, test_pack, measure='auc'):
    if measure == 'auc':
        return {
            'test auc': roc_auc_test(model, test_pack),
            'train auc': roc_auc_train(model, test_pack),
        }
    elif measure == 'accuracy':
        return {
            'test accuracy': accuracy_test(model, test_pack),
            'train accuracy': accuracy_train(model, test_pack),
        }
    else:
        raise ValueError(f'measure={measure}')


def default_param_on_model(classifier, test_packs, fit_param=None):
    if not callable(classifier.fit):
        raise ValueError('Classifier do not support fit')
    try:
        support_predict_proba = callable(classifier.predict_proba)
    except AttributeError:
        support_predict_proba = False

    res = dict()
    for pack_name, test_pack in test_packs.items():
        X_train, X_test, y_train, y_test = test_pack

        fit_begin = time.perf_counter()
        if isinstance(fit_param, dict):
            classifier.fit(X_train, y_train, **fit_param)
        else:
            classifier.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_begin

        if support_predict_proba:
            res[pack_name] = is_model_fit(classifier, test_pack)
        else:
            res[pack_name] = is_model_fit(classifier, test_pack, measure='accuracy')
        res[pack_name]['time'] = np.round(fit_time, 6)

        print(f'{pack_name}: {res[pack_name]}')
    return res


# X_train, X_validate, X_test, y_train, y_validate, y_test
def build_test_validate_pack(test_pack):
    X_train, X_test, y_train, y_test = test_pack

    new_X_train, X_validate, new_y_train, y_validate = train_test_split(
        X_train, y_train, test_size=0.2,
        random_state=CONSTANTS['RANDOM_STATE'], stratify=y_train)

    return new_X_train, X_validate, X_test, new_y_train, y_validate, y_test


def load_data_into(test_packs, test_validate_packs, test_data_cols=None):
    check_constants()

    from_dir = CONSTANTS['DATA_LOAD_FROM']
    random_state = CONSTANTS['RANDOM_STATE']
    available_data = [dname.replace('.csv', '') for dname in os.listdir(from_dir)]

    for dname in available_data:
        print('正在加载：', dname)
        data = pd.read_csv(f'{from_dir}{dname}.csv', encoding='utf-8')

        y = data['Default'].values
        x = data.drop(['Default'], axis=1).values

        if isinstance(test_data_cols, dict):
            test_data_cols[dname] = [*data.drop(['Default'], axis=1).columns]

        test_packs[dname] = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
        test_validate_packs[dname] = build_test_validate_pack(test_packs[dname])


def persistent(obj, name):
    joblib.dump(obj, CONSTANTS['PREFIX'] + name, compress=True)


def dump_to_json(json_obj, name):
    with open(CONSTANTS['PREFIX'] + f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(json_obj, f)

def load_from_json(name):
    with open(CONSTANTS['PREFIX'] + f'{name}.json', 'r', encoding='utf-8') as f:
        res = json.load(f)

    return res


def smote_test_pack(test_pack):
    smote = SMOTE(random_state=CONSTANTS['RANDOM_STATE'])
    X_train, X_test, y_train, y_test = test_pack
    smote_X_train, smote_y_train = smote.fit_sample(X_train, y_train)
    result_test_pack = (smote_X_train, X_test, smote_y_train, y_test)
    return result_test_pack

class BestParam:
    __slots__ = ('pack_name', 'params')

    def __init__(self, pack_name, params):
        self.pack_name = pack_name
        self.params = params

