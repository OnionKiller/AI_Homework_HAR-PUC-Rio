from dataclasses import dataclass, field
import inspect
from io import BytesIO
import json
import logging
from typing import Dict, List, Type
import matplotlib.pyplot as plt #type: ignore
import numpy as np
from sklearn.discriminant_analysis import StandardScaler #type:ignore
from sklearn.model_selection import KFold, cross_validate #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
import tensorflow as tf #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

@dataclass
class MatrixTestResult:
    testcase:dict
    histories:list[tf.keras.callbacks.History] = field(default_factory=list)


def figure_from_histories(histories,test_matrice):
    # set up the figure
    fig, axes = plt.subplots(nrows=len(histories), ncols=2, figsize=(10, 12))

    # loop through the histories and plot each one
    for i, history in enumerate(histories):
        # summarize history for accuracy
        axes[i, 0].plot(history.history['accuracy'])
        axes[i, 0].plot(history.history['val_accuracy'])
        axes[i, 0].set_title(f'Model Accuracy {test_matrice[i]}')
        axes[i, 0].set_ylabel('Accuracy')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].legend(['Train', 'Validation'], loc='lower right')

        # summarize history for loss
        axes[i, 1].plot(history.history['loss'])
        axes[i, 1].plot(history.history['val_loss'])
        axes[i, 1].set_title(f'Model Loss {test_matrice[i]}')
        axes[i, 1].set_ylabel('Loss')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.show()

def score_from_history(history):
    return {k:v[-1] for k,v in history.history.items()}

def score_from_histories(histories):
    r = dict()
    for history in histories:
        scores = score_from_history(history)
        for k,v in scores.items():
            if k not in r:
                r[k] = list()
            r[k].append(v)
    r = {k:np.asarray(v) for k,v in r.items()}
    return r


    for k,v in score_dict.items():
        print(f"{k} = {v}")

def print_score_from_MatrixTestResult(results:MatrixTestResult):
    print(results.testcase)
    scores = score_from_histories(results.histories)
    print_scores(scores)
    print("="*30)

def run_test(testcase,model_creator,model_creator_params,x,y)->MatrixTestResult:

    def create_file_name(dict_obj):
        json_str = json.dumps(dict_obj)
        file_name = "".join(x if x.isalnum() else "" for x in json_str)
        return file_name
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r = MatrixTestResult(testcase)
    for i_fold, (train_indices, val_indices) in enumerate(kf.split(x)):
        X = x[train_indices]
        Y = to_categorical(y[train_indices])
        Xv = x[val_indices]
        Yv = to_categorical(y[val_indices])

        batch_size = 10**2
        

        X = StandardScaler().fit_transform(X).astype(np.float16)
        Xv = StandardScaler().fit_transform(Xv).astype(np.float16)
        log_fname = "checkpoints/"+create_file_name(testcase) + 'model.{epoch:02d}.h5'
        frequency = int(X.shape[0] / batch_size * 10) # save every 10 epochs
        my_callbacks = [
            #tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=log_fname,save_freq=frequency),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{create_file_name(testcase)}_{i_fold}',
                histogram_freq=1,
                #profile_batch='10,30'
                ),
        ]

        tf.keras.backend.clear_session()
        model = model_creator(**model_creator_params)
        history = model.fit(X,Y,
        epochs=50,
        batch_size=batch_size,
        validation_data=(Xv,Yv),
        callbacks=my_callbacks,
        verbose=0)
        r.histories.append(history)
    return r


def wrap_test_case(testcase_list,create_modell,x,y,)->List[MatrixTestResult]:
    if not inspect.isfunction(create_modell):
        raise ValueError("create_modell s not a function!")
    
    results = list()

    attributes_whitelist = inspect.getfullargspec(create_modell).args
    for testcase in testcase_list:
        logging.info(f"Running testcase {testcase}")
        unexpected_args = {k:v for k,v in testcase.items() if k not in attributes_whitelist}
        if len(unexpected_args) > 0:
            logging.warn(f"extra attributes detected: {unexpected_args}")
        create_args = {k:v for k,v in testcase.items() if k in attributes_whitelist}
        res = run_test(testcase,create_modell,create_args,x,y)
        results.append(res)
    
    return results

def wrap_cross_validate(pipe:Pipeline,kf:KFold,x:np.ndarray,y:np.ndarray):
    # Use cross_val_score to evaluate the pipeline on the data
    scores = cross_validate(pipe, x, y,
    cv=kf,
    #n_jobs=2,
    scoring=
    {
        'accuracy': 'accuracy',
        'loss': 'neg_log_loss',
        'neg_mean_squared_error': 'neg_mean_squared_error',
    })
    return scores

def print_scores(scores, validation_only = True):
    ce_scores = scores['test_loss'] if 'test_loss' in scores else scores['loss']
    mse_scores = scores['neg_mean_squared_error'] if 'neg_mean_squared_error' in scores else scores['mean_squared_error']
    acc_scores = scores['test_accuracy'] if 'test_accuracy' in scores else scores['accuracy']

    if not validation_only:
        # Print the mean and standard deviation of the scores
        print('Test Cross Entropy: {:.4f} (+/- {:.4f})'.format(np.absolute(ce_scores.mean()), ce_scores.std()))
        print('Test MSE: {:.4f} (+/- {:.4f})'.format(np.absolute(mse_scores.mean()), mse_scores.std()))
        print('Test Accuracy: {:.4f} (+/- {:.4f})'.format(acc_scores.mean(), acc_scores.std()))

        print('-'*30)
    

    ce_scores = scores['val_loss']
    mse_scores = scores['val_neg_mean_squared_error'] if 'neg_mean_squared_error' in scores else scores['val_mean_squared_error']
    acc_scores = scores['val_accuracy']

    # Print the mean and standard deviation of the scores
    print('Validation Cross Entropy: {:.4f} (+/- {:.4f})'.format(np.absolute(ce_scores.mean()), ce_scores.std()))
    print('Validation MSE: {:.4f} (+/- {:.4f})'.format(np.absolute(mse_scores.mean()), mse_scores.std()))
    print('Validation Accuracy: {:.4f} (+/- {:.4f})'.format(acc_scores.mean(), acc_scores.std()))

