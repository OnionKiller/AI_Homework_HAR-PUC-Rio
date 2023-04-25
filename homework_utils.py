import matplotlib.pyplot as plt #type: ignore
import numpy as np
from sklearn.model_selection import KFold, cross_validate #type: ignore
from sklearn.pipeline import Pipeline #type: ignore

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
        axes[i, 0].legend(['Train', 'Test'], loc='lower right')

        # summarize history for loss
        axes[i, 1].plot(history.history['loss'])
        axes[i, 1].plot(history.history['val_loss'])
        axes[i, 1].set_title(f'Model Loss {test_matrice[i]}')
        axes[i, 1].set_ylabel('Loss')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].legend(['Train', 'Test'], loc='lower right')

    plt.tight_layout()
    plt.show()


def score_from_history(history):
    return {k:v[-1] for k,v in history.history.items()}

def print_historical_scores(score_dict):
    for k,v in score_dict.items():
        print(f"{k} = {v}")

def wrap_cross_validate(pipe:Pipeline,kf:KFold,x:np.ndarray,y:np.ndarray):
    # Use cross_val_score to evaluate the pipeline on the data
    scores = cross_validate(pipe, x, y,
    cv=kf,
    #n_jobs=2,
    scoring=
    {
        'accuracy': 'accuracy',
        'loss': 'neg_log_loss',
        'mse': 'neg_mean_squared_error',
    })
    return scores

def print_scores(scores):
    ce_scores = scores['test_loss']
    mse_scores = scores['test_mse']
    acc_scores = scores['test_accuracy']

    # Print the mean and standard deviation of the scores
    print('Cross Entropy: {:.3f} (+/- {:.3f})'.format(-ce_scores.mean(), ce_scores.std()))
    print('MSE: {:.3f} (+/- {:.3f})'.format(-mse_scores.mean(), mse_scores.std()))
    print('Accuracy: {:.3f} (+/- {:.3f})'.format(acc_scores.mean(), acc_scores.std()))