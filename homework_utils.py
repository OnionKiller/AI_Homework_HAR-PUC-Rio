import matplotlib.pyplot as plt #type: ignore

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
    return {
    'accuracy': history.history['accuracy'][-1],
    'validation_accuracy': history.history['val_accuracy'][-1],
    'loss': history.history['loss'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'mse': history.history['mean_squared_error'][-1],
    'validation_mse': history.history['val_mean_squared_error'][-1]
}

def print_scores(score_dict):
    for k,v in score_dict.items():
        print(f"{k} = {v}")