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
