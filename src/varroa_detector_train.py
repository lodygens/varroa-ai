import os
import argparse
from varroa_detector_earlystop import VarroaDetector
import matplotlib.pyplot as plt

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classification_accuracy'])
    plt.plot(history.history['val_classification_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()



def main():
    
    parser = argparse.ArgumentParser(description='Train the Varroa Detector model.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory containing images')
    parser.add_argument('--labels_dir', type=str, required=True, help='Path to the training data directory containing labels')
    args = parser.parse_args()
    
    detector = VarroaDetector()

    print("Starting model training...")

    history = detector.train(
        videos_dir=args.train_dir,
        labels_dir=args.labels_dir   ,
        epochs=20,
        batch_size=32
    )
    
    print("Training history:", history)
    plot_training_history(history)
    
    detector.save_model('varroa_detector_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()