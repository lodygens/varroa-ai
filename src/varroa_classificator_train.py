import os
import argparse
from varroa_classificator import VarroaClassificator

def main():
    
    parser = argparse.ArgumentParser(description='Train the Varroa Detector model.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--validation_dir', type=str, required=True, help='Path to the validation data directory')
    args = parser.parse_args()
    
    detector = VarroaClassificator()
    
    train_dir = args.train_dir
    validation_dir = args.validation_dir

    print("Starting model training...")
    history = detector.train(
        train_dir=train_dir,
        validation_dir=validation_dir,
        epochs=20,
        batch_size=32
    )
    
    print("Training history:", history)

    detector.save_model('varroa_detector_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()