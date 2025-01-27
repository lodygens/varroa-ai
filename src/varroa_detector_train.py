import os
import argparse
from varroa_detector_earlystop import VarroaDetector

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

    detector.save_model('varroa_detector_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()