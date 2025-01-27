from varroa_detector import VarroaDetector
import argparse

def main():
    # Initialize detector and load trained model
    detector = VarroaDetector()
    detector.load_model('varroa_detector_model.h5')

    parser = argparse.ArgumentParser(description='Process images for varroa detection.')
    parser.add_argument('image_dir', type=str, help='Directory containing images to process')
    args = parser.parse_args()


    # Start video stream processing
    print("Starting PNG processing...")
    detector.process_images(args.image_dir)

if __name__ == '__main__':
    main()