from varroa_classificator import VarroaClassificator

def main():
    # Initialize detector and load trained model
    detector = VarroaClassificator()
    detector.load_model('varroa_detector_model.h5')
    
    # Start video stream processing
    print("Starting video stream processing...")
    print("Press 'q' to quit")
    detector.process_video_stream()

if __name__ == '__main__':
    main()