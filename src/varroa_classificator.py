import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class VarroaClassificator:

    """
    Ce modèle est un classificateur binaire simple (présence/absence de varroa)
    """

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification: Varroa present or not
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, validation_dir, epochs=10, batch_size=32):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary'
        )
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        return history
    
    def process_images(self, image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                frame = cv2.imread(image_path)
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Make prediction
                prediction = self.model.predict(processed_frame)
                
                # Draw results on frame
                if prediction[0] > 0.5:
                    self._draw_results(frame, prediction[0])
                
                    # Draw a rectangle around the detected Varroa
                    height, width, _ = frame.shape
                    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 2)
                                    # Display the frame
                    cv2.imshow('Varroa Detection', frame)
                    cv2.waitKey(0)  # Wait for a key press to move to the next image
   

    def process_video_stream(self, source=0):
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Make prediction
            prediction = self.model.predict(processed_frame)
            
            # Draw results on frame
            self._draw_results(frame, prediction[0])
            
            # Display the frame
            cv2.imshow('Varroa Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def _preprocess_frame(self, frame):
        # Resize frame to match model input shape
        resized = cv2.resize(frame, self.input_shape[:2])
        # Normalize pixel values
        normalized = resized / 255.0
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
#    def _draw_results(self, frame, prediction):
#        text = f"Varroa Probability: {prediction:.2f}"
#        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
#        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def _draw_results(self, frame, prediction):
        # Extract the scalar value from the prediction array
        probability = prediction[0] if isinstance(prediction, np.ndarray) else prediction
        text = f"Varroa Probability: {probability:.2f}"
        color = (0, 0, 255) if probability > 0.5 else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    def save_model(self, path):
        self.model.save(path)
        
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)