import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class VarroaDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        input_layer = Input(shape=self.input_shape)
        
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)
        bbox_output = Dense(4, activation='linear', name='bbox')(x)
        
        model = Model(inputs=input_layer, outputs=[classification_output, bbox_output])
        
        model.compile(
            optimizer='adam',
            loss={
                'classification': 'binary_crossentropy',
                'bbox': 'mean_squared_error'
            },
            loss_weights={
                'classification': 1.0,
                'bbox': 1.0
            },
            metrics={
                'classification': 'accuracy'
            }
        )
        
        return model
    
    def load_and_preprocess_data(self, videos_dir, labels_dir):
        images = []
        classifications = []
        bboxes = []
        
        print("Loading and preprocessing data...")
        
        for subdir in os.listdir(videos_dir):
            subdir_path = os.path.join(videos_dir, subdir)
            if os.path.isdir(subdir_path):
                for image_file in os.listdir(subdir_path):
                    if image_file.endswith('.png'):
                        try:
                            # Charger l'image
                            image_path = os.path.join(subdir_path, image_file)
                            img = load_img(image_path, target_size=self.input_shape[:2])
                            img_array = img_to_array(img) / 255.0
                            
                            # Charger le label correspondant
                            label_file = os.path.join(labels_dir, subdir, 
                                                    os.path.splitext(image_file)[0] + '.txt')
                            
                            if os.path.exists(label_file):
                                with open(label_file, 'r') as f:
                                    lines = f.readlines()
                                    if len(lines) < 1:  # Check for at least one line
                                        print(f"Warning: Invalid format in {label_file}")
                                        continue
                                    
                                    class_label = lines[0].strip()
                                    
                                    if class_label == '0':
                                        images.append(img_array)
                                        classifications.append(0)
                                        bboxes.append([0, 0, 0, 0])  # No varroa
                                    elif len(lines) >= 2:
                                        coords_line = lines[1].strip()
                                        if not coords_line:
                                            print(f"Warning: Empty coordinates line in {label_file}")
                                            continue
                                        
                                        try:
                                            coords = list(map(float, coords_line.split()))
                                            if len(coords) != 4:  # Ensure there are exactly four coordinates
                                                print(f"Warning: Invalid format in {label_file}")
                                                continue
                                            
                                            # Normalize the coordinates
                                            normalized_coords = [
                                                coords[0] / self.input_shape[1],
                                                coords[1] / self.input_shape[0],
                                                coords[2] / self.input_shape[1],
                                                coords[3] / self.input_shape[0]
                                            ]
                                            images.append(img_array)
                                            classifications.append(1)
                                            bboxes.append(normalized_coords)
                                        except (ValueError, IndexError) as e:
                                            print(f"Warning: Error parsing coordinates in {label_file}: {e}")
                                            continue
                                    else:
                                        print(f"Warning: Missing coordinates for class label 1 in {label_file}")
                                
                        except Exception as e:
                            print(f"Error processing {image_file}: {e}")
                            continue
        
        if not images:
            raise ValueError("No valid data was loaded. Check your data directories and file formats.")
            
        print(f"Loaded {len(images)} valid images")
        return np.array(images), np.array(classifications), np.array(bboxes)
    
    def train(self, videos_dir, labels_dir, epochs=20, batch_size=32):
        # Charger et prétraiter les données
        X, y_class, y_bbox = self.load_and_preprocess_data(videos_dir, labels_dir)
        
        # Diviser les données en ensembles d'entraînement et de validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
        y_bbox_train, y_bbox_val = y_bbox[:split_idx], y_bbox[split_idx:]
        
        # Définir les callbacks; on veut minimiser le loss
        early_stopping = EarlyStopping(
            monitor='val_classification_loss',
            patience=5,
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_classification_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        
        print(f"Starting training with {len(X_train)} training samples and {len(X_val)} validation samples")
        
        # Entraîner le modèle
        history = self.model.fit(
            X_train,
            {
                'classification': y_class_train,
                'bbox': y_bbox_train
            },
            validation_data=(
                X_val,
                {
                    'classification': y_class_val,
                    'bbox': y_bbox_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history
    
    def process_image(self, image):
        processed_image = self._preprocess_frame(image)
        class_pred, bbox_pred = self.model.predict(processed_image)
        return class_pred[0], bbox_pred[0]
    
    def process_images(self, image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                frame = cv2.imread(image_path)
                
                class_pred, bbox_pred = self.process_image(frame)
                
                if class_pred > 0.5:
                    height, width, _ = frame.shape
                    x1 = int(bbox_pred[0] * width)
                    y1 = int(bbox_pred[1] * height)
                    x2 = int(bbox_pred[2] * width)
                    y2 = int(bbox_pred[3] * height)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    self._draw_results(frame, class_pred)
                
                cv2.imshow('Varroa Detection', frame)
                cv2.waitKey(0)
    
    def _preprocess_frame(self, frame):
        resized = cv2.resize(frame, self.input_shape[:2])
        normalized = resized / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def _draw_results(self, frame, prediction):
        text = f"Varroa Probability: {prediction:.2f}"
        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    def save_model(self, path):
        self.model.save(path)
        
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)