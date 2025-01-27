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
    """
    Cette class implemente un modèle pour detecter des objects
    """

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # Feature extraction layers
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layers
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
        
        for subdir in os.listdir(videos_dir):
            subdir_path = os.path.join(videos_dir, subdir)
            if os.path.isdir(subdir_path):
                for image_file in os.listdir(subdir_path):
                    if image_file.endswith('.png'):
                        # Charger l'image
                        image_path = os.path.join(subdir_path, image_file)
                        img = load_img(image_path, target_size=self.input_shape[:2])
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        
                        # Charger le label correspondant
                        label_file = os.path.join(labels_dir, subdir, 
                                                os.path.splitext(image_file)[0] + '.txt')
                        
                        if os.path.exists(label_file):
                            with open(label_file, 'r') as f:
                                content = f.readline().strip()
                                if content == '0':
                                    classifications.append(0)
                                    bboxes.append([0, 0, 0, 0])  # pas de varroa
                                else:
                                    classifications.append(1)
                                    coords = list(map(float, content.split()[1:]))
                                    # Normaliser les coordonnées
                                    coords = [
                                        coords[0] / self.input_shape[1],
                                        coords[1] / self.input_shape[0],
                                        coords[2] / self.input_shape[1],
                                        coords[3] / self.input_shape[0]
                                    ]
                                    bboxes.append(coords)
        
        return np.array(images), np.array(classifications), np.array(bboxes)
    
    def train(self, train_dir, labels_dir, epochs=20, batch_size=32):
        """
        train_dir est le repertoire des données d'entrainement

        labels_dir est le repertoire des fichiers texte contenant les caracteristiques des objects detectables
            e.g.  1 47 144 79 176 ; il y a un object dans le rectangle (47, 144, 79, 176)

        epochs est le nombre de passes à effectuer; une "epoch" représente 
            un passage complet de l'ensemble des données d'entraînement
            Les epochs ont une influence sur les risques de sous ou de sur apprentissage
            Il faut surveiller la courbe de validation loss :
                Tant qu'elle continue de descendre, on peut augmenter encore le nombre d'epochs
                Par contre si elle a tendance à remonter, on peut arrêter le process plus tôt (early stopping)
            Le nombre optimal d'epochs dépend de :
                La quantité de données d'entraînement
                La complexité du problème
                La qualité des images pour l'entrainement
                La variabilité des objects à dectecter dans les images d'entrainement

        batch size est le nombre d'images traitées simultanément pendant l'entraînement.
            Le modèle met à jour ses poids après chaque lot
            Le batch size est lié à la mémoire disponible; en cas d'erreur mémoire, réduisez-le.


        Early Stopping (arrêt précoce) arrête l'entraînement si la performance ne s'améliore pas et
        evite le surapprentissage
        ReduceLROnPlateau réduit le learning rate quand la performance stagne. 
        Il permet un apprentissage plus finet aide à trouver un meilleur minimum local
        """

        X, y_class, y_bbox = self.load_and_preprocess_data(train_dir, labels_dir)
        
        # Diviser les données en ensembles d'entraînement et de validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
        y_bbox_train, y_bbox_val = y_bbox[:split_idx], y_bbox[split_idx:]
        
        # Définir les callbacks
        early_stopping = EarlyStopping(
            monitor='val_classification_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_classification_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        
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
        # Prétraiter l'image
        processed_image = self._preprocess_frame(image)
        
        # Faire la prédiction
        class_pred, bbox_pred = self.model.predict(processed_image)
        
        return class_pred[0], bbox_pred[0]
    
    def process_images(self, image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                frame = cv2.imread(image_path)
                
                # Faire la prédiction
                class_pred, bbox_pred = self.process_image(frame)
                
                # Dessiner les résultats
                if class_pred > 0.5:
                    # Convertir les coordonnées normalisées en pixels
                    height, width, _ = frame.shape
                    x1 = int(bbox_pred[0] * width)
                    y1 = int(bbox_pred[1] * height)
                    x2 = int(bbox_pred[2] * width)
                    y2 = int(bbox_pred[3] * height)
                    
                    # Dessiner la boîte englobante
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Afficher la probabilité
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