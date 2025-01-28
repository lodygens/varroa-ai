from ultralytics import YOLO
import os
import shutil
from PIL import Image
import yaml
import argparse

def convert_labels_to_yolo(input_videos_dir, input_labels_dir, output_dir):
    """
    Convertit les labels au format YOLO
    Format YOLO: <class> <x_center> <y_center> <width> <height>
    """
    # Créer les répertoires
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    for subdir in os.listdir(input_videos_dir):
        subdir_path = os.path.join(input_videos_dir, subdir)
        if os.path.isdir(subdir_path):
            for image_file in os.listdir(subdir_path):
                if image_file.endswith('.png'):
                    # Chemin des fichiers
                    image_path = os.path.join(subdir_path, image_file)
                    label_path = os.path.join(input_labels_dir, subdir, 
                                            os.path.splitext(image_file)[0] + '.txt')
                    
                    if os.path.exists(label_path):
                        # Lire l'image pour obtenir ses dimensions
                        with Image.open(image_path) as img:
                            img_width, img_height = img.size
                        
                        # Lire le fichier label
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            
                        if len(lines) > 0:
                            class_label = int(lines[0].strip())
                            
                            # Copier l'image
                            new_image_path = os.path.join(output_dir, 'images', f"{subdir}_{image_file}")
                            shutil.copy2(image_path, new_image_path)
                            
                            # Créer le nouveau fichier label
                            new_label_path = os.path.join(output_dir, 'labels', 
                                                        f"{subdir}_{os.path.splitext(image_file)[0]}.txt")
                            
                            with open(new_label_path, 'w') as f:
                                if class_label == 1 and len(lines) > 1:
                                    for line in lines[1:]:
                                        try:
                                            x1, y1, x2, y2 = map(float, line.strip().split())
                                            
                                            # Convertir en format YOLO (x_center, y_center, width, height)
                                            x_center = (x1 + x2) / (2 * img_width)
                                            y_center = (y1 + y2) / (2 * img_height)
                                            width = (x2 - x1) / img_width
                                            height = (y2 - y1) / img_height
                                            
                                            # Écrire au format YOLO
                                            f.write(f"0 {x_center} {y_center} {width} {height}\n")
                                        except ValueError:
                                            continue

def create_yaml(output_dir):
    """Crée le fichier de configuration YAML pour YOLOv8"""
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images',
        'val': 'images',
        'names': {
            0: 'varroa'
        }
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    
    return yaml_path

def train_yolo(data_yaml_path):
    """Entraîne le modèle YOLOv8"""
    # Charger un modèle YOLOv8 pré-entraîné
    model = YOLO('yolov8n.pt')
    
    # Entraîner le modèle
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        save=True
    )
    
    return results

def main():
    # Définir les chemins
    parser = argparse.ArgumentParser(description='Train the Varroa Detector model.')
    parser.add_argument('--videos_dir', type=str, required=True, help='Path to the training data directory containing images')
    parser.add_argument('--labels_dir', type=str, required=True, help='Path to the training data directory containing labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    input_videos_dir = args.videos_dir
    input_labels_dir = args.labels_dir
    output_dir = args.output_dir
    
    # Convertir les données
    print("Converting data to YOLO format...")
    convert_labels_to_yolo(input_videos_dir, input_labels_dir, output_dir)
    
    # Créer le fichier YAML
    print("Creating YAML configuration...")
    yaml_path = create_yaml(output_dir)
    
    # Entraîner le modèle
    print("Training YOLO model...")
    train_yolo(yaml_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main()