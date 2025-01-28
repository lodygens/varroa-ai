from ultralytics import YOLO
import cv2
import os

def process_directory(input_dir, output_dir, model_path):
    """
    Détecte les varroas dans toutes les images d'un répertoire
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le modèle
    model = YOLO(model_path)
    
    # Parcourir les images
    image_extensions = ('.png', '.jpg', '.jpeg')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"detected_{filename}")
            
            print(f"Processing {filename}...")
            
            # Faire la prédiction
            results = model.predict(input_path, conf=0.25)
            
            # Sauvegarder l'image avec les détections
            for result in results:
                im_array = result.plot()  # Plot les résultats sur l'image
                cv2.imwrite(output_path, im_array)

def main():
    if len(sys.argv) != 4:
        print("Usage: python detect_yolo.py <input_dir> <output_dir> <model_path>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return
    
    process_directory(input_dir, output_dir, model_path)
    print(f"\nProcessing completed. Results are in: {output_dir}")

if __name__ == "__main__":
    main()