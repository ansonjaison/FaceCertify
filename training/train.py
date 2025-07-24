import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import face_recognition
import h5py
import joblib
import json

def load_face_dataset(dataset_path):
    """
    Load face images from a directory structure where:
    - dataset_path is the main folder
    - Each subfolder is named after a student
    - Each subfolder contains face images of that student
    
    Returns face encodings and corresponding labels
    """
    print(f"Loading dataset from {dataset_path}...")
    face_encodings = []
    face_labels = []
    
    # Walk through all directories
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing images for: {person_name}")
        
        # Process each image in the person's directory
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            # Skip if not an image file
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Find face locations in the image
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    print(f"  No face found in {image_path}")
                    continue
                    
                if len(face_locations) > 1:
                    print(f"  Multiple faces found in {image_path}, using the first one")
                
                # Get face encoding for the first face found
                encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                
                # Add encoding and label to our lists
                face_encodings.append(encoding)
                face_labels.append(person_name)
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
    
    print(f"Loaded {len(face_encodings)} face images across {len(set(face_labels))} students")
    return np.array(face_encodings), face_labels

def train_face_recognition_model(face_encodings, face_labels):
    """
    Train a face recognition model using face encodings and labels
    """
    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(face_labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        face_encodings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Train SVM classifier
    print("Training SVM classifier...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, label_encoder, X_test, y_test

def save_model_h5(model, label_encoder, model_path, label_path):
    """
    Save the trained model to an H5 file and labels to a TXT file
    """
    # Save the model using joblib (can be loaded with most ML frameworks)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the label encoder classes to a text file
    with open(label_path, 'w') as f:
        for label in label_encoder.classes_:
            f.write(f"{label}\n")
    
    print(f"Labels saved to {label_path}")

if __name__ == "__main__":
    # Set the path to your dataset
    DATASET_PATH =r"D:/FaceCertify/Project Final/images"  # Change this to your dataset path
    
    # Load the dataset
    face_encodings, face_labels = load_face_dataset(DATASET_PATH)
    
    if len(face_encodings) == 0:
        print("No face encodings found. Check your dataset.")
        exit()
    
    # Train the face recognition model
    model, label_encoder, X_test, y_test = train_face_recognition_model(face_encodings, face_labels)
    
    # Save the model and label encoder
    save_model_h5(model, label_encoder, "face_recognition_model.h5", "face_labels.txt")
    
    print("Face recognition training completed!")