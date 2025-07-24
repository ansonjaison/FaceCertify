import cv2
import numpy as np
import face_recognition
import joblib
import os

def load_model_h5(model_path, label_path):
    """
    Load the trained face recognition model from H5 file and labels from TXT file
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Load the labels
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    return model, labels

def recognize_face(image_path, model, labels, confidence_threshold=0.7):
    """
    Recognize faces in an image using the trained model
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        return image, []
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    predictions = []
    
    # Recognize each face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Predict with the model
        face_encoding = face_encoding.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(face_encoding)[0]
        
        # Find the best match
        best_match_idx = np.argmax(probabilities)
        confidence = probabilities[best_match_idx]
        
        # If confidence is high enough, use the predicted name
        if confidence >= confidence_threshold:
            name = labels[best_match_idx]
        else:
            name = "Unknown"
        
        predictions.append((name, face_location, confidence))
    
    # Draw boxes and labels on the image
    for name, (top, right, bottom, left), confidence in predictions:
        # Draw box
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label
        label = f"{name} ({confidence:.2f})"
        cv2.putText(rgb_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return rgb_image, predictions

def test_with_image(image_path, model, labels):
    """
    Test the face recognition model with a single image
    """
    result_image, predictions = recognize_face(image_path, model, labels)
    
    if len(predictions) == 0:
        print(f"No faces detected in {image_path}")
    else:
        print(f"Found {len(predictions)} faces in {image_path}:")
        for name, _, confidence in predictions:
            print(f"  - {name} (confidence: {confidence:.2f})")
    
    # Display the result
    cv2.imshow("Face Recognition", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_camera(model, labels):
    """
    Test the face recognition model with webcam feed
    """
    video_capture = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Recognize each face
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Predict with the model
            face_encoding = face_encoding.reshape(1, -1)
            probabilities = model.predict_proba(face_encoding)[0]
            best_match_idx = np.argmax(probabilities)
            confidence = probabilities[best_match_idx]
            
            # If confidence is high enough, use the predicted name
            if confidence >= 0.7:
                name = labels[best_match_idx]
            else:
                name = "Unknown"
            
            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the trained model
    model, labels = load_model_h5("face_recognition_model.h5", "face_labels.txt")
    
    # Choose test mode
    mode = input("Select test mode (1: Image, 2: Camera): ")
    
    if mode == "1":
        image_path = input("Enter image path: ")
        test_with_image(image_path, model, labels)
    elif mode == "2":
        test_with_camera(model, labels)
    else:
        print("Invalid mode selected")