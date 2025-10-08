import cv2
import os
import pickle
import numpy as np
from pathlib import Path

# Create directories for data storage
DATA_DIR = "face_data"
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Data files
FACES_DATA_FILE = os.path.join(DATA_DIR, "faces_data.pkl")
NAMES_FILE = os.path.join(DATA_DIR, "names.pkl")

def load_known_faces():
    """Load known faces and names from disk"""
    if os.path.exists(FACES_DATA_FILE) and os.path.exists(NAMES_FILE):
        with open(FACES_DATA_FILE, 'rb') as f:
            faces = pickle.load(f)
        with open(NAMES_FILE, 'rb') as f:
            names = pickle.load(f)
        return faces, names
    return [], []

def save_known_faces(faces, names):
    """Save known faces and names to disk"""
    with open(FACES_DATA_FILE, 'wb') as f:
        pickle.dump(faces, f)
    with open(NAMES_FILE, 'wb') as f:
        pickle.dump(names, f)

def preprocess_face(face_img):
    """Preprocess face image for recognition"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (200, 200))

def add_new_person(face_img, name):
    """Add new person to database"""
    faces, names = load_known_faces()
    processed_face = preprocess_face(face_img)
    faces.append(processed_face)
    names.append(name)
    save_known_faces(faces, names)
    print(f"Added new person: {name}")
    return faces, names

def train_recognizer():
    """Train the face recognizer with known faces"""
    faces, names = load_known_faces()
    if faces:
        labels = list(range(len(faces)))
        face_recognizer.train(faces, np.array(labels))
        return {i: name for i, name in enumerate(names)}
    return {}

def detect_and_recognize_faces(frame, label_map):
    """Detect and recognize faces in a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        
        if label_map:  # If we have trained data
            label, confidence = face_recognizer.predict(processed_face)
            if confidence < 80:  # Threshold for recognition (lower = more confident)
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0)  # Green for recognized
            else:
                name = "New Person"
                color = (0, 0, 255)  # Red for new person
        else:
            name = "New Person"
            color = (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Handle new person registration
        if name == "New Person":
            cv2.putText(frame, "Press 'n' to register", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    return frame, faces

def register_new_person(frame, faces):
    """Register a new person when 'n' is pressed"""
    if len(faces) > 0:
        # Use the first detected face
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        
        # Get name from user
        name = input("Enter person's name: ").strip()
        if name:
            add_new_person(face_img, name)
            # Retrain recognizer
            return train_recognizer()
    return None

def process_image(image_path):
    """Process a static image for face recognition"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    label_map = train_recognizer()
    frame, _ = detect_and_recognize_faces(frame, label_map)
    
    cv2.imshow('Image Recognition', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Face Recognition System")
    print("Options:")
    print("1. Live detection (press '1')")
    print("2. Image detection (press '2')")
    print("Press 'q' to quit during live detection")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == '1':
        # Live detection mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        label_map = train_recognizer()
        print("Starting live detection... Press 'n' to register new person, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame, faces = detect_and_recognize_faces(frame, label_map)
            cv2.imshow('Live Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and len(faces) > 0:
                new_label_map = register_new_person(frame, faces)
                if new_label_map is not None:
                    label_map = new_label_map
                    print("Recognizer updated!")
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif choice == '2':
        # Image detection mode
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print("File not found!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()