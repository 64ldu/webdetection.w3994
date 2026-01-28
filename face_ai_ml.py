import cv2
import numpy as np
from collections import deque
import pickle
import os
from datetime import datetime
import json

class FaceAI:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.expression_history = deque(maxlen=10)
        self.last_expression = "Neutral"
        
        # ML Model components
        self.training_data = []
        self.feature_history = deque(maxlen=100)
        self.expression_labels = ["Happy", "Sad", "Surprised", "Angry", "Neutral", "Sleepy"]
        self.model_trained = False
        
        # Load existing training data if available
        self.load_training_data()
        
    def load_training_data(self):
        """Load existing training data from file"""
        if os.path.exists('face_training_data.pkl'):
            try:
                with open('face_training_data.pkl', 'rb') as f:
                    self.training_data = pickle.load(f)
                print(f"Loaded {len(self.training_data)} training samples")
                self.train_model()
            except Exception as e:
                print(f"Error loading training data: {e}")
    
    def save_training_data(self):
        """Save training data to file"""
        try:
            with open('face_training_data.pkl', 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"Saved {len(self.training_data)} training samples")
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def extract_features(self, face_roi):
        """Extract features from face region for ML training"""
        if face_roi.size == 0:
            return None
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Basic facial features
        eyes = self.eye_cascade.detectMultiScale(gray_face)
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        
        # Geometric features
        height, width = gray_face.shape
        
        # Eye features
        eye_count = len(eyes)
        if len(eyes) > 0:
            avg_eye_y = np.mean([y + h/2 for (x, y, w, h) in eyes])
            eye_spacing = np.std([x + w/2 for (x, y, w, h) in eyes]) if len(eyes) > 1 else 0
        else:
            avg_eye_y = height/2
            eye_spacing = 0
        
        # Mouth features
        smile_count = len(smiles)
        if len(smiles) > 0:
            avg_smile_y = np.mean([y + h/2 for (x, y, w, h) in smiles])
            smile_width = np.mean([w for (x, y, w, h) in smiles])
        else:
            avg_smile_y = height * 0.7
            smile_width = 0
        
        # Intensity features
        mean_intensity = np.mean(gray_face)
        std_intensity = np.std(gray_face)
        
        # Eye-mouth ratio (important for expressions)
        eye_mouth_distance = avg_smile_y - avg_eye_y if len(eyes) > 0 else height * 0.3
        eye_mouth_ratio = eye_mouth_distance / height
        
        # Create feature vector
        features = [
            eye_count,
            smile_count,
            avg_eye_y / height,
            avg_smile_y / height,
            eye_spacing / width,
            smile_width / width,
            mean_intensity / 255,
            std_intensity / 255,
            eye_mouth_ratio
        ]
        
        return np.array(features)
    
    def train_model(self):
        """Train a simple neural network on collected data"""
        if len(self.training_data) < 10:
            return False
        
        try:
            X = np.array([data['features'] for data in self.training_data])
            y = np.array([data['label'] for data in self.training_data])
            
            # Simple distance-based classifier
            self.feature_centroids = {}
            for label in self.expression_labels:
                label_features = X[y == label]
                if len(label_features) > 0:
                    self.feature_centroids[label] = np.mean(label_features, axis=0)
            
            self.model_trained = True
            print(f"Model trained with {len(self.training_data)} samples")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_expression(self, features):
        """Predict expression using trained model"""
        if not self.model_trained or features is None:
            return "Neutral"
        
        try:
            # Find closest centroid
            min_distance = float('inf')
            predicted_label = "Neutral"
            
            for label, centroid in self.feature_centroids.items():
                distance = np.linalg.norm(features - centroid)
                if distance < min_distance:
                    min_distance = distance
                    predicted_label = label
            
            return predicted_label
            
        except Exception as e:
            print(f"Error predicting expression: {e}")
            return "Neutral"
    
    def add_training_sample(self, features, label):
        """Add a training sample with user-provided label"""
        if features is not None:
            self.training_data.append({
                'features': features,
                'label': label,
                'timestamp': datetime.now().isoformat()
            })
            
            # Retrain model every 10 new samples
            if len(self.training_data) % 10 == 0:
                self.train_model()
                self.save_training_data()
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append(((x, y, w, h), 1.0))
        
        return face_list
    
    def analyze_expression(self, frame, face_bbox):
        if face_bbox is None:
            return "No face detected", None
        
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return "No face detected", None
        
        # Extract features for ML
        features = self.extract_features(face_roi)
        
        if features is not None:
            self.feature_history.append(features)
            
            # Use trained model if available
            if self.model_trained:
                expression = self.predict_expression(features)
            else:
                # Fallback to rule-based detection
                expression = self._classify_expression_simple(face_roi)
            
            self.expression_history.append(expression)
            
            if len(self.expression_history) >= 5:
                most_common = max(set(self.expression_history), 
                                 key=self.expression_history.count)
                self.last_expression = most_common
            
            return expression, features
        
        return "No face detected", None
    
    def _classify_expression_simple(self, face_roi):
        """Simple rule-based expression detection"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(gray_face)
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        
        if len(smiles) > 0:
            return "Happy"
        elif len(eyes) >= 2:
            return "Neutral"
        elif len(eyes) == 1:
            return "Winking"
        else:
            return "Sleepy"
    
    def read_lips(self, face_bbox):
        if face_bbox is None:
            return "No lips detected"
        
        return "Silent"
    
    def draw_landmarks(self, frame, face_bbox):
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.drawMarker(frame, (x + w//2, y + h//2), (0, 255, 0), 
                          cv2.MARKER_CROSS, 10, 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Face AI with ML Training started!")
        print("Features:")
        print("- Red boxes around all detected faces")
        print("- AI expression recognition (improves with training)")
        print("- Press number keys 1-6 to train expressions:")
        print("  1=Happy, 2=Sad, 3=Surprised, 4=Angry, 5=Neutral, 6=Sleepy")
        print("- Press 't' to show training statistics")
        print("- Press 'q' to quit, 'l' to toggle lip reading")
        
        lip_reading_enabled = False
        current_face_bbox = None
        current_features = None
        
        # Expression key mapping
        expression_keys = {
            '1': 'Happy',
            '2': 'Sad', 
            '3': 'Surprised',
            '4': 'Angry',
            '5': 'Neutral',
            '6': 'Sleepy'
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            faces = self.detect_faces(frame)
            
            # Display face count and expressions for all detected faces
            if faces:
                cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show expression for primary face (largest one)
                current_face_bbox = max(faces, key=lambda x: x[0][2] * x[0][3])[0]
                expression, current_features = self.analyze_expression(frame, current_face_bbox)
                
                model_status = "ML" if self.model_trained else "Rule-based"
                cv2.putText(frame, f"Expression ({model_status}): {self.last_expression}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                current_face_bbox = None
                current_features = None
            
            # Draw red boxes around all detected faces
            for i, (bbox, confidence) in enumerate(faces):
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                face_label = f"Face {i+1}: {confidence:.2f}"
                cv2.putText(frame, face_label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            if lip_reading_enabled:
                lip_status = self.read_lips(current_face_bbox)
                cv2.putText(frame, f"Lip Reading: {lip_status}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show training status
            cv2.putText(frame, f"Training samples: {len(self.training_data)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            frame = self.draw_landmarks(frame, current_face_bbox)
            
            cv2.imshow("Face AI with ML Training", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                lip_reading_enabled = not lip_reading_enabled
                print(f"Lip reading: {'ON' if lip_reading_enabled else 'OFF'}")
            elif key == ord('t'):
                self.show_training_stats()
            elif key in expression_keys:
                if current_features is not None:
                    label = expression_keys[key]
                    self.add_training_sample(current_features, label)
                    print(f"Added training sample: {label}")
                else:
                    print("No face detected for training")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_training_data()
    
    def show_training_stats(self):
        """Display training statistics"""
        print("\n=== Training Statistics ===")
        print(f"Total samples: {len(self.training_data)}")
        
        if self.training_data:
            label_counts = {}
            for data in self.training_data:
                label = data['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print("Samples per expression:")
            for label in self.expression_labels:
                count = label_counts.get(label, 0)
                print(f"  {label}: {count}")
        
        print(f"Model trained: {self.model_trained}")
        print("========================\n")

if __name__ == "__main__":
    ai = FaceAI()
    ai.run()
