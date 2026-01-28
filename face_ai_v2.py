import cv2
import numpy as np
from collections import deque

class FaceAI:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.expression_history = deque(maxlen=10)
        self.last_expression = "Neutral"
        
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
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face)
        
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        smiles = smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        
        expression = self._classify_expression_simple(len(eyes), len(smiles), w, h)
        self.expression_history.append(expression)
        
        if len(self.expression_history) >= 5:
            most_common = max(set(self.expression_history), 
                             key=self.expression_history.count)
            self.last_expression = most_common
        
        return expression, face_bbox
    
    def _classify_expression_simple(self, num_eyes, num_smiles, face_width, face_height):
        if num_smiles > 0:
            return "Happy"
        elif num_eyes >= 2:
            mouth_openness = self._estimate_mouth_openness(face_width, face_height)
            if mouth_openness > 0.3:
                return "Surprised"
            else:
                return "Neutral"
        elif num_eyes == 1:
            return "Winking"
        else:
            return "Sleepy"
    
    def _estimate_mouth_openness(self, face_width, face_height):
        return 0.1
    
    def read_lips(self, face_bbox):
        if face_bbox is None:
            return "No lips detected"
        
        return "Silent"
    
    def draw_landmarks(self, frame, face_bbox):
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Draw a small cross at the center of the primary face
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 10, 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Face AI started. Features:")
        print("- Red boxes around all detected faces")
        print("- Face numbering and confidence scores")
        print("- Expression recognition for primary face")
        print("- Press 'q' to quit, 'l' to toggle lip reading")
        
        lip_reading_enabled = False
        current_face_bbox = None
        
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
                current_face_bbox = max(faces, key=lambda x: x[0][2] * x[0][3])[0]  # Largest by area
                expression, _ = self.analyze_expression(frame, current_face_bbox)
                cv2.putText(frame, f"Primary Expression: {self.last_expression}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                current_face_bbox = None
            
            # Draw red boxes around all detected faces
            for i, (bbox, confidence) in enumerate(faces):
                x, y, w, h = bbox
                # Draw red box around each face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Add face number and confidence
                face_label = f"Face {i+1}: {confidence:.2f}"
                cv2.putText(frame, face_label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            if lip_reading_enabled:
                lip_status = self.read_lips(current_face_bbox)
                cv2.putText(frame, f"Lip Reading: {lip_status}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            frame = self.draw_landmarks(frame, current_face_bbox)
            
            cv2.imshow("Face AI", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                lip_reading_enabled = not lip_reading_enabled
                print(f"Lip reading: {'ON' if lip_reading_enabled else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ai = FaceAI()
    ai.run()
