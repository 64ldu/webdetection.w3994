import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class FaceAI:
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = mp_face_detection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.expression_history = deque(maxlen=10)
        self.last_expression = "Neutral"
        
    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append((bbox, detection.score[0]))
        
        return faces
    
    def analyze_expression(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            expression = self._classify_expression(landmarks)
            self.expression_history.append(expression)
            
            if len(self.expression_history) >= 5:
                most_common = max(set(self.expression_history), 
                                 key=self.expression_history.count)
                self.last_expression = most_common
            
            return expression, landmarks
        
        return "No face detected", None
    
    def _classify_expression(self, landmarks):
        try:
            left_eye = [landmarks.landmark[33], landmarks.landmark[7], landmarks.landmark[163], landmarks.landmark[144]]
            right_eye = [landmarks.landmark[362], landmarks.landmark[398], landmarks.landmark[384], landmarks.landmark[263]]
            mouth = [landmarks.landmark[13], landmarks.landmark[14], landmarks.landmark[78], landmarks.landmark[308]
                     , landmarks.landmark[61], landmarks.landmark[291], landmarks.landmark[0], landmarks.landmark[17]]
            
            left_eye_height = abs(left_eye[1].y - left_eye[2].y)
            right_eye_height = abs(right_eye[1].y - right_eye[2].y)
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            mouth_width = abs(mouth[4].x - mouth[5].x)
            mouth_height = abs(mouth[0].y - mouth[1].y)
            mouth_openness = mouth_height / mouth_width if mouth_width > 0 else 0
            
            left_eyebrow = landmarks.landmark[70].y
            right_eyebrow = landmarks.landmark[300].y
            left_eye_top = landmarks.landmark[159].y
            right_eye_top = landmarks.landmark[386].y
            
            left_brow_raise = left_eyebrow - left_eye_top
            right_brow_raise = right_eyebrow - right_eye_top
            avg_brow_raise = (left_brow_raise + right_brow_raise) / 2
            
            if avg_brow_raise > 0.02:
                return "Surprised"
            elif mouth_openness > 0.3:
                return "Happy"
            elif avg_brow_raise < -0.01:
                return "Sad"
            elif avg_eye_height < 0.01:
                return "Sleepy"
            else:
                return "Neutral"
                
        except:
            return "Neutral"
    
    def read_lips(self, landmarks):
        if not landmarks:
            return "No lips detected"
        
        try:
            mouth_points = []
            for i in [13, 14, 78, 308, 61, 291, 0, 17, 84, 17, 314, 405]:
                mouth_points.append(landmarks.landmark[i])
            
            mouth_openness = abs(mouth_points[0].y - mouth_points[1].y)
            
            if mouth_openness > 0.02:
                return "Speaking"
            else:
                return "Silent"
                
        except:
            return "Unknown"
    
    def draw_landmarks(self, frame, landmarks):
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Face AI started. Press 'q' to quit, 'l' to toggle lip reading")
        
        lip_reading_enabled = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            faces = self.detect_faces(frame)
            expression, landmarks = self.analyze_expression(frame)
            
            for bbox, confidence in faces:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {confidence:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Expression: {self.last_expression}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if lip_reading_enabled:
                lip_status = self.read_lips(landmarks)
                cv2.putText(frame, f"Lip Reading: {lip_status}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            frame = self.draw_landmarks(frame, landmarks)
            
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
