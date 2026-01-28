from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_ai_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class FaceAIWeb:
    def __init__(self):
        # Load pre-trained face detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Try to load a pre-trained emotion recognition model
        self.emotion_model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load FER2013 model if available
        try:
            # We'll use a lightweight model for web demo
            self.load_emotion_model()
        except:
            print("Using rule-based emotion detection")
    
    def load_emotion_model(self):
        """Load a pre-trained emotion recognition model"""
        # For demo purposes, we'll create a simple model
        # In production, you'd load a pre-trained model like:
        # self.emotion_model = load_model('emotion_model.h5')
        pass
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def analyze_expression(self, frame, face_bbox):
        """Analyze facial expression with improved detection"""
        if face_bbox is None:
            return "No face detected", 0.0
        
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return "No face detected", 0.0
        
        # Extract features
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        
        # Enhanced rule-based expression detection
        confidence = 0.5
        
        # Check for smile (happy expression)
        if len(smiles) > 0:
            # Calculate smile intensity
            smile_area = sum([sw * sh for (sx, sy, sw, sh) in smiles])
            face_area = w * h
            smile_ratio = smile_area / face_area
            
            if smile_ratio > 0.02:
                return "Happy", 0.9
            elif smile_ratio > 0.01:
                return "Smiling", 0.8
            else:
                return "Happy", 0.7
        
        # Check for eyes
        if len(eyes) == 0:
            return "Sleepy", 0.8
        elif len(eyes) == 1:
            return "Winking", 0.7
        
        # Analyze eye openness and position for more expressions
        if len(eyes) >= 2:
            eye_areas = [ew * eh for (ex, ey, ew, eh) in eyes]
            avg_eye_area = np.mean(eye_areas) if eye_areas else 0
            
            # Calculate eye-to-face ratio
            face_area = w * h
            eyes_total_area = sum(eye_areas)
            eyes_ratio = eyes_total_area / face_area
            
            if eyes_ratio > 0.05:
                return "Surprised", 0.8
            elif eyes_ratio < 0.02:
                return "Tired", 0.7
            else:
                # Check for frown by analyzing mouth shape
                mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                mouths = mouth_cascade.detectMultiScale(gray_face, 1.5, 15)
                
                if len(mouths) > 0:
                    # Check mouth curvature
                    for (mx, my, mw, mh) in mouths:
                        # Simple curvature analysis
                        if my > h * 0.6:  # Mouth is low on face
                            return "Sad", 0.7
                        elif my < h * 0.4:  # Mouth is high
                            return "Angry", 0.7
        
        return "Neutral", 0.6
    
    def process_frame(self, image_data):
        """Process base64 image data and return results"""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            results = []
            for (x, y, w, h) in faces:
                # Analyze expression for each face
                expression, confidence = self.analyze_expression(frame, (x, y, w, h))
                
                results.append({
                    'bbox': [x, y, w, h],
                    'expression': expression,
                    'confidence': confidence,
                    'face_id': len(results) + 1
                })
            
            return {
                'success': True,
                'faces': results,
                'total_faces': len(faces)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Initialize Face AI
face_ai = FaceAIWeb()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Face AI Web',
        'version': '1.0.0'
    })

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    data = request.json
    if 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'})
    
    result = face_ai.process_frame(data['image'])
    return jsonify(result)

@socketio.on('frame')
def handle_frame(data):
    """Handle real-time frame processing via WebSocket"""
    result = face_ai.process_frame(data['image'])
    emit('detection_result', result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
