import cv2

def test_camera():
    print("Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera with index 0")
        
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("Error: Could not open camera with index 1")
            return False
        else:
            print("Camera 1 opened successfully!")
    else:
        print("Camera 0 opened successfully!")
    
    ret, frame = cap.read()
    if ret:
        print("Successfully captured frame!")
        print(f"Frame dimensions: {frame.shape}")
        
        cv2.imshow("Test Camera", frame)
        print("Press any key to close test window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read frame")
        cap.release()
        return False
    
    cap.release()
    return True

if __name__ == "__main__":
    test_camera()
