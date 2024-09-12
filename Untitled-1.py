import cv2
import numpy as np

# Load pre-trained face detector and mask detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = # Load your trained mask detection model here

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0  # Normalize
        
        mask, withoutMask = model.predict(face_roi)
        
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            # Trigger alert mechanism here
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import serial
import time

# Initialize serial connection with Arduino (adjust port and baud rate as needed)
arduino = serial.Serial('COM3', 9600, timeout=0.1)
time.sleep(2)  # Allow time for Arduino to initialize

while True:
    try:
        data = arduino.readline().decode().strip()
        if data:
            temperature = float(data)
            print(f'Temperature: {temperature} °C')
            # Add logic to check temperature against thresholds and trigger alerts
    except serial.SerialException:
        print("Error reading serial data")
        break

arduino.close()

