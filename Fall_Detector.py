
import cv2
import cvzone
import math
import requests
from ultralytics import YOLO
import time

# Telegram Bot Settings
TELEGRAM_TOKEN = 'Your_Telegram_Bot_Token'  # Replace with your bot token
CHAT_ID = 'Your_Chat_Id'  # Replace with your chat ID

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    requests.post(url, data=payload)

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Load class names from the COCO dataset
classnames = []
with open('coco.txt', 'r') as f:
    classnames = f.read().splitlines()

# Load the logo (resize only once)
logo = cv2.imread('Your_Logo.Png')
logo = cv2.resize(logo, (100, 100))
logo_height, logo_width, _ = logo.shape

# Skip frames setting
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for processing
    frame = cv2.resize(frame, (960, 720))
    frame_count += 1

    # Skip frames
    if frame_count % frame_skip != 0:
        continue

    # Add the logo once
    frame[10:10 + logo_height, -10 - logo_width:-10] = logo

    # Perform object detection
    results = model(frame, stream=True)  # Stream mode for faster processing

    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_detect = classnames[int(box.cls[0])]
            conf = math.ceil(confidence * 100)

            # Calculate height and width
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            # Check if the detected object is a person
            if conf > 5 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            # Fall detection alert
            if threshold < -20:
                cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 50], thickness=2, scale=2)
                send_telegram_message('🚨 Emergency Alert: A person has fallen!')

    # Display the frame
    cv2.imshow('NMU SYSTEM', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
