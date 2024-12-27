# Fall Detection System with YOLO and Telegram Alerts

## Description
This project is a Python-based fall detection system that uses the YOLO object detection model to monitor live video streams and detect falls in real-time. If a fall is detected, an emergency alert is sent via a Telegram bot.

## Features
- Real-time object detection using YOLO.
- Fall detection algorithm based on bounding box dimensions.
- Sends alerts via a Telegram bot when a fall is detected.
- Displays processed video feed with detections and a custom logo.

## Requirements
To run this project, you need:
- Python 3.x
- Required libraries (mentioned in `requirements.txt`):
  - `opencv-python`
  - `cvzone`
  - `ultralytics`
  - `requests`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/u0sf/Fall-Detection-System.git
