"""Minimal auth-free upload test."""
import requests
import sys

VIDEO_PATH = r"C:\Users\Asaf\Videos\DevOps For Beginners\Episode 2 - Linux Essential\שיעור 37 -  Resources In Linux System.mp4"
URL = "http://localhost:8000/api/tasks/upload"

print(f"Starting upload for: {VIDEO_PATH}")

try:
    with open(VIDEO_PATH, "rb") as f:
        files = {"file": ("video.mp4", f, "video/mp4")}
        data = {"mode": "whisper_local", "language": "he"}
        
        print("Uploading to server... please wait.")
        response = requests.post(URL, files=files, data=data)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except FileNotFoundError:
    print("Error: Video file not found. Check the path.")
    sys.exit(1)
