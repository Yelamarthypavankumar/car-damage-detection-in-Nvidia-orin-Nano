# car-damage-detection-in-Nvidia-orin-Nano
Car damage detection using YOLOX and deployed on Nvidia Orin Nano development Kit
This project implements a real-time car damage detection system using YOLOX on the NVIDIA Jetson Orin Development Kit.
YOLOX is a high-performance object detection model known for its accuracy and speed.
The Jetson Orin enables efficient edge deployment with GPU-accelerated inference.
This system detects vehicle damage from images or video streams with low latency.
It is suitable for use in automated inspections, insurance assessments, and smart garages.


To run the inference follow the below steps:
        1.clone this repo and extract.
        2.create the env with python 3.10.(conda create -n car_damage python=3.10).
        3.Install the dependencies (pip install -r requirements.txt).
        4.streamlit run detect_app.py

Below screenshot is the streamlit GUI:
![image](https://github.com/user-attachments/assets/d5b69b64-d653-44a0-bf10-f0cdbcddc510)
