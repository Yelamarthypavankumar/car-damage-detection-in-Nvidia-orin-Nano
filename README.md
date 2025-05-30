# car-damage-detection-in-Nvidia-orin-Nano
Car damage detection using YOLOX and deployed on Nvidia Orin Nano development Kit
This project implements a real-time car damage detection system using YOLOX on the NVIDIA Jetson Orin Development Kit.
YOLOX is a high-performance object detection model known for its accuracy and speed.
The Jetson Orin enables efficient edge deployment with GPU-accelerated inference.
This system detects vehicle damage from images or video streams with low latency.
It is suitable for use in automated inspections, insurance assessments, and smart garages.
## 🧠 About the Project

This project provides a lightweight, high-performance app for detecting car damages such as:

- Dent
- Scratch
- Crack
- Glass Shatter
- Lamp Broken
- Tire Flat

All processing is optimized for edge deployment using **TensorRT** on **Jetson Orin Nano**.
---

## 🎯 Key Features

- 🚀 **Real-time Inference** powered by TensorRT
- 📸 **Image Upload and Selection** via Streamlit
- 🧾 **Defect Summary Output** with counts per class
- 💡 **GPU-accelerated** using CUDA and pyCUDA
- 🧩 Uses YOLOX for accurate object detection

---


🔧 Setup & Inference
Follow these steps to run the app:
1. Clone the Repository
   git clone https://github.com/Yelamarthypavankumar/car-damage-detection-in-Nvidia-orin-Nano.git

   1. cd car-damage-detection-in-Nvidia-orin-Nano
3. Create a Virtual Environment with Python 3.10
   conda create -n car_damage python=3.10
4. Activate the Environment
   conda activate car_damage
5. Install Dependencies
   pip install -r requirements.txt
6. Run the Streamlit App
   streamlit run detect_app.py




Below screenshot is the streamlit GUI:
![image](https://github.com/user-attachments/assets/d5b69b64-d653-44a0-bf10-f0cdbcddc510)


References:https://github.com/Megvii-BaseDetection/YOLOX
