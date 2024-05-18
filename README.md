To begin:
1. Create a virtual environment: python -m venv venv
2. Activate environment: venv/Scripts/activate
3. Install requirements: pip install -r requirements.txt

To train model to recognize you:
1. Install CMake for your desired OS: https://cmake.org/download/
2. Create a folder with your name in training and add photos of yourself
3. Train the model: python train.py

To run detection on the camera: python camera.py

To run validation: python validation.py

Snipes can be seen in the recognized_faces folder