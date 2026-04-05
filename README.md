# Waste Detection System
This repository contains a machine learning-based application designed to detect and classify waste. 
It utilizes a PyTorch model served via a Flask web application.
# Features
Deep Learning Backend: Powered by a pre-trained PyTorch model (waste_model.pth).
Web API: A Flask server (server.py) to handle image uploads and return classification results.
Cloud Ready: Includes a Procfile and requirements.txt for easy deployment to platforms like Heroku.
# File                                                 # Description
server.py                                                The main Flask application script that serves the model.
waste_model.pth                                          The saved PyTorch model weights for waste classification.
requirements.txt                                         List of Python dependencies (Flask, torch, torchvision, etc.).
Procfile                                                 Configuration for production process managers (used for deployment).


