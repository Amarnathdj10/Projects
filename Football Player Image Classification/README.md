A football player image classification web application built using classical machine learning techniques and deployed on Render.

Live Demo

🔗 https://football-player-image-classification.onrender.com

This project identifies football players from uploaded images using a custom feature-engineered ML pipeline.
The system performs:

-Face detection using Haar Cascades
-Feature extraction using Wavelet Transform
-RGB + Wavelet feature stacking
-Dimensionality reduction using PCA
-Classification using Logistic Regression

Tech Stack

-Python
-Flask
-Scikit-learn
-OpenCV
-PyWavelets
-HTML / CSS / JavaScript
-Gunicorn
-Render (Deployment

Architecture

-User uploads image (drag-and-drop UI)
-Image is preprocessed (face detection + resizing)
-Feature engineering (RGB + wavelet features)
-PCA transformation
-Logistic Regression prediction
-Result displayed on frontend

Future Improvements

-Replace classical ML with CNN (Deep Learning)
-Add confidence score display
-Improve face detection robustness
-Dockerize the application
