This project predicts a football player's match rating before an upcoming fixture using historical performance data and machine learning.
It is a full-stack ML application consisting of:
-Feature engineered dataset
-XGBoost regression model
-Flask backend API
-Interactive frontend interface
-Production deployment on Render

Problem Statement

Can we estimate a player's expected match rating using:
-Recent form (rolling average)
-Historical baseline rating
-Rest days
-Minutes played
-Opponent strength
-Home/Away factor
-Player position

This system aims to simulate pre-match rating prediction similar to professional analytics platforms.

Tech Stack

-Python
-Pandas
-NumPy
-XGBoost
-Scikit-Learn
-Flask
-HTML/CSS/JavaScript
-Render (Deployment)

Feature Engineering

Key engineered features include:
-Rolling 5-match form
-Expanding historical baseline
-Rest days calculation
-Normalized opponent strength
-Position one-hot encoding
-Home/Away binary indicator

Model Performance

Model: XGBoost Regressor
Mean Absolute Error: ~0.5

The model demonstrates strong predictive consistency on small but structured datasets.

Backend : https://player-rating-predictor.onrender.com/

Future Improvements

-Expand dataset to 100+ players
-Add real-time fixture scraping
-Implement bias detection system
-Introduce model comparison dashboard
-Add database storage (PostgreSQL)
