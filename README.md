# CropSense – Smart Crop Advisory System

An AI-powered web application that helps farmers make data-driven crop decisions 
based on soil and climate conditions.

## Features

- **Crop Recommendation** – Recommends the most suitable crop based on soil 
  nutrients (N, P, K, pH) and climate parameters (temperature, humidity, rainfall)
- **Yield Query** – Evaluates expected yield for a user-selected crop with a 
  confidence score and suitability rating
- **Fertilizer Guide** – Provides per-nutrient status (Good / Excess / Deficient) 
  and fertilizer suggestions based on soil values
- **Alternative Crop Suggestions** – Recommends better-suited crops when the 
  selected crop is unsuitable
- **Auto Location & Weather** – Auto-detects location and fetches real-time 
  weather data to reduce manual input
- **Multilingual Support** – Supports multiple Indian regional languages

## Tech Stack

- **Backend:** Python, Flask
- **ML Model:** Random Forest (Scikit-learn)
- **Frontend:** HTML
- **Dataset:** Crop Recommendation Dataset (Kaggle) – 2,200 samples, 22 crops

## ML Model Performance

| Model | Accuracy |
|---|---|
| Random Forest | 99.32% |
| Decision Tree | 98.18% |
| KNN | 97.05% |

## How It Works

1. User enters their location – weather is auto-fetched
2. User inputs soil values from their Soil Health Card
3. System predicts the best crop with a confidence score and explanation
4. Fertilizer guide and alternative crops are displayed alongside

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
## Contributors
KRUTIKA

KSHETRAGNA TALASILA

