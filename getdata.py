import requests

def fetch_prediction(data):
    url = 'http://127.0.0.1:5000/predict'
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        prediction = response.json()
        return prediction
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Example symptom data
data = {
   'Fever': 1,
    'Cough': 0,
    'Fatigue': 1,
    'Difficulty Breathing': 0,
    'Sore Throat': 1,
    'Rash': 0,
    'Headache': 1,
    'Nausea': 0,
    'Vomiting': 0,
    'Diarrhea': 0,
    'Muscle Pain': 1
}

# Fetch prediction
prediction = fetch_prediction(data)
if prediction:
    print(f"Random Forest Prediction: {prediction['RandomForestPrediction']}")
    print(f"Gradient Boosting Prediction: {prediction['GradientBoostingPrediction']}")
