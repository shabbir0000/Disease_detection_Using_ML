

# without scalar ke code ha is sa prediction sahi nahi a rahi thi 
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the pre-trained models
# rf_model = joblib.load('best_rf_classifier.joblib')
# gb_model = joblib.load('best_gb_classifier.joblib')

# # Define the prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     symptoms = [
#         'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
#         'Sore Throat', 'Rash', 'Headache', 'Nausea',
#         'Vomiting', 'Diarrhea', 'Muscle Pain'
#     ]

#     # Ensure all required symptoms are provided
#     if not all(symptom in data for symptom in symptoms):
#         return jsonify({"error": "Missing symptoms in the input data"}), 400

#     # Extract symptom values and convert to a list of integers
#     input_data = [data[symptom] for symptom in symptoms]
#     input_data = np.array(input_data).reshape(1, -1)

#     # Make predictions with both models
#     rf_prediction = rf_model.predict(input_data)
#     gb_prediction = gb_model.predict(input_data)

#     response = {
#         'RandomForestPrediction': rf_prediction[0],
#         'GradientBoostingPrediction': gb_prediction[0]
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models and scaler
rf_model = joblib.load('best_rf_classifier.joblib')
gb_model = joblib.load('best_gb_classifier.joblib')
scaler = joblib.load('scaler.joblib')  # Assuming you saved the scaler as 'scaler.joblib'

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = [
        'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
        'Sore Throat', 'Rash', 'Headache', 'Nausea',
        'Vomiting', 'Diarrhea', 'Muscle Pain'
    ]

    # Ensure all required symptoms are provided
    if not all(symptom in data for symptom in symptoms):
        return jsonify({"error": "Missing symptoms in the input data"}), 400

    # Extract symptom values and convert to a list of integers
    input_data = [data[symptom] for symptom in symptoms]
    input_data = np.array(input_data).reshape(1, -1)

    # Standardize the features
    feature_df_scaled = scaler.transform(input_data)

    # Make predictions with both models
    rf_prediction = rf_model.predict(feature_df_scaled)
    gb_prediction = gb_model.predict(feature_df_scaled)

    response = {
        'RandomForestPrediction': rf_prediction[0],
        'GradientBoostingPrediction': gb_prediction[0]
    }

    return jsonify(response)

# agr neche wala main run na ho to ye kar laina 
# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)