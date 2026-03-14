import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = joblib.load('student_score_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        hours_studied = float(request.form['hours_studied'])
        sleep_hours = float(request.form['sleep_hours'])
        attendance_percent = float(request.form['attendance_percent'])
        previous_scores = float(request.form['previous_scores'])

        # Prepare features for the model
        # Features order must match training: ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
        features = pd.DataFrame([[hours_studied, sleep_hours, attendance_percent, previous_scores]],
                               columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f'Estimated Student Score: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
