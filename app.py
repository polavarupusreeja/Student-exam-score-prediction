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
        prediction = float(model.predict(features)[0])

        # Determine performance level and suggestion
        if prediction < 40:
            performance = "Poor"
            color = "#ef4444" # Red
            suggestion = "Consider increasing your study hours and ensuring consistent attendance to improve your scores."
        elif prediction < 75:
            performance = "Average"
            color = "#f59e0b" # Yellow
            suggestion = "You're doing okay, but more consistent practice and reviewing key concepts could help you reach the next level."
        else:
            performance = "Good"
            color = "#22c55e" # Green
            suggestion = "Good keep it up!"

        return render_template('result.html', 
                             prediction=f'{prediction:.2f}', 
                             performance=performance, 
                             color=color, 
                             suggestion=suggestion)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
