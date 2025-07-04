from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('tree_sustainability_model.pkl')

# Mapping for dropdowns
soil_type_map = {'clay': 0, 'silt': 1, 'sand': 2, 'loam': 3}
land_use_map = {'agriculture': 0, 'urban': 1, 'forest': 2, 'mixed': 3}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get input values from form
        tree_height = float(request.form['tree_height'])
        tree_diameter = float(request.form['tree_diameter'])
        age = float(request.form['age'])
        soil_ph = float(request.form['soil_ph'])
        soil_type = soil_type_map[request.form['soil_type']]
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        sun_hours = float(request.form['sun_hours'])
        slope = float(request.form['slope'])
        elevation = float(request.form['elevation'])
        land_use = land_use_map[request.form['land_use']]
        proximity_water = float(request.form['proximity_water'])
        competition_index = float(request.form['competition_index'])
        disease_index = float(request.form['disease_index'])
        pollution_level = float(request.form['pollution_level'])

        # Arrange values in the same order as training
        input_features = np.array([[tree_height, tree_diameter, age, soil_ph,
                                    soil_type, rainfall, temperature, sun_hours,
                                    slope, elevation, land_use, proximity_water,
                                    competition_index, disease_index, pollution_level]])

        # Predict
        result = model.predict(input_features)[0]
        result_text = "🌳 The tree is Sustainable ✅" if result == 1 else "🪵 The tree is NOT Sustainable ❌"

        return render_template('index.html', prediction=result_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)