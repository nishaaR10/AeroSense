from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Global variables for model and analyzer
predictor = None
health_analyzer = None

# Simple Health Impact Analyzer
class SimpleHealthImpactAnalyzer:
    def __init__(self):
        self.health_data = {
            (0, 50): {'asthma_risk': '5%', 'lung_function': 'None', 'cardio_risk': '3%', 'vulnerable_alert': 'None', 'recommendation': 'Air quality is satisfactory. No health impacts expected.'},
            (51, 100): {'asthma_risk': '15%', 'lung_function': 'Mild', 'cardio_risk': '8%', 'vulnerable_alert': 'Low', 'recommendation': 'Sensitive individuals should consider reducing prolonged outdoor exertion.'},
            (101, 150): {'asthma_risk': '30%', 'lung_function': 'Moderate', 'cardio_risk': '18%', 'vulnerable_alert': 'Moderate', 'recommendation': 'Children and people with respiratory issues should limit outdoor activities.'},
            (151, 200): {'asthma_risk': '50%', 'lung_function': 'Significant', 'cardio_risk': '30%', 'vulnerable_alert': 'High', 'recommendation': 'Everyone should reduce outdoor activities. Sensitive groups stay indoors.'},
            (201, 300): {'asthma_risk': '75%', 'lung_function': 'Severe', 'cardio_risk': '45%', 'vulnerable_alert': 'Very High', 'recommendation': 'Avoid all outdoor activities. Sensitive groups remain indoors.'},
            (301, 500): {'asthma_risk': '95%', 'lung_function': 'Extreme', 'cardio_risk': '65%', 'vulnerable_alert': 'Extreme', 'recommendation': 'Health emergency. Remain indoors with air purifiers.'}
        }
    
    def get_health_impact(self, aqi):
        """Get health impact based on AQI value"""
        for (min_aqi, max_aqi), impact in self.health_data.items():
            if min_aqi <= aqi <= max_aqi:
                return impact
        return self.health_data[(301, 500)]  # Default to worst case

def initialize_models():
    """Initialize the ML model and health analyzer"""
    global predictor, health_analyzer
    
    try:
        # Try to import and load the trained model
        model_path = os.path.join(os.path.dirname(__file__), 'air_quality_model.pkl')
        
        if not os.path.exists(model_path):
            print("❌ Model file not found. Please run model_training.py first.")
            print("🔧 Starting in demo mode with simulated predictions...")
            predictor = None
        else:
            # Load the model directly
            predictor = joblib.load(model_path)
            print("✅ ML model loaded successfully!")
        
        # Initialize health impact analyzer
        health_analyzer = SimpleHealthImpactAnalyzer()
        
        print("✅ AeroSense models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("🔧 Starting in demo mode with simulated predictions...")
        health_analyzer = SimpleHealthImpactAnalyzer()
        return True  # Continue in demo mode

@app.route('/')
def home():
    return jsonify({
        "message": "AeroSense Air Quality Prediction API", 
        "status": "running",
        "mode": "demo" if predictor is None else "ml",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "models_loaded": predictor is not None,
        "project": "AeroSense",
        "mode": "demo" if predictor is None else "ml"
    })

@app.route('/predict', methods=['POST'])
def predict_air_quality():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        city = data.get('city', '').strip()
        
        if not city:
            return jsonify({'error': 'City name is required'}), 400
            
        print(f"🎯 AeroSense received prediction request for: {city}")
        
        # If model is loaded, use it; otherwise use demo mode
        if predictor is not None:
            try:
                # Generate sample data based on city
                sample_data = generate_sample_data(city)
                predicted_aqi = predictor.predict(sample_data)[0]
                mode = "ml"
            except Exception as e:
                print(f"❌ ML prediction failed, using demo mode: {e}")
                predicted_aqi = generate_demo_aqi(city)
                mode = "demo"
        else:
            # Demo mode - generate realistic AQI based on city
            predicted_aqi = generate_demo_aqi(city)
            mode = "demo"
        
        # Get health impact
        health_impact = health_analyzer.get_health_impact(predicted_aqi)
        
        # Generate trend data
        trend = generate_trend_data(city, predicted_aqi)
        
        # Generate pollutant data
        pollutants = generate_pollutant_data(city)
        
        response = {
            'city': city,
            'current_aqi': trend[-2],  # Second last is "today"
            'predicted_aqi': round(predicted_aqi, 1),
            'status': get_aqi_status(predicted_aqi),
            'trend': [round(x, 1) for x in trend],
            'pollutants': pollutants,
            'health_impact': health_impact,
            'mode': mode
        }
        
        print(f"✅ AeroSense prediction completed for {city}: AQI {predicted_aqi:.1f} ({mode} mode)")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def generate_sample_data(city):
    """Generate sample feature data for ML model"""
    base_values = {
        'Los Angeles': {'no2': 15, 'pm25': 22, 'ozone': 48},
        'New York': {'no2': 18, 'pm25': 25, 'ozone': 52},
        'Chicago': {'no2': 16, 'pm25': 23, 'ozone': 46},
        'London': {'no2': 12, 'pm25': 18, 'ozone': 42},
        'Tokyo': {'no2': 14, 'pm25': 20, 'ozone': 45},
        'Delhi': {'no2': 25, 'pm25': 35, 'ozone': 55},
        'Paris': {'no2': 13, 'pm25': 19, 'ozone': 43},
        'Berlin': {'no2': 11, 'pm25': 17, 'ozone': 41}
    }
    
    city_data = base_values.get(city, base_values['Los Angeles'])
    
    # Create feature array in the same order as training
    features = [
        city_data['no2'],                    # no2_nrt
        city_data['no2'] * 0.7,              # hcho_nrt
        city_data['pm25'] / 15,              # aerosol_index
        city_data['pm25'],                   # pm2_5
        city_data['pm25'] * 1.8,             # pm10
        city_data['ozone'],                  # ozone
        city_data['pm25'] - 2,               # pm2_5_lag_1
        city_data['no2'] - 1,                # no2_lag_1
        (city_data['pm25'] * 2.5) - 5,       # aqi_lag_1
        city_data['pm25'] - 1,               # pm2_5_rolling_3
        city_data['no2'] - 0.5,              # no2_rolling_3
        150,                                 # day_of_year
        6,                                   # month
        2                                    # day_of_week
    ]
    
    return np.array([features])

def generate_demo_aqi(city):
    """Generate demo AQI when model is not available"""
    base_aqi = {
        'Los Angeles': 65, 'New York': 72, 'Chicago': 58,
        'London': 45, 'Tokyo': 52, 'Delhi': 85,
        'Paris': 48, 'Berlin': 42
    }
    return base_aqi.get(city, 55) + np.random.randint(-10, 10)

def generate_trend_data(city, predicted_aqi):
    """Generate trend data for the chart"""
    base = predicted_aqi - np.random.randint(10, 30)
    trend = []
    for i in range(6):
        if i < 5:  # Past days
            value = base + np.random.randint(0, 15)
            trend.append(max(10, value))
        else:  # Prediction
            trend.append(predicted_aqi)
    return trend

def generate_pollutant_data(city):
    """Generate pollutant concentration data"""
    base = {
        'Los Angeles': {'pm25': 22, 'pm10': 40, 'o3': 48, 'no2': 15, 'so2': 8, 'co': 1.2},
        'New York': {'pm25': 25, 'pm10': 45, 'o3': 52, 'no2': 18, 'so2': 10, 'co': 1.5},
        'Chicago': {'pm25': 23, 'pm10': 42, 'o3': 46, 'no2': 16, 'so2': 9, 'co': 1.3},
        'London': {'pm25': 18, 'pm10': 32, 'o3': 42, 'no2': 12, 'so2': 6, 'co': 0.9},
        'Tokyo': {'pm25': 20, 'pm10': 35, 'o3': 45, 'no2': 14, 'so2': 7, 'co': 1.0},
        'Delhi': {'pm25': 35, 'pm10': 65, 'o3': 55, 'no2': 25, 'so2': 15, 'co': 2.0},
        'Paris': {'pm25': 19, 'pm10': 33, 'o3': 43, 'no2': 13, 'so2': 7, 'co': 0.8},
        'Berlin': {'pm25': 17, 'pm10': 30, 'o3': 41, 'no2': 11, 'so2': 5, 'co': 0.7}
    }
    
    pollutants = base.get(city, base['Los Angeles'])
    return {
        'PM2.5': pollutants['pm25'],
        'PM10': pollutants['pm10'],
        'O3': pollutants['o3'],
        'NO2': pollutants['no2'],
        'SO2': pollutants['so2'],
        'CO': pollutants['co']
    }

def get_aqi_status(aqi):
    """Convert AQI value to status string"""
    if aqi <= 50: return 'Good'
    if aqi <= 100: return 'Moderate'
    if aqi <= 150: return 'Poor'
    return 'Severe'

if __name__ == '__main__':
    print("🚀 Starting AeroSense Air Quality Prediction Server...")
    print("📍 Project: AeroSense - NASA Space Apps Challenge")
    
    # Initialize models (will work in demo mode if model not found)
    if initialize_models():
        print("🌐 AeroSense server starting on http://127.0.0.1:5000")
        print("💡 Use Ctrl+C to stop the server")
        try:
            app.run(debug=True, host='127.0.0.1', port=5000)
        except Exception as e:
            print(f"❌ Server error: {e}")
            print("💡 Trying alternative port...")
            app.run(debug=True, host='127.0.0.1', port=5001)
    else:
        print("❌ Failed to start AeroSense server.")