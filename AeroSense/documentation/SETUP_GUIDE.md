AeroSense Project - Setup Guide
Quick Start (5 Minutes)
Step 1: Create Project Structure
bash
# Create main project folder
mkdir AeroSense
cd AeroSense

# Create subfolders
mkdir frontend backend data documentation
Step 2: Set Up Backend
bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install flask==2.3.3 flask-cors==4.0.0 pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 joblib==1.3.2

# Train the ML model
python model_training.py

# Start the backend server
python app.py
Step 3: Launch Frontend
Open frontend/index.html in your web browser

Enter a city name and click "Predict Air Quality"

📁 Complete File Structure
text
AeroSense/
├── frontend/index.html
├── backend/
│   ├── app.py
│   ├── model_training.py
│   ├── requirements.txt
│   └── air_quality_model.pkl (created after training)
├── data/
│   ├── nasa_tempo_data.csv
│   └── health_impact_data.csv
└── documentation/SETUP_GUIDE.md
🎯 What AeroSense Does
AeroSense provides:

✅ Real-time air quality predictions for any city

✅ Health impact analysis based on WHO guidelines

✅ NASA TEMPO satellite data integration

✅ Multi-pollutant analysis (PM2.5, Ozone, NO2, etc.)

✅ Interactive charts and visualizations

✅ 24-hour AQI forecasts

🏙️ Supported Cities
Los Angeles, New York, Chicago

London, Paris, Berlin

Tokyo, Delhi

Plus any city you enter!

🔧 Troubleshooting
Backend won't start?

bash
# Check if port 5000 is available
netstat -an | findstr :5000  # Windows
lsof -i :5000                # Mac/Linux

# Try different port
python app.py --port 5001
Model training errors?

bash
# Reinstall scikit-learn
pip uninstall scikit-learn
pip install scikit-learn==1.3.0
Frontend not loading?

Open browser developer tools (F12)

Check for CORS errors

Ensure backend is running on http://127.0
.0.1:5000

🚀 Deployment Ready
AeroSense is production-ready with:

Clean, responsive design

RESTful API architecture

Modular ML pipeline

Comprehensive error handling