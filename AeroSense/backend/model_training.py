import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = ['no2_nrt', 'hcho_nrt', 'aerosol_index', 'pm2_5', 'pm10', 'ozone']
        self.target_column = 'aqi'
        
    def load_data(self, file_path):
        """Load and preprocess the air quality data"""
        # Use absolute path
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    
    def create_features(self, df):
        """Create additional features for the model"""
        # Lag features for time series prediction
        for lag in [1, 2, 3]:
            df[f'pm2_5_lag_{lag}'] = df.groupby('city')['pm2_5'].shift(lag)
            df[f'no2_lag_{lag}'] = df.groupby('city')['no2_nrt'].shift(lag)
            df[f'aqi_lag_{lag}'] = df.groupby('city')['aqi'].shift(lag)
        
        # Rolling statistics
        df['pm2_5_rolling_3'] = df.groupby('city')['pm2_5'].rolling(3).mean().reset_index(0, drop=True)
        df['no2_rolling_3'] = df.groupby('city')['no2_nrt'].rolling(3).mean().reset_index(0, drop=True)
        
        # Seasonal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        # Drop rows with NaN values created by lag features
        df_clean = df.dropna()
        
        # Select features
        feature_cols = self.feature_columns + [
            'pm2_5_lag_1', 'no2_lag_1', 'aqi_lag_1',
            'pm2_5_rolling_3', 'no2_rolling_3',
            'day_of_year', 'month', 'day_of_week'
        ]
        
        X = df_clean[feature_cols]
        y = df_clean[self.target_column]
        
        return X, y, feature_cols
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        return X_test, y_test, y_pred
    
    def predict_aqi(self, input_features):
        """Predict AQI for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        prediction = self.model.predict(input_features)
        return prediction[0]
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, file_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """Load a pre-trained model"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")

# Health Impact Analyzer
class HealthImpactAnalyzer:
    def __init__(self, health_data_path):
        # Use absolute path
        if not os.path.exists(health_data_path):
            health_data_path = os.path.join(os.path.dirname(__file__), health_data_path)
        
        self.health_data = pd.read_csv(health_data_path)
    
    def get_health_impact(self, aqi):
        """Get health impact based on AQI value"""
        for _, row in self.health_data.iterrows():
            aqi_min, aqi_max = map(int, row['aqi_range'].split('-'))
            if aqi_min <= aqi <= aqi_max:
                return {
                    'asthma_risk': row['asthma_risk_percent'],
                    'lung_function': row['lung_function_impact'],
                    'cardio_risk': row['cardio_risk_percent'],
                    'vulnerable_alert': row['vulnerable_alert'],
                    'recommendation': row['recommendation']
                }
        return None

def main():
    print("🚀 Starting AeroSense Air Quality Model Training...")
    
    # Initialize the predictor
    predictor = AirQualityPredictor()
    
    try:
        # Load and prepare data
        print("📊 Loading NASA TEMPO data...")
        df = predictor.load_data('../data/nasa_tempo_data.csv')
        df = predictor.create_features(df)
        X, y, feature_cols = predictor.prepare_training_data(df)
        
        print(f"✅ Data loaded: {len(X)} samples with {len(feature_cols)} features")
        
        # Train the model
        print("🤖 Training Random Forest model...")
        X_test, y_test, y_pred = predictor.train_model(X, y)
        
        # Get feature importance
        importance = predictor.get_feature_importance(feature_cols)
        print("\n📈 Feature Importance:")
        print(importance.head(10))
        
        # Save the model
        model_path = 'air_quality_model.pkl'
        predictor.save_model(model_path)
        
        # Initialize health impact analyzer
        health_analyzer = HealthImpactAnalyzer('../data/health_impact_data.csv')
        
        # Example prediction
        print("\n🎯 Example Prediction:")
        sample_city_data = {
            'no2_nrt': 15.0, 'hcho_nrt': 9.5, 'aerosol_index': 1.8,
            'pm2_5': 22.0, 'pm10': 38.0, 'ozone': 48.0,
            'pm2_5_lag_1': 20.5, 'no2_lag_1': 14.2, 'aqi_lag_1': 62.0,
            'pm2_5_rolling_3': 21.2, 'no2_rolling_3': 14.8,
            'day_of_year': 150, 'month': 6, 'day_of_week': 2
        }
        
        sample_df = pd.DataFrame([sample_city_data])
        predicted_aqi = predictor.predict_aqi(sample_df[feature_cols])
        
        print(f"Predicted AQI: {predicted_aqi:.1f}")
        
        # Get health impact
        health_impact = health_analyzer.get_health_impact(predicted_aqi)
        if health_impact:
            print(f"🏥 Health Impact Analysis:")
            print(f"  • Asthma Risk Increase: {health_impact['asthma_risk']}%")
            print(f"  • Lung Function Impact: {health_impact['lung_function']}")
            print(f"  • Cardiovascular Risk: {health_impact['cardio_risk']}%")
            print(f"  • Vulnerable Groups: {health_impact['vulnerable_alert']}")
            print(f"  • Recommendation: {health_impact['recommendation']}")
            
        print("\n✅ AeroSense model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check if the data files exist in the correct location.")

if __name__ == "__main__":
    main()