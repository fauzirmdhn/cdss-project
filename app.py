"""
Disease Prediction API Backend
Flask REST API for Diabetes, Hypertension, Cervical Cancer, and Oral Cancer prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelLoader:
    """Load and manage ML models"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all disease models"""
        diseases = ['diabetes', 'hypertension', 'cervical_cancer', 'oral_cancer']
        
        for disease in diseases:
            try:
                model_path = f'{self.model_dir}/{disease}_model.pkl'
                scaler_path = f'{self.model_dir}/{disease}_scaler.pkl'
                metadata_path = f'{self.model_dir}/{disease}_metadata.pkl'
                
                if os.path.exists(model_path):
                    self.models[disease] = {
                        'model': joblib.load(model_path),
                        'scaler': joblib.load(scaler_path),
                        'metadata': joblib.load(metadata_path)
                    }
                    
                    # Load label encoder if exists
                    encoder_path = f'{self.model_dir}/{disease}_label_encoder.pkl'
                    if os.path.exists(encoder_path):
                        self.models[disease]['label_encoder'] = joblib.load(encoder_path)
                    
                    print(f"✓ Loaded {disease} model successfully")
                else:
                    print(f"✗ Model file not found: {model_path}")
                    
            except Exception as e:
                print(f"✗ Error loading {disease} model: {str(e)}")
    
    def get_model(self, disease):
        """Get model for specific disease"""
        return self.models.get(disease.lower().replace(' ', '_'))

# Initialize model loader
model_loader = ModelLoader()

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_disease(disease_name, input_data):
    """Make prediction for a specific disease"""
    
    # Get model components
    model_info = model_loader.get_model(disease_name)
    
    if not model_info:
        return {
            'success': False,
            'error': f'Model for {disease_name} not found. Please train the model first.'
        }
    
    try:
        model = model_info['model']
        scaler = model_info['scaler']
        metadata = model_info['metadata']
        feature_names = metadata['feature_names']
        class_names = metadata['class_names']
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        
        # Ensure all required features are present
        missing_features = []
        for feature in feature_names:
            if feature not in df.columns:
                missing_features.append(feature)
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Handle any string values (encode them)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    df[col] = 0
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Decode prediction if label encoder exists
        if 'label_encoder' in model_info:
            prediction_label = model_info['label_encoder'].inverse_transform([prediction])[0]
        else:
            prediction_label = str(prediction)
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[str(class_name)] = float(probabilities[i])
        
        # Determine risk level
        confidence = float(max(probabilities))
        risk_level = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'
        
        # Generate recommendations
        recommendations = generate_recommendations(disease_name, prediction_label, confidence)
        
        return {
            'success': True,
            'disease': disease_name.replace('_', ' ').title(),
            'prediction': prediction_label,
            'confidence': confidence,
            'risk_level': risk_level,
            'probabilities': prob_dict,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'missing_features': missing_features if missing_features else None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_recommendations(disease, prediction, confidence):
    """Generate health recommendations based on prediction"""
    
    recommendations = []
    
    if disease == 'diabetes':
        if prediction in ['1', '2', 1, 2]:  # Prediabetes or Diabetes
            recommendations = [
                "Consult with an endocrinologist for proper diagnosis and treatment plan",
                "Monitor blood glucose levels regularly",
                "Adopt a low-glycemic diet rich in vegetables and whole grains",
                "Engage in at least 150 minutes of moderate exercise per week",
                "Maintain a healthy weight (BMI 18.5-24.9)",
                "Consider medication if prescribed by your doctor",
                "Regular check-ups for complications (eyes, kidneys, feet)"
            ]
        else:
            recommendations = [
                "Maintain healthy lifestyle to prevent diabetes",
                "Regular health screenings every 1-2 years",
                "Keep BMI in healthy range",
                "Stay physically active"
            ]
    
    elif disease == 'hypertension':
        if prediction in ['1', 1, 'Yes']:
            recommendations = [
                "Consult a cardiologist for proper blood pressure management",
                "Reduce sodium intake (less than 2,300mg per day)",
                "Follow DASH diet (fruits, vegetables, low-fat dairy)",
                "Exercise regularly (30 minutes most days)",
                "Limit alcohol consumption",
                "Manage stress through meditation or yoga",
                "Take prescribed medications as directed",
                "Monitor blood pressure at home regularly"
            ]
        else:
            recommendations = [
                "Maintain heart-healthy lifestyle",
                "Regular blood pressure checks",
                "Limit sodium and maintain healthy diet",
                "Stay active and manage stress"
            ]
    
    elif disease == 'cervical_cancer':
        if prediction in ['1', 1, 'Yes']:
            recommendations = [
                "URGENT: Consult a gynecologic oncologist immediately",
                "Schedule comprehensive pelvic examination",
                "Get HPV testing and Pap smear",
                "Discuss colposcopy and biopsy with your doctor",
                "Consider HPV vaccination if not already vaccinated",
                "Discuss treatment options (surgery, radiation, chemotherapy)",
                "Regular follow-up appointments are critical",
                "Seek support from cancer support groups"
            ]
        else:
            recommendations = [
                "Continue regular cervical cancer screenings",
                "HPV vaccination if eligible",
                "Practice safe sex",
                "Avoid smoking",
                "Regular Pap smears as recommended by age"
            ]
    
    elif disease == 'oral_cancer':
        if prediction in ['Yes', 'yes', 1]:
            recommendations = [
                "URGENT: Consult an oral and maxillofacial surgeon or oncologist",
                "Get comprehensive oral examination and biopsy",
                "Discuss imaging studies (CT, MRI, PET scan)",
                "Stop all tobacco and alcohol use immediately",
                "Consider treatment options based on staging",
                "Maintain excellent oral hygiene",
                "Nutritional support may be needed during treatment",
                "Join oral cancer support groups"
            ]
        else:
            recommendations = [
                "Regular dental check-ups every 6 months",
                "Avoid tobacco in all forms",
                "Limit alcohol consumption",
                "Maintain good oral hygiene",
                "Eat a diet rich in fruits and vegetables",
                "Protect lips from sun exposure"
            ]
    
    # Add confidence-based disclaimer
    if confidence < 0.7:
        recommendations.insert(0, "⚠️ Note: Prediction confidence is moderate. Additional testing recommended.")
    
    return recommendations

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Disease Prediction API',
        'version': '1.0',
        'available_models': list(model_loader.models.keys()),
        'endpoints': {
            '/predict/diabetes': 'POST - Predict diabetes',
            '/predict/hypertension': 'POST - Predict hypertension',
            '/predict/cervical_cancer': 'POST - Predict cervical cancer',
            '/predict/oral_cancer': 'POST - Predict oral cancer',
            '/predict/all': 'POST - Predict all diseases',
            '/health': 'GET - Health check',
            '/models/info': 'GET - Get model information'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_loader.models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models/info')
def models_info():
    """Get information about loaded models"""
    info = {}
    for disease, model_info in model_loader.models.items():
        metadata = model_info['metadata']
        info[disease] = {
            'disease_name': metadata['disease_name'],
            'features': metadata['feature_names'],
            'classes': metadata['class_names'],
            'num_features': len(metadata['feature_names'])
        }
    return jsonify(info)

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """Predict diabetes"""
    try:
        data = request.get_json()
        result = predict_disease('diabetes', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/hypertension', methods=['POST'])
def predict_hypertension():
    """Predict hypertension"""
    try:
        data = request.get_json()
        result = predict_disease('hypertension', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/cervical_cancer', methods=['POST'])
def predict_cervical_cancer():
    """Predict cervical cancer"""
    try:
        data = request.get_json()
        result = predict_disease('cervical_cancer', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/oral_cancer', methods=['POST'])
def predict_oral_cancer():
    """Predict oral cancer"""
    try:
        data = request.get_json()
        result = predict_disease('oral_cancer', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Predict all diseases with same input (if applicable)"""
    try:
        data = request.get_json()
        
        results = {
            'diabetes': predict_disease('diabetes', data),
            'hypertension': predict_disease('hypertension', data),
            'cervical_cancer': predict_disease('cervical_cancer', data),
            'oral_cancer': predict_disease('oral_cancer', data)
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/features/<disease>', methods=['GET'])
def get_features(disease):
    """Get required features for a specific disease"""
    model_info = model_loader.get_model(disease)
    
    if not model_info:
        return jsonify({'success': False, 'error': 'Disease model not found'}), 404
    
    metadata = model_info['metadata']
    return jsonify({
        'success': True,
        'disease': disease,
        'features': metadata['feature_names'],
        'num_features': len(metadata['feature_names']),
        'classes': metadata['class_names']
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Disease Prediction API Server")
    print("="*60)
    print(f"Models loaded: {len(model_loader.models)}")
    print(f"Available diseases: {list(model_loader.models.keys())}")
    print("\nStarting server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)