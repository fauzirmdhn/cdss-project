from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import traceback

import sys
sys.path.append('.')  # Ensure current directory is in path

try:
    from recommendationsys import UltraAdvancedRecommendationEngine
    ADVANCED_RECS_AVAILABLE = True
    print("✓ Advanced recommendation engine loaded successfully")
except ImportError:
    ADVANCED_RECS_AVAILABLE = False
    print("⚠️ Advanced recommendation engine not found - using basic recommendations")

app = Flask(__name__)
CORS(app)


# MODEL LOADING

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

# Initialize model loader and recommendation engine
model_loader = ModelLoader()

if ADVANCED_RECS_AVAILABLE:
    recommendation_engine = UltraAdvancedRecommendationEngine()
    print("✓ Recommendation engine initialized")

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_disease(disease_name, input_data):
    """Make prediction for a specific disease with advanced recommendations"""
    
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
        
        # Store original input for recommendations
        original_input = input_data.copy() if isinstance(input_data, dict) else input_data.to_dict('records')[0]
        
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
        
        # Determine confidence
        confidence = float(max(probabilities))
        
        # Generate advanced recommendations
        if ADVANCED_RECS_AVAILABLE:
            try:
                recommendations = recommendation_engine.generate_comprehensive_recommendations(
                    disease=disease_name,
                    prediction=str(prediction_label),
                    confidence=confidence,
                    patient_data=original_input
                )
                recommendations_available = True
            except Exception as e:
                print(f"Error generating advanced recommendations: {str(e)}")
                recommendations = generate_basic_recommendations(disease_name, prediction_label, confidence)
                recommendations_available = False
        else:
            recommendations = generate_basic_recommendations(disease_name, prediction_label, confidence)
            recommendations_available = False
        
        return {
            'success': True,
            'disease': disease_name.replace('_', ' ').title(),
            'prediction': str(prediction_label),
            'confidence': confidence,
            'risk_level': determine_simple_risk_level(confidence),
            'probabilities': prob_dict,
            'recommendations': recommendations,
            'advanced_recommendations_enabled': recommendations_available,
            'timestamp': datetime.now().isoformat(),
            'missing_features': missing_features if missing_features else None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def determine_simple_risk_level(confidence):
    """Simple risk level determination for backward compatibility"""
    if confidence > 0.8:
        return 'High'
    elif confidence > 0.6:
        return 'Medium'
    else:
        return 'Low'

def generate_basic_recommendations(disease, prediction, confidence):
    """Basic fallback recommendations if advanced engine not available"""
    
    recommendations = {
        'immediate_actions': [],
        'lifestyle_modifications': [],
        'follow_up_schedule': {},
        'warning_signs': []
    }
    
    if prediction in ['1', '2', 1, 2, 'Yes', 'yes']:
        recommendations['immediate_actions'] = [
            f"Schedule appointment with healthcare provider for {disease.replace('_', ' ')}",
            "Begin tracking symptoms and relevant health metrics",
            "Review current medications with doctor",
            "Consider lifestyle modifications"
        ]
        
        recommendations['lifestyle_modifications'] = [
            "Maintain a healthy diet",
            "Exercise regularly (consult doctor first)",
            "Manage stress levels",
            "Get adequate sleep (7-9 hours)",
            "Avoid smoking and limit alcohol"
        ]
        
        recommendations['warning_signs'] = [
            "Worsening of current symptoms",
            "New or unusual symptoms",
            "Side effects from medications",
            "Signs of complications"
        ]
    else:
        recommendations['immediate_actions'] = [
            "Continue healthy lifestyle practices",
            "Schedule routine health screening"
        ]
    
    return recommendations

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Disease Prediction API with Advanced Recommendations',
        'version': '2.0',
        'advanced_recommendations': ADVANCED_RECS_AVAILABLE,
        'available_models': list(model_loader.models.keys()),
        'endpoints': {
            '/predict/diabetes': 'POST - Predict diabetes with comprehensive recommendations',
            '/predict/hypertension': 'POST - Predict hypertension with recommendations',
            '/predict/cervical_cancer': 'POST - Predict cervical cancer with recommendations',
            '/predict/oral_cancer': 'POST - Predict oral cancer with recommendations',
            '/predict/all': 'POST - Predict all diseases',
            '/health': 'GET - Health check',
            '/models/info': 'GET - Get model information',
            '/features/<disease>': 'GET - Get required features for disease',
            '/visualization': 'GET - View recommendation hierarchy visualization'
        },
        'features': {
            'ml_prediction': True,
            'risk_stratification': ADVANCED_RECS_AVAILABLE,
            'clinical_guidelines': ADVANCED_RECS_AVAILABLE,
            'treatment_pathways': ADVANCED_RECS_AVAILABLE,
            'medication_algorithms': ADVANCED_RECS_AVAILABLE,
            'smart_goals': ADVANCED_RECS_AVAILABLE,
            'monitoring_schedules': ADVANCED_RECS_AVAILABLE,
            'cost_considerations': ADVANCED_RECS_AVAILABLE
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_loader.models),
        'advanced_recommendations': ADVANCED_RECS_AVAILABLE,
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
    """Predict diabetes with advanced recommendations"""
    try:
        data = request.get_json()
        result = predict_disease('diabetes', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/hypertension', methods=['POST'])
def predict_hypertension():
    """Predict hypertension with advanced recommendations"""
    try:
        data = request.get_json()
        result = predict_disease('hypertension', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/cervical_cancer', methods=['POST'])
def predict_cervical_cancer():
    """Predict cervical cancer with advanced recommendations"""
    try:
        data = request.get_json()
        result = predict_disease('cervical_cancer', data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/oral_cancer', methods=['POST'])
def predict_oral_cancer():
    """Predict oral cancer with advanced recommendations"""
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

@app.route('/visualization')
def visualization():
    """Serve the recommendation hierarchy visualization"""
    return send_from_directory('.', 'recommendation_hierarchy.html')

@app.route('/test/diabetes', methods=['GET'])
def test_diabetes():
    """Test endpoint with sample diabetes data"""
    sample_data = {
        'HighBP': 1,
        'HighChol': 1,
        'CholCheck': 1,
        'BMI': 32.5,
        'Smoker': 1,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 3,
        'MentHlth': 5,
        'PhysHlth': 10,
        'DiffWalk': 0,
        'Sex': 1,
        'Age': 58,
        'Education': 5,
        'Income': 6
    }
    
    result = predict_disease('diabetes', sample_data)
    return jsonify(result)

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
    # Server configuration
    HOST = '192.168.18.8'
    PORT = 5000
    
    print("\n" + "="*70)
    print("Disease Prediction API Server v2.0")
    print("="*70)
    print(f"Models loaded: {len(model_loader.models)}")
    print(f"Available diseases: {list(model_loader.models.keys())}")
    print(f"Advanced recommendations: {'✓ ENABLED' if ADVANCED_RECS_AVAILABLE else '✗ DISABLED (using basic)'}")
    print(f"\nServer Host: {HOST}")
    print(f"Server Port: {PORT}")
    print("\nEndpoints:")
    print(f"  - http://{HOST}:{PORT}/ (API home)")
    print(f"  - http://{HOST}:{PORT}/health (health check)")
    print(f"  - http://{HOST}:{PORT}/models/info (model details)")
    print(f"  - http://{HOST}:{PORT}/predict/diabetes (POST prediction)")
    print(f"  - http://{HOST}:{PORT}/test/diabetes (GET test endpoint)")
    print(f"  - http://{HOST}:{PORT}/visualization (recommendation hierarchy)")
    print(f"\nStarting server on http://{HOST}:{PORT}")
    print("="*70 + "\n")
    
    # Start the Flask development server
    app.run(debug=True, host=HOST, port=PORT)
