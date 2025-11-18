"""
API Testing Script
Test the Disease Prediction API endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://192.168.18.8:5000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_models_info():
    """Test models info endpoint"""
    print("\n" + "="*60)
    print("Testing Models Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_diabetes_prediction():
    """Test diabetes prediction"""
    print("\n" + "="*60)
    print("Testing Diabetes Prediction")
    print("="*60)
    
    # Sample patient data
    patient_data = {
        'HighBP': 1,
        'HighChol': 1,
        'CholCheck': 1,
        'BMI': 32.5,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 3,
        'MentHlth': 2,
        'PhysHlth': 5,
        'DiffWalk': 0,
        'Sex': 1,
        'Age': 9,
        'Education': 5,
        'Income': 6
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/diabetes",
        json=patient_data
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"Error: {result.get('error')}")

def test_hypertension_prediction():
    """Test hypertension prediction"""
    print("\n" + "="*60)
    print("Testing Hypertension Prediction")
    print("="*60)
    
    patient_data = {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 145,
        'chol': 240,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.5,
        'slope': 2,
        'ca': 1,
        'thal': 2
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/hypertension",
        json=patient_data
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"Error: {result.get('error')}")

def test_cervical_cancer_prediction():
    """Test cervical cancer prediction"""
    print("\n" + "="*60)
    print("Testing Cervical Cancer Prediction")
    print("="*60)
    
    patient_data = {
        'Age': 35,
        'Number of sexual partners': 3,
        'First sexual intercourse': 18,
        'Num of pregnancies': 2,
        'Smokes': 0,
        'Smokes (years)': 0,
        'Smokes (packs/year)': 0,
        'Hormonal Contraceptives': 1,
        'Hormonal Contraceptives (years)': 5,
        'IUD': 0,
        'IUD (years)': 0,
        'STDs': 0,
        'STDs (number)': 0,
        'STDs:condylomatosis': 0,
        'STDs:cervical condylomatosis': 0,
        'STDs:vaginal condylomatosis': 0,
        'STDs:vulvo-perineal condylomatosis': 0,
        'STDs:syphilis': 0,
        'STDs:pelvic inflammatory disease': 0,
        'STDs:genital herpes': 0,
        'STDs:molluscum contagiosum': 0,
        'STDs:AIDS': 0,
        'STDs:HIV': 0,
        'STDs:Hepatitis B': 0,
        'STDs:HPV': 1,
        'STDs: Number of diagnosis': 1,
        'STDs: Time since first diagnosis': 2,
        'STDs: Time since last diagnosis': 2,
        'Dx:Cancer': 0,
        'Dx:CIN': 0,
        'Dx:HPV': 1,
        'Dx': 0,
        'Hinselmann': 0,
        'Schiller': 0,
        'Citology': 0
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/cervical_cancer",
        json=patient_data
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"Error: {result.get('error')}")

def test_oral_cancer_prediction():
    """Test oral cancer prediction"""
    print("\n" + "="*60)
    print("Testing Oral Cancer Prediction")
    print("="*60)
    
    patient_data = {
        'Country': 0,  # Encoded
        'Age': 55,
        'Gender': 1,
        'Tobacco Use': 1,
        'Alcohol Consumption': 1,
        'HPV Infection': 0,
        'Betel Quid Use': 0,
        'Chronic Sun Exposure': 1,
        'Poor Oral Hygiene': 1,
        'Diet (Fruits & Vegetables Intake)': 0,
        'Family History of Cancer': 0,
        'Compromised Immune System': 0,
        'Oral Lesions': 1,
        'Unexplained Bleeding': 1,
        'Difficulty Swallowing': 0,
        'White or Red Patches in Mouth': 1,
        'Tumor Size (cm)': 2.5,
        'Cancer Stage': 2,
        'Treatment Type': 1,
        'Survival Rate (5-Year, %)': 65,
        'Cost of Treatment (USD)': 15000,
        'Economic Burden (Lost Workdays per Year)': 30,
        'Early Diagnosis': 1
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/oral_cancer",
        json=patient_data
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"Error: {result.get('error')}")

def test_get_features():
    """Test get features endpoint"""
    print("\n" + "="*60)
    print("Testing Get Features for Diabetes")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/features/diabetes")
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nDisease: {result['disease']}")
        print(f"Number of features: {result['num_features']}")
        print(f"Features: {', '.join(result['features'][:5])}... (showing first 5)")
    else:
        print(f"Error: {result.get('error')}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Disease Prediction API - Test Suite")
    print("#"*60)
    
    try:
        # Test basic endpoints
        test_health_check()
        test_models_info()
        test_get_features()
        
        # Test predictions
        test_diabetes_prediction()
        test_hypertension_prediction()
        test_cervical_cancer_prediction()
        test_oral_cancer_prediction()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the server is running on http://localhost:5000")
        print("Run: python app.py")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")