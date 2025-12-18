"""
Test script for Chess Opening Recommender API
"""
import requests
import json

BASE_URL = "http://127.0.0.1:7860"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict_white():
    """Test prediction for white"""
    data = {
        "side": "white",
        "openings": []
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("White Predictions:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict_black():
    """Test prediction for black"""
    data = {
        "side": "black",
        "openings": []
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Black Predictions:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_predict_with_openings():
    """Test prediction with opening data"""
    data = {
        "side": "white",
        "openings": [
            {
                "opening_name": "Italian Game",
                "opening_eco": "C50",
                "win_rate": 0.55,
                "num_games": 42
            },
            {
                "opening_name": "Ruy Lopez",
                "opening_eco": "C70",
                "win_rate": 0.48,
                "num_games": 28
            }
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predictions with Opening Data:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing Chess Opening Recommender API")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_predict_white()
        test_predict_black()
        test_predict_with_openings()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  uvicorn app:app --port 7860")
