import requests
import json

def test_api_connection():
    """Test the API connection and endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API Connection")
    print("=" * 40)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Models: {data.get('models', [])}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the Python service is running on port 8000")
        return False
    
    # Test 2: Single prediction
    try:
        test_data = {
            "model": "Random Forest",
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction test passed")
            print(f"   Model: {result.get('model')}")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Single prediction request failed: {e}")
    
    # Test 3: All models prediction
    try:
        test_data = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        
        response = requests.post(
            f"{base_url}/predict-all",
            json=test_data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            print("✅ All models prediction test passed")
            print(f"   Number of models: {len(predictions)}")
            
            for pred in predictions:
                print(f"   {pred.get('model')}: {pred.get('prediction')} ({pred.get('confidence', 0):.2%})")
        else:
            print(f"❌ All models prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ All models prediction request failed: {e}")
    
    print("\n🎯 API testing complete!")
    print("   If all tests passed, your frontend should work correctly.")
    return True

if __name__ == "__main__":
    test_api_connection()
