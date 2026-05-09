import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ASPICE AI Requirement Agent API is running"}

def test_generate_sys2():
    payload = {
        "user_input": "Brake control system",
        "level": "SYS.2"
    }
    response = client.post("/generate/sys2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "requirements" in data
    assert len(data["requirements"]) > 0
    assert "sys_id" in data["requirements"][0]
    print("\nSYS.2 Test Passed!")
    print(f"Sample Requirement: {data['requirements'][0]['title']}")

def test_generate_swe1():
    # Mock SYS.2 data
    sys2_req = {
        "sys_id": "SYS_REQ_001",
        "title": "Brake Logic",
        "description": "The system shall...",
        "category": "Safety",
        "verification_criteria": "Test"
    }
    payload = {
        "user_input": json.dumps(sys2_req),
        "level": "SWE.1"
    }
    response = client.post("/generate/swe1", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "requirements" in data
    assert len(data["requirements"]) > 0
    assert data["requirements"][0]["parent_sys_id"] == "SYS_REQ_001"
    print("\nSWE.1 Test Passed!")
    print(f"Sample Decomposed Requirement: {data['requirements'][0]['title']}")

if __name__ == "__main__":
    print("Running basic tests...")
    test_root()
    test_generate_sys2()
    test_generate_swe1()
    print("\nAll tests passed successfully!")
