from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_main_resource():
    response = client.get("/")
    assert response.status_code == 200

def test_child_resource():
    response = client.get("/api/v1/test")
    assert response.status_code == 200