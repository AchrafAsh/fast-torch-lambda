from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_main_resource():
    response = client.get("/")
    assert response.status_code == 200

def test_child_resource():
    response = client.get("/api/v1/test")
    assert response.status_code == 200

def test_translate_resource():
    sentence = "Bonjour le monde!"
    response = client.post("/api/v1/translate", json={'sentence': sentence})
    assert response.status_code == 200
    res = response.json()
    
    assert "fr" in res.keys()
    assert "en" in res.keys()
    assert res["fr"] == sentence
    assert type(res["en"]) == str