from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_schema_and_recommendations() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hiring a senior Java backend developer with Spring and SQL"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"]
    assert {"name", "url", "test_type"} <= set(payload["recommendations"][0])
    assert "Java" in " ".join(item["name"] for item in payload["recommendations"])


def test_clarification_when_missing_signal() -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Can you help me?"}]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "role" in payload["reply"].lower()


def test_refusal_for_legal_advice() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Does this SHL test satisfy legal HIPAA hiring requirements?"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert "legal" in payload["reply"].lower()
