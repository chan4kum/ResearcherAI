from app.config import Settings
from app.main import create_app
from app.models.schemas import ErrorResponse
from app.models.security import Role, UserIdentity
from fastapi.testclient import TestClient


def test_unauthenticated_request_rejected(settings: Settings) -> None:
    """Verify request to protected endpoint without credentials returns 401 UNAUTHORIZED."""
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    data = response.json()
    err_res = ErrorResponse.model_validate(data)
    assert err_res.error.code == "UNAUTHORIZED"
    assert "credentials required" in err_res.error.message.lower()


def test_invalid_token_request_rejected(settings: Settings) -> None:
    """Verify request with non-existent token returns 401 UNAUTHORIZED."""
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer non-existent-token-xyz"},
    )
    assert response.status_code == 401
    data = response.json()
    err_res = ErrorResponse.model_validate(data)
    assert err_res.error.code == "UNAUTHORIZED"
    assert "invalid" in err_res.error.message.lower()


def test_authenticated_request_via_bearer_and_api_key(settings: Settings) -> None:
    """Verify valid credentials via Bearer token or X-API-Key return 200 OK with identity."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Bearer token auth
    res_bearer = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer user-token-secret-789"},
    )
    assert res_bearer.status_code == 200
    user_data = res_bearer.json()
    assert user_data["username"] == "regular_user"
    assert "user" in user_data["roles"]
    assert user_data["is_active"] is True

    # 2. X-API-Key auth
    res_api_key = client.get(
        "/api/v1/auth/me",
        headers={"X-API-Key": "admin-token-secret-123"},
    )
    assert res_api_key.status_code == 200
    admin_data = res_api_key.json()
    assert admin_data["username"] == "admin"
    assert "admin" in admin_data["roles"]


def test_unauthorized_role_rejected(settings: Settings) -> None:
    """Verify authenticated user with insufficient role is rejected with 403 FORBIDDEN."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # Regular user attempting admin-only action
    response = client.get(
        "/api/v1/auth/admin-only",
        headers={"Authorization": "Bearer user-token-secret-789"},
    )
    assert response.status_code == 403
    data = response.json()
    err_res = ErrorResponse.model_validate(data)
    assert err_res.error.code == "FORBIDDEN"
    assert "access forbidden" in err_res.error.message.lower()

    # Guest viewer attempting researcher-only action
    res_viewer = client.get(
        "/api/v1/auth/researcher-only",
        headers={"Authorization": "Bearer viewer-token-secret-000"},
    )
    assert res_viewer.status_code == 403
    err_viewer = ErrorResponse.model_validate(res_viewer.json())
    assert err_viewer.error.code == "FORBIDDEN"


def test_authorized_role_admin_access(settings: Settings) -> None:
    """Verify administrator role has access to admin and researcher endpoints."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # Admin accessing admin endpoint
    res_admin = client.get(
        "/api/v1/auth/admin-only",
        headers={"Authorization": "Bearer admin-token-secret-123"},
    )
    assert res_admin.status_code == 200
    assert res_admin.json()["status"] == "authorized"

    # Admin accessing researcher endpoint
    res_res = client.get(
        "/api/v1/auth/researcher-only",
        headers={"Authorization": "Bearer admin-token-secret-123"},
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "authorized"


def test_authorized_role_researcher_access(settings: Settings) -> None:
    """Verify researcher role has access to researcher endpoint but not admin endpoint."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # Researcher accessing researcher endpoint
    res_res = client.get(
        "/api/v1/auth/researcher-only",
        headers={"Authorization": "Bearer researcher-token-secret-456"},
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "authorized"

    # Researcher accessing admin endpoint -> 403
    res_admin = client.get(
        "/api/v1/auth/admin-only",
        headers={"Authorization": "Bearer researcher-token-secret-456"},
    )
    assert res_admin.status_code == 403


def test_user_identity_role_evaluation_helpers() -> None:
    """Unit tests verifying UserIdentity role evaluation logic."""
    admin_user = UserIdentity(
        user_id="u1",
        username="admin",
        roles=[Role.ADMIN],
    )
    assert admin_user.has_role(Role.ADMIN) is True
    assert admin_user.has_role(Role.RESEARCHER) is True  # Admin superuser
    assert admin_user.has_any_role(Role.USER, Role.VIEWER) is True

    standard_user = UserIdentity(
        user_id="u2",
        username="john",
        roles=[Role.USER],
    )
    assert standard_user.has_role(Role.USER) is True
    assert standard_user.has_role(Role.ADMIN) is False
    assert standard_user.has_role(Role.RESEARCHER) is False
    assert standard_user.has_any_role(Role.RESEARCHER, Role.USER) is True
    assert standard_user.has_any_role(Role.ADMIN, Role.RESEARCHER) is False
