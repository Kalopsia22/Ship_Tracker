import json

import pytest

from maritime.alerts import AlertCandidate
from maritime.config import Settings
from maritime.database import Repository, make_engine
from maritime.security import Authenticator, LoginThrottle, Role, User, has_role, hash_password, load_users, verify_password

FAST = 1000  # PBKDF2 iterations: fast for tests only


def test_hash_verify_and_salting():
    h1, h2 = hash_password("correct horse", FAST), hash_password("correct horse", FAST)
    assert h1 != h2 and verify_password("correct horse", h1) and not verify_password("wrong horse!", h1)
    assert not verify_password("x", "garbage") and not verify_password("x", "md5$1$a$b")
    with pytest.raises(ValueError):
        hash_password("short", FAST)


def test_role_ordering():
    assert has_role(Role.ADMIN, Role.ANALYST) and has_role(Role.ANALYST, Role.ANALYST) and not has_role(Role.VIEWER, Role.ANALYST)
    assert Role.parse(" Analyst ") is Role.ANALYST
    with pytest.raises(ValueError):
        Role.parse("root")


def _auth(max_attempts=3, clock=None):
    users = {"alice": User("alice", Role.ANALYST, hash_password("alice-password", FAST))}
    return Authenticator(users, LoginThrottle(max_attempts, 60, clock=clock or (lambda: 0.0)))


def test_authenticate_success_and_uniform_failure_message():
    a = _auth()
    assert a.authenticate("ALICE", "alice-password")[0].role is Role.ANALYST
    m1 = a.authenticate("alice", "nope")[1]
    m2 = a.authenticate("ghost", "nope")[1]
    assert m1 == m2 == "Invalid username or password."


def test_lockout_and_expiry():
    t = [0.0]
    a = _auth(3, clock=lambda: t[0])
    for _ in range(3):
        a.authenticate("alice", "bad")
    assert "Too many" in a.authenticate("alice", "alice-password")[1]  # locked even with right password
    t[0] = 120.0
    assert a.authenticate("alice", "alice-password")[0] is not None


def test_strict_mode_requires_users_and_rejects_plaintext():
    with pytest.raises(RuntimeError, match="requires AUTH_USERS_JSON"):
        load_users(Settings(auth_mode="strict"))
    with pytest.raises(ValueError, match="never a plaintext"):
        load_users(Settings(auth_mode="strict", auth_users_json=json.dumps({"bob": {"password_hash": "hunter22", "role": "admin"}})))
    ok = load_users(Settings(auth_mode="strict", auth_users_json=json.dumps({"bob": {"password_hash": hash_password("hunter2222", FAST), "role": "admin"}})))
    assert ok["bob"].role is Role.ADMIN


def test_demo_auth_cannot_be_combined_with_real_data(monkeypatch):
    monkeypatch.setenv("AIS_DATA_MODE", "csv"); monkeypatch.setenv("AUTH_MODE", "demo")
    with pytest.raises(ValueError, match="only allowed with"):
        Settings.from_env()
    monkeypatch.delenv("AUTH_MODE")
    assert Settings.from_env().auth_mode == "strict"  # real data defaults to strict


@pytest.fixture
def repo(tmp_path):
    return Repository(make_engine(f"sqlite:///{tmp_path / 'a.db'}"))


def _c(subject="1", sev="Warning", msg="m", t="AIS_GAP"):
    return AlertCandidate(t, sev, subject, msg, {"k": 1})


def test_alert_sync_create_update_resolve(repo):
    assert repo.sync_alerts([_c("1"), _c("2")]) == {"created": 2, "updated": 0, "resolved": 0}
    assert repo.sync_alerts([_c("1", msg="changed")]) == {"created": 0, "updated": 1, "resolved": 1}
    open_ = repo.list_alerts(("open",))
    assert [a["subject"] for a in open_] == ["1"] and open_[0]["message"] == "changed"
    assert len(repo.list_alerts(("resolved",))) == 1


def test_acknowledge_workflow_is_audited_and_idempotent(repo):
    repo.sync_alerts([_c("1")])
    aid = repo.list_alerts(("open",))[0]["id"]
    assert repo.acknowledge(aid, "alice") is True and repo.acknowledge(aid, "bob") is False
    a = repo.list_alerts(("acknowledged",))[0]
    assert a["ack_by"] == "alice"
    assert any(e["action"] == "alert.acknowledge" and e["user"] == "alice" for e in repo.audit_log())
    # an acknowledged alert whose condition persists is not re-raised as a duplicate
    assert repo.sync_alerts([_c("1")])["created"] == 0
