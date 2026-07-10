from __future__ import annotations

from tastepack.logging import job_log_context, redact_secrets


def test_redaction_removes_authorization_and_key_bearing_values(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-secret")
    message = (
        "Authorization: Bearer header-secret "
        "x-goog-api-key=query-secret "
        "https://example.test/?key=url-secret "
        "env-secret"
    )

    redacted = redact_secrets(message)

    for secret in ("header-secret", "query-secret", "url-secret", "env-secret"):
        assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_job_log_context_is_scoped_to_the_active_job() -> None:
    with job_log_context("job-123"):
        assert redact_secrets("safe") == "safe"
