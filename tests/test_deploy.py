"""Deployment-readiness checks (static: Dockerfile, config, HF Space template).

These do not require a container runtime; CI *also* builds and health-checks the
image in the docker-image job of .github/workflows/ci.yml.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_dockerfile_is_lean_and_serves_the_app() -> None:
    df = (REPO / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in df
    assert "requirements.txt" in df
    assert "streamlit run app/Home.py" in df
    assert "--server.address=0.0.0.0" in df
    assert "USER app" in df
    assert "groupadd" in df and "useradd" in df
    assert "_stcore/health" in df


def test_dockerfile_default_port_matches_hf_convention() -> None:
    df = (REPO / "Dockerfile").read_text()
    assert "ARG PORT=7860" in df
    assert "EXPOSE ${PORT}" in df


def test_streamlit_config_forces_production_sane_defaults() -> None:
    cfg = (REPO / ".streamlit" / "config.toml").read_text()
    assert 'address = "0.0.0.0"' in cfg
    assert "headless = true" in cfg
    assert "gatherUsageStats = false" in cfg
    assert "showErrorDetails = false" in cfg


def test_dockerignore_keeps_secrets_and_datasets_out() -> None:
    di = (REPO / ".dockerignore").read_text()
    for excluded in (".env", ".venv", "data", "notebooks", ".git"):
        assert excluded in di


def test_mood_cue_onnx_is_trackable_for_fresh_clone_builds() -> None:
    gi = (REPO / ".gitignore").read_text()
    assert "!models/face_emotion.onnx" in gi
    assert (REPO / "models" / "face_emotion.onnx").is_file()
    assert (REPO / "models" / "metadata.json").is_file()


def test_hf_space_readme_has_docker_metadata() -> None:
    hf = (REPO / "deploy" / "hf_space_README.md").read_text()
    assert "sdk: docker" in hf
    assert "app_port: 7860" in hf
    assert "GROQ_API_KEY" in hf


def test_ci_builds_docker_image_and_probes_health() -> None:
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "docker-image" in ci
    assert "docker/build-push-action" in ci
    assert "_stcore/health" in ci


def test_kubernetes_free_compose_wraps_the_image() -> None:
    compose = (REPO / "docker-compose.yml").read_text()
    assert "build: ." in compose
    assert '"8501:7860"' in compose
