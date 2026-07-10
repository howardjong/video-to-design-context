from pathlib import Path


def test_readme_documents_install_usage_config_mocking_and_troubleshooting():
    readme = Path("README.md").read_text()

    required_phrases = [
        "uv run tastepack process input.mp4 --out ./claude-pack",
        "ffmpeg",
        "GEMINI_API_KEY",
        "--mock-gemini",
        "--allow-no-audio",
        "--max-duration-seconds",
        "--no-cleanup-uploaded-files",
        "--log-file",
        "strict preflight",
        "Troubleshooting",
        "taste_packet.pdf",
        "analysis.json",
        "untrusted source evidence",
        "source_sha256",
        "ffprobe_timeout_seconds",
        ".env.example",
    ]
    for phrase in required_phrases:
        assert phrase in readme
