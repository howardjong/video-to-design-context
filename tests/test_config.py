import json

import pytest

from tastepack.config import TastepackConfig


def test_default_model_is_gemini_3_5_flash():
    config = TastepackConfig()

    assert config.gemini_model == "gemini-3.5-flash"
    assert config.max_duration_seconds == 1800
    assert config.max_file_size_bytes == 2_147_483_648
    assert config.allow_no_audio is False
    assert config.gemini_max_retries == 3
    assert config.gemini_retry_base_delay_seconds == 1.0
    assert config.cleanup_uploaded_files is True


def test_default_config_has_separate_gemini_operation_timeouts():
    config = TastepackConfig()

    assert getattr(config, "gemini_upload_timeout_seconds", None) == 600
    assert getattr(config, "gemini_file_processing_timeout_seconds", None) == 600
    assert getattr(config, "gemini_generation_timeout_seconds", None) == 300
    assert getattr(config, "gemini_cleanup_timeout_seconds", None) == 30


def test_legacy_request_timeout_configures_file_processing_timeout():
    config = TastepackConfig.model_validate({"request_timeout_seconds": 42})

    assert config.gemini_file_processing_timeout_seconds == 42


def test_config_values_can_be_loaded_from_file_and_cli_overrides(tmp_path):
    config_path = tmp_path / "tastepack.json"
    config_path.write_text(
        json.dumps(
            {
                "gemini_model": "gemini-2.5-pro",
                "frame_confidence_threshold": 0.75,
                "max_frames_per_asset": 4,
                "produce_pdf": False,
            }
        )
    )

    config = TastepackConfig.from_sources(
        config_path=config_path,
        overrides={"max_frames_per_asset": 2, "verbosity": "debug"},
    )

    assert config.gemini_model == "gemini-2.5-pro"
    assert config.frame_confidence_threshold == 0.75
    assert config.max_frames_per_asset == 2
    assert config.produce_pdf is False
    assert config.verbosity == "debug"


def test_qa_requires_a_configured_model_when_enabled():
    with pytest.raises(ValueError, match="qa_model"):
        TastepackConfig(qa_enabled=True)

    config = TastepackConfig(qa_enabled=True, qa_model="claude-for-qa")

    assert config.qa_mode == "warn"
    assert config.qa_coverage_interval_seconds == 3
