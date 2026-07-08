import json

from tastepack.config import TastepackConfig


def test_default_model_is_gemini_3_5_flash():
    assert TastepackConfig().gemini_model == "gemini-3.5-flash"


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
