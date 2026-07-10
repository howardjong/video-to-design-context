from tastepack.providers import FUTURE_SYNTHESIS_EVALUATION


def test_future_synthesis_evaluation_keeps_gemini_as_the_primary_video_analyzer():
    protocol = FUTURE_SYNTHESIS_EVALUATION

    assert protocol.primary_video_provider == "gemini-3.5-flash"
    assert protocol.candidate_synthesis_provider == "gpt-5.6"
    assert "frame-to-moment traceability" in protocol.required_uat_metrics
    assert "UAT" in protocol.promotion_rule
