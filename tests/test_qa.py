from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader

from tastepack.artifacts import (
    ArtifactGenerationError,
    generate_artifacts,
    validate_complete_metadata,
)
from tastepack.config import TastepackConfig
from tastepack.frames import ExtractedFrame, build_coverage_frames
from tastepack.gemini import MOCK_ANALYSIS
from tastepack.qa import (
    AuditResult,
    ClaimVerdict,
    EvidenceCitation,
    FrameInventory,
    MockQualityAuditProvider,
    QualityAuditError,
    SourceClaim,
    audit_existing_pack,
    audit_staged_pack,
    load_anthropic_api_key,
    validate_audit_result,
)
from tastepack.schema import TasteAnalysis


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), "white").save(path, format="JPEG")


def _make_qa_ready_pack(tmp_path: Path, *, produce_pdf: bool = False) -> Path:
    output_dir = tmp_path / "pack"
    source_transcript = tmp_path / "source-transcript.md"
    source_transcript.write_text("[00:00.000] I like the large heading.\n", encoding="utf-8")
    selected_frame_path = output_dir / "frames" / "asset-1_000012000.jpg"
    coverage_frame_path = output_dir / "evidence" / "coverage_frames" / "qa-coverage_000000000.jpg"
    _write_jpeg(selected_frame_path)
    _write_jpeg(coverage_frame_path)
    selected_frames = [
        ExtractedFrame(
            id="selected-frame",
            asset_id="asset-1",
            timestamp_seconds=12.0,
            relative_path="frames/asset-1_000012000.jpg",
            reason="Gemini selection",
            confidence=0.9,
        )
    ]
    coverage_frames = [
        ExtractedFrame(
            id="coverage-frame",
            asset_id="qa-coverage",
            timestamp_seconds=0.0,
            relative_path="evidence/coverage_frames/qa-coverage_000000000.jpg",
            reason="Independent QA coverage frame",
            confidence=1.0,
        )
    ]
    generate_artifacts(
        output_dir=output_dir,
        analysis=TasteAnalysis.model_validate(MOCK_ANALYSIS),
        extracted_frames=selected_frames,
        coverage_frames=coverage_frames,
        source_transcript_path=source_transcript,
        config=TastepackConfig(
            produce_pdf=produce_pdf,
            qa_enabled=True,
            qa_model="test-claude",
        ),
        source_video_name="input.mp4",
        source_video_metadata={"source_sha256": "source-hash"},
    )
    return output_dir


def test_coverage_frames_are_deterministic_independent_of_gemini_suggestions() -> None:
    frames = build_coverage_frames(7.2, TastepackConfig(qa_coverage_interval_seconds=3))

    assert [frame.timestamp_seconds for frame in frames] == [0.0, 3.0, 6.0, 7.1]
    assert {frame.asset_id for frame in frames} == {"qa-coverage"}
    assert all(frame.reason == "Independent QA coverage frame" for frame in frames)


def test_audit_preserves_exact_source_transcript_and_writes_reviewed_delivery(
    tmp_path: Path,
) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)

    result = audit_staged_pack(
        output_dir,
        TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
        provider=MockQualityAuditProvider(),
    )

    assert result.release_status == "pass"
    assert (output_dir / "qa" / "raw" / "analysis.gemini.json").is_file()
    assert (output_dir / "qa" / "raw" / "design_preferences.gemini.md").is_file()
    assert (output_dir / "qa" / "raw" / "taste_packet.gemini.md").is_file()
    assert (output_dir / "qa" / "audit.json").is_file()
    assert (output_dir / "qa" / "visual_inventory.json").is_file()
    assert (output_dir / "qa_report.md").is_file()
    assert (output_dir / "START_HERE.md").is_file()
    assert (output_dir / "evidence" / "source_transcript.md").read_text() == (
        "[00:00.000] I like the large heading.\n"
    )
    assert "## Corrections Log" in (output_dir / "design_preferences.md").read_text()
    assert "QA-Reviewed" in (output_dir / "taste_packet.md").read_text()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["qa"]["provider"] == "anthropic"
    assert metadata["qa"]["model"] == "test-claude"
    assert metadata["qa"]["evidence"]["source_transcript"]["sha256"]
    assert metadata["qa"]["evidence"]["coverage_frames"][0]["sha256"]
    with zipfile.ZipFile(output_dir / "taste_packet.zip") as archive:
        assert "evidence/source_transcript.md" in archive.namelist()
        assert "evidence/coverage_frames/qa-coverage_000000000.jpg" in archive.namelist()
        assert "qa/audit.json" in archive.namelist()
        assert "START_HERE.md" in archive.namelist()


def test_inventory_is_blind_to_gemini_claims_but_audit_receives_them(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)
    provider = MockQualityAuditProvider()

    audit_staged_pack(
        output_dir,
        TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
        provider=provider,
    )

    assert provider.inventory_requests
    assert provider.inventory_requests[0].claims == []
    assert provider.audit_requests
    assert provider.audit_requests[0].claims
    assert provider.audit_requests[0].source_transcript == "[00:00.000] I like the large heading.\n"


def test_audit_rejects_missing_claim_verdict_and_invalid_citations(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)

    class InvalidProvider(MockQualityAuditProvider):
        def audit(self, request):
            result = super().audit(request)
            result.claim_verdicts = result.claim_verdicts[:-1]
            result.corrected_preferences[0].evidence = [
                EvidenceCitation(kind="frame", frame_path="frames/missing.jpg", timestamp_seconds=0)
            ]
            return result

    with pytest.raises(QualityAuditError, match="verdict"):
        audit_staged_pack(
            output_dir,
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            provider=InvalidProvider(),
        )

    class InvalidCitationProvider(MockQualityAuditProvider):
        def audit(self, request):
            result = super().audit(request)
            result.corrected_preferences[0].evidence = [
                EvidenceCitation(kind="frame", frame_path="frames/missing.jpg", timestamp_seconds=0)
            ]
            return result

    with pytest.raises(QualityAuditError, match="unknown coverage frame"):
        audit_staged_pack(
            output_dir,
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            provider=InvalidCitationProvider(),
        )


def test_enforce_rejects_unresolved_hallucinations_before_pack_mutation(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)
    original_preferences = (output_dir / "design_preferences.md").read_bytes()

    class HallucinationProvider(MockQualityAuditProvider):
        def audit(self, request):
            result = super().audit(request)
            result.release_status = "fail"
            result.claim_verdicts[0].disposition = "hallucinated"
            return result

    with pytest.raises(QualityAuditError, match="enforce"):
        audit_staged_pack(
            output_dir,
            TastepackConfig(
                produce_pdf=False,
                qa_enabled=True,
                qa_model="test-claude",
                qa_mode="enforce",
            ),
            provider=HallucinationProvider(),
        )

    assert (output_dir / "design_preferences.md").read_bytes() == original_preferences
    assert not (output_dir / "qa" / "audit.json").exists()


def test_warn_promotes_a_valid_audit_that_reports_hallucinations(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)

    class HallucinationProvider(MockQualityAuditProvider):
        def audit(self, request):
            result = super().audit(request)
            result.release_status = "warn"
            result.claim_verdicts[0].disposition = "hallucinated"
            result.claim_verdicts[0].reason = (
                "No supplied frame or transcript evidence supports it."
            )
            result.claim_verdicts[0].evidence = []
            return result

    result = audit_staged_pack(
        output_dir,
        TastepackConfig(
            produce_pdf=False,
            qa_enabled=True,
            qa_model="test-claude",
            qa_mode="warn",
        ),
        provider=HallucinationProvider(),
    )

    assert result.release_status == "warn"
    assert (output_dir / "qa" / "audit.json").is_file()
    assert "hallucinated" in (output_dir / "qa_report.md").read_text()


def test_existing_pack_audit_is_atomic_and_never_uses_gemini(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)
    original_zip = (output_dir / "taste_packet.zip").read_bytes()

    class ProviderFailure(MockQualityAuditProvider):
        def inventory(self, request):
            raise QualityAuditError("provider unavailable")

    with pytest.raises(QualityAuditError, match="provider unavailable"):
        audit_existing_pack(
            output_dir,
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            provider=ProviderFailure(),
        )

    assert (output_dir / "taste_packet.zip").read_bytes() == original_zip
    assert not (output_dir / "qa" / "audit.json").exists()


def test_existing_pack_audit_promotes_a_complete_reviewed_copy_without_gemini(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)

    monkeypatch.setattr(
        "tastepack.gemini.analyze_video",
        lambda *_args, **_kwargs: pytest.fail("audit must never call Gemini"),
    )

    audit_existing_pack(
        output_dir,
        TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
        provider=MockQualityAuditProvider(),
    )

    assert (output_dir / "qa" / "audit.json").is_file()
    assert (output_dir / "taste_packet.zip").is_file()


def test_audit_regenerates_reviewed_pdf_and_packages_it(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path, produce_pdf=True)

    audit_staged_pack(
        output_dir,
        TastepackConfig(produce_pdf=True, qa_enabled=True, qa_model="test-claude"),
        provider=MockQualityAuditProvider(),
    )

    assert "QA-Reviewed Delivery" in "\n".join(
        page.extract_text() or "" for page in PdfReader(output_dir / "taste_packet.pdf").pages
    )
    with zipfile.ZipFile(output_dir / "taste_packet.zip") as archive:
        assert "taste_packet.pdf" in archive.namelist()
        archived_pdf = PdfReader(BytesIO(archive.read("taste_packet.pdf")))
        pdf_text = "\n".join(
            page.extract_text() or "" for page in archived_pdf.pages
        )
    assert "QA-Reviewed Delivery" in pdf_text


def test_anthropic_key_is_never_written_to_reviewed_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-secret")
    output_dir = _make_qa_ready_pack(tmp_path)

    audit_staged_pack(
        output_dir,
        TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
        provider=MockQualityAuditProvider(),
    )

    for path in output_dir.rglob("*"):
        if path.is_file():
            assert "anthropic-test-secret" not in path.read_text(errors="ignore")


def test_anthropic_key_loads_without_terminal_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=anthropic-test-secret\n", encoding="utf-8")

    assert load_anthropic_api_key(env_path) == "anthropic-test-secret"
    captured = capsys.readouterr()
    assert "anthropic-test-secret" not in captured.out
    assert "anthropic-test-secret" not in captured.err


def test_complete_metadata_detects_tampered_qa_coverage_evidence(tmp_path: Path) -> None:
    output_dir = _make_qa_ready_pack(tmp_path)
    audit_staged_pack(
        output_dir,
        TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
        provider=MockQualityAuditProvider(),
    )
    coverage_frame = next((output_dir / "evidence" / "coverage_frames").glob("*.jpg"))
    coverage_frame.write_bytes(b"tampered")

    with pytest.raises(ArtifactGenerationError, match="fingerprint"):
        validate_complete_metadata(output_dir)


def test_audit_result_schema_rejects_unknown_disposition() -> None:
    with pytest.raises(Exception, match="disposition"):
        ClaimVerdict.model_validate(
            {
                "claim_id": "claim-1",
                "disposition": "probably right",
                "reason": "No." ,
                "evidence": [],
            }
        )


def test_inventory_schema_requires_each_coverage_frame() -> None:
    with pytest.raises(Exception, match="relative_path"):
        FrameInventory.model_validate({"observations": [{"timestamp_seconds": 0}]})


def test_motion_claim_cannot_be_confirmed_from_static_frame_only() -> None:
    frame = ExtractedFrame(
        id="coverage-frame",
        asset_id="qa-coverage",
        timestamp_seconds=0.0,
        relative_path="evidence/coverage_frames/frame.jpg",
        reason="Independent QA coverage frame",
        confidence=1.0,
    )
    result = AuditResult.model_validate(
        {
            "release_status": "pass",
            "release_reason": "Incorrect static-only audit.",
            "claim_verdicts": [
                {
                    "claim_id": "motion-claim",
                    "disposition": "confirmed",
                    "reason": "Frame appears animated.",
                    "evidence": [
                        {
                            "kind": "frame",
                            "frame_path": frame.relative_path,
                            "timestamp_seconds": 0,
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(QualityAuditError, match="Motion or interaction"):
        validate_audit_result(
            result,
            [
                SourceClaim(
                    claim_id="motion-claim",
                    source="analysis.motion_details.animations[0]",
                    text="Smooth page sweep.",
                    requires_transcript=True,
                )
            ],
            [frame],
            "[00:00.000] The page sweeps in.\n",
        )
