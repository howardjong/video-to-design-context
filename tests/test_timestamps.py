import pytest

from tastepack.timestamps import TimestampError, normalize_timestamp


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12", 12.0),
        ("00:12", 12.0),
        ("00:00:12", 12.0),
        ("00:00:12.500", 12.5),
        (12, 12.0),
        (12.5, 12.5),
    ],
)
def test_normalize_timestamp_accepts_common_formats(raw, expected):
    assert normalize_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "abc", "00:xx:12", "-1", "NaN", "inf", float("nan"), float("inf")],
)
def test_normalize_timestamp_rejects_invalid_values(raw):
    with pytest.raises(TimestampError):
        normalize_timestamp(raw)
