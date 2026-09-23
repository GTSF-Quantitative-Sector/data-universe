import pytest

from data_universe.options.iv30 import iv30


def test_iv30_raises_not_implemented_naming_its_owner():
    with pytest.raises(NotImplementedError) as exc_info:
        iv30("AAPL", "2024-01-02")
    message = str(exc_info.value)
    assert "vrp-research" in message
    assert "Lionel" in message
