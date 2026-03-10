"""
Phase 24 tests — runtime bug fixes, input validation, endpoint wiring.

Covers:
  24B  NEODyS provisional designation misparse fix
  24C  Gaia stdout suppression during module import
  24D  Pydantic OrbitalInput / AnalysisRequest / BatchAnalysisRequest validation
"""

import io
import sys
import pytest


# ---------------------------------------------------------------------------
# 24B — NEODyS provisional designation fix
# ---------------------------------------------------------------------------

class TestNEODySResolveNumber:
    """_resolve_number must not treat provisional year as catalogue number."""

    def _get_source(self):
        from aneos_core.data.sources.neodys import NEODySSource
        return NEODySSource()

    def test_provisional_1998ky26_not_misparse(self):
        src = self._get_source()
        result = src._resolve_number("1998 KY26")
        # Must NOT return 1998 (the year)
        assert result != 1998, f"Still returning year: {result}"

    def test_provisional_2004mn4_not_misparse(self):
        src = self._get_source()
        result = src._resolve_number("2004 MN4")
        # Must NOT return 2004 (the year)
        assert result != 2004, f"Still returning year: {result}"

    def test_numbered_with_name_resolves(self):
        src = self._get_source()
        result = src._resolve_number("99942 Apophis")
        assert result == 99942

    def test_plain_number_resolves(self):
        src = self._get_source()
        result = src._resolve_number("99942")
        assert result == 99942

    def test_plain_number_with_spaces_resolves(self):
        src = self._get_source()
        result = src._resolve_number("  101955  ")
        assert result == 101955


# ---------------------------------------------------------------------------
# 24C — Gaia stdout suppression
# ---------------------------------------------------------------------------

def test_gaia_import_no_stdout():
    """Importing gaia_astrometric_calibration must not leak anything to stdout."""
    # Remove cached module so the import triggers fresh
    mods_to_remove = [
        k for k in sys.modules
        if "gaia_astrometric_calibration" in k
    ]
    for m in mods_to_remove:
        del sys.modules[m]

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        import aneos_core.validation.gaia_astrometric_calibration  # noqa: F401
    finally:
        sys.stdout = old_stdout

    leaked = buf.getvalue()
    assert not leaked, f"Gaia import leaked to stdout: {leaked[:200]!r}"


# ---------------------------------------------------------------------------
# 24D — OrbitalInput Pydantic validation
# ---------------------------------------------------------------------------

@pytest.fixture
def OrbitalInput():
    from aneos_api.schemas.detection import OrbitalInput as _OI
    return _OI


def test_orbital_input_rejects_invalid_eccentricity(OrbitalInput):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OrbitalInput(a=1.5, e=3.0, i=5.0)


def test_orbital_input_rejects_negative_sma(OrbitalInput):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OrbitalInput(a=-1.0, e=0.3, i=5.0)


def test_orbital_input_rejects_bad_inclination(OrbitalInput):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OrbitalInput(a=1.5, e=0.3, i=200.0)


def test_orbital_input_accepts_valid_values(OrbitalInput):
    obj = OrbitalInput(a=1.5, e=0.3, i=5.0)
    assert obj.a == 1.5
    assert obj.e == 0.3
    assert obj.i == 5.0


def test_orbital_input_accepts_hyperbolic(OrbitalInput):
    """Eccentricity up to 2.0 is allowed (hyperbolic)."""
    obj = OrbitalInput(a=10.0, e=1.5, i=90.0)
    assert obj.e == 1.5


# ---------------------------------------------------------------------------
# 24D — AnalysisRequest / BatchAnalysisRequest validation
# ---------------------------------------------------------------------------

@pytest.fixture
def AnalysisRequest():
    from aneos_api.models import AnalysisRequest as _AR
    return _AR


@pytest.fixture
def BatchAnalysisRequest():
    from aneos_api.models import BatchAnalysisRequest as _BAR
    return _BAR


def test_analysis_request_rejects_empty_designation(AnalysisRequest):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnalysisRequest(designation="")


def test_analysis_request_accepts_valid(AnalysisRequest):
    req = AnalysisRequest(designation="99942 Apophis")
    assert req.designation == "99942 Apophis"


def test_batch_rejects_long_designation(BatchAnalysisRequest):
    from pydantic import ValidationError
    long_desg = "A" * 51
    with pytest.raises(ValidationError):
        BatchAnalysisRequest(designations=[long_desg])


def test_batch_strips_whitespace(BatchAnalysisRequest):
    req = BatchAnalysisRequest(designations=["  99942 Apophis  "])
    assert req.designations[0] == "99942 Apophis"
