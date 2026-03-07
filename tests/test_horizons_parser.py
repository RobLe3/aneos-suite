"""Unit test for _fetch_from_horizons element parser (no network required)."""
from unittest.mock import patch, MagicMock
from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
import logging

_SAMPLE_RESULT = """
$$SOE
2458849.500000000 = A.D. 2020-Jan-01 00:00:00.0000 TDB
 EC= 5.552003124456866E-01 QR= 4.481977069034823E-01 IN= 9.180426018781726E+01
 OM= 4.679244527993226E+01 W = 1.769695476793039E+02 Tp=  2458840.553734498937
 N = 7.165898831665516E-01 MA= 6.360781640547069E+00 TA= 2.013521505226268E+01
 A = 1.002248618219024E+00 AD= 1.556299529534565E+00 PR= 5.024055849499397E+02
$$EOE
"""


def test_parser_extracts_keplerian_elements():
    builder = GroundTruthDatasetBuilder.__new__(GroundTruthDatasetBuilder)
    builder.logger = logging.getLogger(__name__)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": _SAMPLE_RESULT}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp):
        result = builder._fetch_from_horizons("-31", "Voyager 1")
    assert result is not None, "Parser returned None — regex fix not applied"
    oe = result["orbital_elements"]
    assert abs(oe["a"] - 1.002248618219024) < 1e-6, f"a={oe['a']}"
    assert abs(oe["e"] - 0.5552003124456866) < 1e-6, f"e={oe['e']}"
    assert abs(oe["i"] - 91.80426018781726) < 1e-4, f"i={oe['i']}"


def test_parser_returns_none_on_empty_block():
    builder = GroundTruthDatasetBuilder.__new__(GroundTruthDatasetBuilder)
    builder.logger = logging.getLogger(__name__)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": "No ephemeris data available."}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp):
        result = builder._fetch_from_horizons("-31", "Voyager 1")
    assert result is None


_HYPERBOLIC_RESULT = """
$$SOE
2458849.500000000 = A.D. 2020-Jan-01 00:00:00.0000 TDB
 EC= 3.728528088688059E+00 QR= 2.315782669753419E+01 IN= 3.541890697407429E+01
 OM= 1.740398375143797E+02 W = 3.583419872040095E+01 Tp=  2448046.373789682984
 N = 4.127027695826097E-04 MA= 1.818832041987019E+03 TA= 1.791034226519895E+02
 A =-8.791855249834960E+00 AD= 9.999999999999998E+99 PR= 9.999999999999998E+99
$$EOE
"""


def test_parser_handles_negative_semi_major_axis():
    """Voyager/Pioneer style hyperbolic orbit — A is negative."""
    builder = GroundTruthDatasetBuilder.__new__(GroundTruthDatasetBuilder)
    builder.logger = logging.getLogger(__name__)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": _HYPERBOLIC_RESULT}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp):
        result = builder._fetch_from_horizons("-31", "Voyager 1")
    assert result is not None, "Parser returned None for hyperbolic orbit"
    oe = result["orbital_elements"]
    assert oe["a"] < 0, f"Expected negative a for hyperbolic orbit, got {oe['a']}"
    assert abs(oe["a"] - (-8.791855249834960)) < 1e-6, f"a={oe['a']}"


def test_hyperbolic_elements_accepted_by_orbital_elements():
    """OrbitalElements must store hyperbolic (e>1, a<0) parameters without raising."""
    from aneos_core.data.models import OrbitalElements
    oe = OrbitalElements(
        semi_major_axis=-8.791855249834960,
        eccentricity=3.728528088688059,
        inclination=35.41890697407429,
        ascending_node=174.0398375143797,
        argument_of_perihelion=35.83419872040095,
        mean_anomaly=1818.832041987019,
    )
    assert oe.semi_major_axis < 0, "hyperbolic a should be negative"
    assert oe.eccentricity > 1,    "hyperbolic e should exceed 1"
