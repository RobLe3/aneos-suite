"""
Phase 18 tests — Menu Gap Analysis: Bug Fixes + Missing Features

Covers:
  TestPersistenceCorrectness  (3)  — import path, detection dict, impact dict
  TestShowDbResults           (1)  — dict .get() access
  TestBatchPrintReplaced      (2)  — show_info / show_error instead of print
  TestVerboseEvidenceBreakdown(2)  — wrapper includes breakdown; display uses it
  TestHorizonsEpochAccumulation(2) — 2 epochs → 2 rows; each row has a/e/i/epoch
  TestFormatSigmaInconclusive (2)  — sigma<2 → INCONCLUSIVE; sigma=2 → INTERESTING
  TestMissingFeatures         (3)  — _start_api_server, _detection_analytics, _show_help exist
"""

import inspect
import textwrap
import sys
import os
import types
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_menu():
    """Return an ANEOSMenuV2 instance with console disabled."""
    from aneos_menu_v2 import ANEOSMenuV2
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu._detection_results = {}
    menu._impact_results = {}
    return menu


# ===========================================================================
# 1. Persistence Correctness
# ===========================================================================

class TestPersistenceCorrectness:

    def test_detection_persist_imports_from_database(self):
        """_persist_detection_result_async must import from aneos_api.database, not services."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._persist_detection_result_async)
        assert "aneos_api.database" in source
        assert "aneos_api.services" not in source

    def test_detection_result_dict_has_designation(self):
        """result_data passed to save_analysis_result must include 'designation' key."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._persist_detection_result_async)
        assert '"designation"' in source or "'designation'" in source

    def test_impact_result_dict_has_designation(self):
        """impact result_data must include 'designation' key."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._persist_impact_result_async)
        assert '"designation"' in source or "'designation'" in source
        assert "aneos_api.services" not in source


# ===========================================================================
# 2. _show_db_results — dict .get() access
# ===========================================================================

class TestShowDbResults:

    def test_db_results_uses_get_not_getattr(self):
        """_show_db_results must use dict .get(), not getattr, since service returns List[Dict]."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._show_db_results)
        # Must contain .get( calls for designation / classification / analysis_date
        assert 'r.get("designation"' in source or "r.get('designation'" in source
        assert 'r.get("classification"' in source or "r.get('classification'" in source
        assert 'r.get("analysis_date"' in source or "r.get('analysis_date'" in source
        # Must NOT use getattr for these fields
        assert 'getattr(r, "designation"' not in source
        assert 'getattr(r, "result_type"' not in source
        assert 'getattr(r, "created_at"' not in source


# ===========================================================================
# 3. Batch print() replacement
# ===========================================================================

class TestBatchPrintReplaced:

    def test_batch_skip_uses_show_info(self):
        """Skip message must use self.show_info, not print()."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._detect_batch)
        # The "no orbital data" message should call show_info
        assert "show_info" in source
        # Should NOT be a bare print() for the skip message
        assert 'print(f"  ⚠' not in source

    def test_batch_error_uses_show_error(self):
        """Error message in batch loop must use self.show_error, not print()."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._detect_batch)
        assert "show_error" in source
        assert 'print(f"  ❌' not in source


# ===========================================================================
# 4. Verbose Evidence Breakdown
# ===========================================================================

class TestVerboseEvidenceBreakdown:

    def test_wrapper_adds_evidence_breakdown_to_analysis(self):
        """ValidatedWrapper.analyze_neo must include 'evidence_breakdown' in analysis dict."""
        import aneos_core.detection.detection_manager as dm
        source = inspect.getsource(dm)
        assert "evidence_breakdown" in source

    def test_display_reads_evidence_breakdown_key(self):
        """_display_detection_result must look for 'evidence_breakdown' key in analysis."""
        import aneos_menu_v2
        source = inspect.getsource(aneos_menu_v2.ANEOSMenuV2._display_detection_result)
        assert "evidence_breakdown" in source


# ===========================================================================
# 5. Horizons epoch accumulation
# ===========================================================================

class TestHorizonsEpochAccumulation:

    def _make_two_epoch_text(self) -> str:
        """Simulate Horizons ELEMENTS output with two epochs, each on three lines."""
        return textwrap.dedent("""\
            $$SOE
            2020-Jan-01 00:00:00.0000 TDB
             EC= 1.915000E-01 QR= 7.458000E-01 TP= 2458900.5
             OM= 2.040000E+02  W= 1.266000E+02  IN= 3.336751E+00
             A=  9.225000E-01 MA= 1.273225E+02
            2021-Jan-01 00:00:00.0000 TDB
             EC= 1.916000E-01 QR= 7.459000E-01 TP= 2459200.5
             OM= 2.041000E+02  W= 1.267000E+02  IN= 3.337000E+00
             A=  9.226000E-01 MA= 1.274000E+02
            $$EOE
        """)

    def test_two_epochs_produce_two_rows(self):
        """Two-epoch input must yield exactly 2 rows (not 6 partial records)."""
        from aneos_core.data.sources.horizons import HorizonsSource
        src = HorizonsSource.__new__(HorizonsSource)
        rows = src._parse_elements_table(self._make_two_epoch_text())
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}: {rows}"

    def test_each_row_has_orbital_elements_and_epoch(self):
        """Each parsed row must contain a/e/i/epoch."""
        from aneos_core.data.sources.horizons import HorizonsSource
        src = HorizonsSource.__new__(HorizonsSource)
        rows = src._parse_elements_table(self._make_two_epoch_text())
        for row in rows:
            assert "a" in row, f"Missing 'a' in row: {row}"
            assert "e" in row, f"Missing 'e' in row: {row}"
            assert "i" in row, f"Missing 'i' in row: {row}"
            assert "epoch" in row, f"Missing 'epoch' in row: {row}"


# ===========================================================================
# 6. format_sigma — INCONCLUSIVE for σ<2
# ===========================================================================

class TestFormatSigmaInconclusive:

    def test_sigma_below_2_returns_inconclusive(self):
        """σ=1.5 must return 'INCONCLUSIVE (σ<2)', not NOTABLE or ROUTINE."""
        menu = _make_menu()
        label = menu.format_sigma(1.5)
        assert "INCONCLUSIVE" in label
        assert "NOTABLE" not in label
        assert "ROUTINE" not in label

    def test_sigma_exactly_2_returns_interesting(self):
        """σ=2.0 must return 'INTERESTING (σ≥2)'."""
        menu = _make_menu()
        label = menu.format_sigma(2.0)
        assert "INTERESTING" in label


# ===========================================================================
# 7. Missing features exist as methods
# ===========================================================================

class TestMissingFeatures:

    def test_start_api_server_method_exists(self):
        from aneos_menu_v2 import ANEOSMenuV2
        assert callable(getattr(ANEOSMenuV2, "_start_api_server", None))

    def test_detection_analytics_method_exists(self):
        from aneos_menu_v2 import ANEOSMenuV2
        assert callable(getattr(ANEOSMenuV2, "_detection_analytics", None))

    def test_show_help_method_exists(self):
        from aneos_menu_v2 import ANEOSMenuV2
        assert callable(getattr(ANEOSMenuV2, "_show_help", None))
