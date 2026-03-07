"""Verify chunk boundary objects are neither missed nor duplicated."""
from aneos_core.polling.historical_chunked_poller import HistoricalChunkedPoller


def _obj(desig, date):
    return {"designation": desig, "close_approach_date": date}


def test_boundary_no_missed_no_duplicate():
    chunk1 = [_obj("A", "2015-12-31"), _obj("C", "2016-01-04")]
    chunk2 = [_obj("B", "2016-01-01"), _obj("C", "2016-01-04")]
    poller = HistoricalChunkedPoller.__new__(HistoricalChunkedPoller)
    merged = poller._merge_chunks([chunk1, chunk2])
    designations = [o["designation"] for o in merged]
    assert "A" in designations
    assert "B" in designations
    assert "C" in designations
    assert designations.count("C") == 1, f"Duplicate C: {designations}"
