"""Self-check for the BCB SGS client. Run: python tests/test_bcb_client.py
The parsing test is hermetic (stubbed HTTP) and runs in CI; the live one is local.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bcb_client


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_parse_and_retry():
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        if len(calls) == 1:
            raise bcb_client.requests.ConnectionError("boom")
        return _Resp([{"data": "01/03/1995", "valor": "4.26"}, {"data": "01/04/1995", "valor": ""}])

    real_get, real_sleep = bcb_client.requests.get, bcb_client.time.sleep
    bcb_client.requests.get, bcb_client.time.sleep = fake_get, lambda s: None
    try:
        assert bcb_client.fetch_series(4390) == [("199503", 4.26)]  # empty value dropped, 1 retry used
    finally:
        bcb_client.requests.get, bcb_client.time.sleep = real_get, real_sleep
    assert len(calls) == 2


def test_fetch_live():
    for s in bcb_client.SERIES.values():
        points = bcb_client.fetch_series(s["code"])
        assert len(points) > 300 and points[0][0] == "199501", s
        assert len(points[-1][0]) == 6


if __name__ == "__main__":
    test_parse_and_retry()
    print("test_parse_and_retry: ok")
    if "--live" in sys.argv:
        test_fetch_live()
        print("test_fetch_live: ok")
