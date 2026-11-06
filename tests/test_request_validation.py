"""Tests for drawing validation on the recognition endpoints.

The model and knowledge-base assets are not in this repo, so the classes that
load them are replaced with stubs. The transforms themselves are real.

Run from the repo root with: python -m pytest tests
"""

import inspect
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _StubCalc:
    def __init__(self, *args):
        pass

    def get_result(self, *transforms):
        return [{"class": 0, "value": 1.0}]


class _StubExplainer:
    def __init__(self, *args):
        pass

    def explain_feature(self, image):
        return {"prediction": 0, "pixels": len(image)}

    def explain_both(self, image):
        return {"prediction": 0, "pixels": len(image)}


sys.modules["calculate_sample"] = types.SimpleNamespace(CalcObject=_StubCalc)
sys.modules["case_explainer_service"] = types.SimpleNamespace(
    CaseExplainerService=_StubExplainer
)

import main_api  # noqa: E402
from request_validation import IMAGE_PIXELS, MAX_BODY_BYTES  # noqa: E402

ROUTES = ["/submit", "/submit_pbe", "/submit_cbe", "/submit_similarity_compare"]
HEADERS = {"Content-Type": "application/json"}

# A filled square, so the centering and thresholding steps have a real shape.
DRAWING = [
    255 if 8 <= row < 20 and 8 <= col < 20 else 0
    for row in range(28)
    for col in range(28)
]


@pytest.fixture(scope="module")
def client():
    return TestClient(main_api.app)


@pytest.mark.parametrize("route", ROUTES)
def test_valid_drawing_is_accepted(client, route):
    response = client.post(route, content=json.dumps(DRAWING), headers=HEADERS)
    assert response.status_code == 200


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"{}", id="object"),
        pytest.param(b'"text"', id="string"),
        pytest.param(json.dumps([0] * (IMAGE_PIXELS - 1)).encode(), id="too-few"),
        pytest.param(json.dumps([0] * (IMAGE_PIXELS + 1)).encode(), id="too-many"),
        pytest.param(json.dumps([256] + [0] * 783).encode(), id="above-range"),
        pytest.param(json.dumps([-1] + [0] * 783).encode(), id="below-range"),
        pytest.param(json.dumps([1.5] + [0] * 783).encode(), id="float"),
        pytest.param(json.dumps([True] + [0] * 783).encode(), id="bool"),
        pytest.param(json.dumps(["1"] + [0] * 783).encode(), id="string-pixel"),
        pytest.param(json.dumps([None] + [0] * 783).encode(), id="null-pixel"),
        pytest.param(b"[NaN" + b",0" * 783 + b"]", id="nan"),
        pytest.param(b"[" * 4000, id="deep-nesting"),
        pytest.param(b"\xff\xfe", id="bad-encoding"),
    ],
)
def test_invalid_body_is_rejected(client, route, body):
    response = client.post(route, content=body, headers=HEADERS)
    assert response.status_code == 400


@pytest.mark.parametrize("route", ROUTES)
def test_wrong_content_type_is_rejected(client, route):
    response = client.post(
        route, content=json.dumps(DRAWING), headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415


@pytest.mark.parametrize("route", ROUTES)
def test_oversized_body_is_rejected(client, route):
    body = json.dumps([0] * 640000)
    response = client.post(route, content=body, headers=HEADERS)
    assert response.status_code == 413


def test_oversized_body_without_content_length_is_rejected(client):
    def chunks():
        for _ in range(100):
            yield b"0," * 1024

    response = client.post("/submit", content=chunks(), headers=HEADERS)
    assert response.status_code == 413


def test_body_at_limit_is_accepted(client):
    padding = b" " * (MAX_BODY_BYTES - len(json.dumps(DRAWING)))
    body = json.dumps(DRAWING).encode() + padding
    assert len(body) == MAX_BODY_BYTES
    response = client.post("/submit", content=body, headers=HEADERS)
    assert response.status_code == 200


def test_handlers_do_not_block_the_event_loop():
    for route in main_api.app.routes:
        if getattr(route, "path", None) in ROUTES:
            assert not inspect.iscoroutinefunction(route.endpoint), route.path
