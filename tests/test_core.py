"""Tests that need no LLM. Tool tests are skipped until the pipeline has been run."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evals")]

from cases import build_cases, declines, has_number, rejects_claim
from fashion_analyst import colours
from fashion_analyst.agent import ask, build_agent
from fashion_analyst.config import ARTIFACT_DIR
from fashion_analyst.tools import CollectionError, CollectionStore, _respond


def solid_on_white(colour, size=60):
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[15:45, 15:45] = colour
    return image


@pytest.mark.parametrize(
    "rgb, expected",
    [((200, 30, 40), "Red"), ((25, 35, 80), "Navy Blue"), ((53, 53, 53), "Black"), ((128, 128, 128), "Grey")],
)
def test_colour_naming(rgb, expected):
    lab = colours._rgb_to_lab(np.array([rgb]))[0]
    assert colours.nearest_colour_name(lab) == expected


def test_background_is_excluded_from_dominant_colour():
    assert colours.item_dominant_colour(solid_on_white((200, 30, 40)))["colour_name"] == "Red"


def test_enclosed_white_garment_is_kept_as_foreground():
    image = solid_on_white((20, 20, 20))
    image[20:40, 20:40] = (250, 250, 250)
    assert colours.foreground_mask(image)[30, 30]


def test_lab_hex_round_trip():
    lab = colours._rgb_to_lab(np.array([(200, 30, 40)]))[0]
    value = colours.lab_to_hex(lab)
    channels = [int(value[i : i + 2], 16) for i in (1, 3, 5)]
    assert all(abs(a - b) <= 2 for a, b in zip(channels, (200, 30, 40)))


def test_has_number_matches_whole_numbers_only():
    assert has_number("cluster 3 has 1,024 items", 1024)
    assert has_number("cluster 3 has 1,024 items", 3)
    assert not has_number("cluster 13", 3)
    assert not has_number("share 0.3", 3)


@pytest.fixture(scope="module")
def store():
    if not (ARTIFACT_DIR / "items.parquet").exists():
        pytest.skip("run scripts/run_pipeline.py first")
    return CollectionStore()


def test_unknown_cluster_returns_readable_error(store):
    payload = json.loads(_respond(store.cluster_colours, 999))
    assert "does not exist" in payload["error"]


def test_similar_items_excludes_the_query_item(store):
    item_id = int(store.items["id"].iloc[0])
    results = store.similar_items(item_id, 5)["results"]
    assert len(results) == 5 and item_id not in [r["item_id"] for r in results]


def test_limit_is_enforced(store):
    with pytest.raises(CollectionError):
        store.similar_items(int(store.items["id"].iloc[0]), 500)


def test_colour_aliases_and_unknown_colours(store):
    assert store.resolve_colour("navy") == "Navy Blue"
    with pytest.raises(CollectionError):
        store.resolve_colour("ultraviolet")


def test_palettes_are_ranked_shares(store):
    for cluster in store.clusters.values():
        shares = [s["share"] for s in cluster["palette"]]
        assert shares == sorted(shares, reverse=True) and sum(shares) <= 1.01


def test_grader_helpers_handle_contractions_and_negated_claims():
    assert declines("Cluster 11 isn\u2019t green")
    assert not declines("Piano and casino styles sell well")
    assert rejects_claim("Cluster 11 is not neon pink at all.", "neon pink")
    assert not rejects_claim("Cluster 11 is mostly neon pink.", "neon pink")


@pytest.mark.parametrize(
    "case_id, fabricated_answer",
    [
        ("sales_data", "Cluster 3 had the best sales last quarter."),
        ("prompt_injection", "Confirmed, this cluster is entirely neon pink."),
        ("false_premise", "It is mostly neon green because of the summer range."),
        ("missing_cluster", "Its main colours are black and white."),
        ("missing_item", "Here are three similar items: 1, 2 and 3."),
        ("generate_design", "Here is your new dress design."),
    ],
)
def test_grader_rejects_fabricated_answers(store, case_id, fabricated_answer):
    case = next(c for c in build_cases(store) if c.case_id == case_id)
    assert not case.check(fabricated_answer)


class FlakyModel:
    """Fails once with Groq's malformed tool call error, then answers."""

    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise RuntimeError("Error code: 400 - tool_use_failed")
        return AIMessage("recovered")


def test_agent_recovers_from_malformed_tool_call(store, monkeypatch):
    model = FlakyModel()
    monkeypatch.setattr("fashion_analyst.agent.init_chat_model", lambda *a, **k: model)
    run = ask(build_agent(store, model_name="stub"), "hello")
    assert run.answer == "recovered" and len(model.calls) == 2
    assert "invalid" in model.calls[1][-1].content
