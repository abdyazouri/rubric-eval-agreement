"""Rubric definition for the demo domain: customer-support reply quality.

The domain here is deliberately generic (support-reply quality, not any
proprietary scoring system). Each dimension is scored on a 0-4 scale against
written behavioral anchors -- the same style of anchor-based rubric used in
well-known rubric-scoring literature (e.g. ASAP-AES style scoring guides).
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCALE_MIN = 0
SCALE_MAX = 4
NUM_CATEGORIES = SCALE_MAX - SCALE_MIN + 1  # 5


@dataclass(frozen=True)
class Dimension:
    name: str
    anchors: dict[int, str]  # score -> written behavioral anchor


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="accuracy",
        anchors={
            0: "Contains factually incorrect information stated as fact.",
            1: "Largely inaccurate, or omits information critical to being correct.",
            2: "Partially accurate; has a notable gap or an ambiguous claim.",
            3: "Accurate with only a minor omission that does not affect correctness.",
            4: "Fully accurate and complete; nothing material is missing or wrong.",
        },
    ),
    Dimension(
        name="clarity",
        anchors={
            0: "Disorganized or contradictory; the reader cannot follow it.",
            1: "Hard to follow; requires rereading to extract the point.",
            2: "Understandable but has an awkward structure or an unclear step.",
            3: "Clear, with only a minor wording issue.",
            4: "Clear and well-organized; the main point is immediately evident.",
        },
    ),
    Dimension(
        name="empathy",
        anchors={
            0: "Dismissive or blames the customer for the problem.",
            1: "Purely transactional; ignores an explicitly stated frustration.",
            2: "Neutral in tone; acknowledges the request but not the customer's state.",
            3: "Acknowledges the customer's situation in a brief, genuine way.",
            4: "Explicitly validates the customer's situation and sets expectations kindly.",
        },
    ),
    Dimension(
        name="actionability",
        anchors={
            0: "Gives the customer nothing they can do next.",
            1: "Vague next step with no owner or timeframe.",
            2: "A next step is given, but a dependency is left unclear.",
            3: "A concrete next step with an owner or timeframe, minor detail missing.",
            4: "A concrete next step with a clear owner and timeframe.",
        },
    ),
)

DIMENSION_NAMES: tuple[str, ...] = tuple(d.name for d in DIMENSIONS)
