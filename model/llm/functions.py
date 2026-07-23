"""
Function-calling actions available to Gemma's triage stage.

`TriageResult.next_action` (see reasoning.py) names one of these
functions; model/pipeline.py executes it after getting the triage
judgment back. This is the "function calling" stage from the
redesign: Gemma decides what should happen next, and the backend
carries it out.

Only two actions exist right now, both intentionally simple:
  - emergency_escalation: no external call — just a strong, structured
    signal the frontend uses to show an urgent banner instead of a
    normal answer card.
  - find_nearest_hospital: a PLACEHOLDER. No hospital directory / maps
    API is wired up yet, so this returns a clear placeholder note
    rather than pretending to have located a real facility. Swap the
    body of this function for a real lookup (e.g. a facility registry,
    or a maps API) when you have one to connect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FunctionCallResult:
    action: str
    note: str


def emergency_escalation() -> FunctionCallResult:
    return FunctionCallResult(
        action="emergency_escalation",
        note="This sounds urgent. Call 108 (India's emergency ambulance number) or go to "
        "the nearest hospital now.",
    )


def find_nearest_hospital(speciality: str = "General Medicine") -> FunctionCallResult:
    """PLACEHOLDER — no real facility directory is connected yet."""
    # TODO(sanjeevani): replace with a real lookup (facility registry, maps API, etc.)
    return FunctionCallResult(
        action="find_nearest_hospital",
        note=f"Finding a nearby {speciality.lower()} facility isn't connected yet in this "
        "build — for now, please ask your local ASHA worker or Primary Health Centre "
        "for the nearest option.",
    )


def dispatch(next_action: str) -> FunctionCallResult | None:
    """Maps a TriageResult.next_action string to the function it names."""
    if next_action == "emergency_escalation":
        return emergency_escalation()
    if next_action == "find_nearest_hospital":
        return find_nearest_hospital()
    return None
