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


def find_nearest_hospital(speciality: str = "General Medicine", lat: float | None = None, lng: float | None = None) -> FunctionCallResult:
    """Finds real nearby hospitals using the Overpass API if coordinates are given."""
    if lat is None or lng is None:
        return FunctionCallResult(
            action="find_nearest_hospital",
            note="I cannot find nearby hospitals because your location is not available. Please allow location access in your browser or ask your local ASHA worker."
        )

    import requests
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:10000,{lat},{lng});
      node["amenity"="clinic"](around:10000,{lat},{lng});
    );
    out body 5;
    """
    
    try:
        headers = {'User-Agent': 'SanjeevaniBot/1.0'}
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        elements = data.get("elements", [])
        
        if not elements:
            return FunctionCallResult(
                action="find_nearest_hospital",
                note="I couldn't find any hospitals within 10km of your location using OpenStreetMap. Please call 108 in an emergency."
            )
            
        note = "Here are the nearest facilities I found (click for Google Maps directions):<br><ul style='margin-top: 8px; padding-left: 20px;'>"
        for i, el in enumerate(elements[:5], 1):
            name = el.get("tags", {}).get("name", "Medical Facility")
            h_lat = el.get("lat")
            h_lon = el.get("lon")
            maps_link = f"https://www.google.com/maps/dir/?api=1&destination={h_lat},{h_lon}"
            note += f"<li><a href='{maps_link}' target='_blank' rel='noopener' style='text-decoration: underline; color: var(--teal);'>{name}</a></li>"
            
        note += "</ul>"
        return FunctionCallResult(action="find_nearest_hospital", note=note)
        
    except Exception as exc:
        return FunctionCallResult(
            action="find_nearest_hospital",
            note="I tried to find a nearby hospital, but the mapping service is currently unavailable. Please ask a local health worker."
        )


def dispatch(next_action: str, lat: float | None = None, lng: float | None = None) -> FunctionCallResult | None:
    """Maps a TriageResult.next_action string to the function it names."""
    if next_action == "emergency_escalation":
        return emergency_escalation()
    if next_action == "find_nearest_hospital":
        return find_nearest_hospital(lat=lat, lng=lng)
    return None
