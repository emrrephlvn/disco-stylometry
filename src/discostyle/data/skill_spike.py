"""Week 1, Spike B (V2 gate) — is 24-skill attribution recoverable?

Timebox: 3 hours. This module is allowed to stay ugly; it exists to produce a
verdict, not to ship.

Background (verified before this repo existed):
- Easy corpora expose ``actorName`` = character (Kim, Cuno, Narrator, ...).
  Skill voices (LOGIC, INLAND EMPIRE, ...) appear to be folded into
  "Narrator"-like actors there.
- BUT the in-game UI displays the skill name per line, so attribution exists
  somewhere in the raw dialogue bundle (the ~266 MB exported asset JSON —
  Articy/Pixel Crushers "Dialogue System" format: conversations -> dialogue
  entries with actor/conversant IDs and an actors table).

Procedure:
1. Point ``scan_asset`` at the exported asset JSON.
2. It inventories every distinct actor name and counts lines per actor.
3. PASS if skill names (see ``SKILLS``) appear as actors with meaningful line
   counts. Record the verdict in ROADMAP.md Week 1.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SKILLS = {
    # Intellect
    "Logic", "Encyclopedia", "Rhetoric", "Drama", "Conceptualization", "Visual Calculus",
    # Psyche
    "Volition", "Inland Empire", "Empathy", "Authority", "Esprit de Corps", "Suggestion",
    # Physique
    "Endurance", "Pain Threshold", "Physical Instrument", "Electrochemistry",
    "Shivers", "Half Light",
    # Motorics
    "Hand/Eye Coordination", "Perception", "Reaction Speed", "Savoir Faire",
    "Interfacing", "Composure",
}


def scan_asset(asset_json: Path) -> Counter:
    """Inventory actor names in the raw dialogue asset. Adjust key paths once
    the real structure is known — this is a spike, iterate in a notebook."""
    with open(asset_json, encoding="utf-8") as f:
        data = json.load(f)  # ~266 MB: fine on 16 GB RAM; else switch to ijson

    actors: Counter = Counter()
    # Hypothesized structure (Pixel Crushers Dialogue System export):
    #   data["actors"] -> id -> name;  data["conversations"][i]["dialogueEntries"][j]["actorID"]
    # Fix during the spike.
    actor_names = {a.get("id"): a.get("fields", {}).get("Name") for a in data.get("actors", [])}
    for conv in data.get("conversations", []):
        for entry in conv.get("dialogueEntries", []):
            name = actor_names.get(entry.get("actorID"))
            if name:
                actors[name] += 1
    return actors


def verdict(actors: Counter, min_lines_per_skill: int = 100) -> str:
    found = {s: actors[s] for s in SKILLS if actors.get(s, 0) >= min_lines_per_skill}
    if len(found) >= 12:  # half the skills with real volume is enough to unlock V2
        return f"PASS — {len(found)} skills with >= {min_lines_per_skill} lines: {sorted(found)}"
    return f"FAIL — only {len(found)} skills found with enough lines; V1 continues unchanged."
