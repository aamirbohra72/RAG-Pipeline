"""Unit checks for ordinal-aware query rewrite (no API key required)."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.query_rewrite_service import (  # noqa: E402
    _extract_ordered_entities,
    _resolve_ordinal_entity,
    rewrite_query_if_needed,
)

# Matches the real UI answer format (no markdown italics on product names)
ASSISTANT_PLAIN = """NovaGrid sells the following products as of fiscal year 2025:

1. **Residential Battery Systems**:
   - NovaCell Home 10 (10 kWh, launched 2018) [novagrid_report.pdf, page 2]
   - NovaCell Home 20 (20 kWh, launched 2021) [novagrid_report.pdf, page 2]

2. **Commercial Battery Systems**:
   - NovaGrid Commercial X (100 kWh, launched 2020) [novagrid_report.pdf, page 2]

3. **Solar Inverters**:
   - SolarSync Inverter II (launched 2023) [novagrid_report.pdf, page 2]

4. **Grid Management Software**:
   - GridPilot Software (subscription-based, launched 2022) [novagrid_report.pdf, page 2]
"""

ASSISTANT_MARKDOWN = """NovaGrid sells the following products:

1. **Residential Batteries**:
   - *NovaCell Home 10* (10 kWh, launched 2018)
   - *NovaCell Home 20* (20 kWh, launched 2021)
2. **Commercial Battery**:
   - *NovaGrid Commercial X* (100 kWh, launched 2020)
3. **Solar Inverter**:
   - *SolarSync Inverter II* (launched 2023)
4. **Grid Management Software**:
   - *GridPilot Software* (subscription-based, launched 2022)
"""


def _check(label: str, assistant: str) -> None:
    entities = _extract_ordered_entities(assistant)
    print(f"{label} entities:", entities)
    assert entities[0].startswith("NovaCell Home 10"), entities
    assert entities[1].startswith("NovaCell Home 20"), entities
    assert any("Commercial X" in e for e in entities), entities
    assert any("GridPilot" in e for e in entities), entities
    assert not any(e.endswith("Systems") for e in entities), entities

    history = [
        {"role": "user", "content": "What products does NovaGrid sell?"},
        {"role": "assistant", "content": assistant},
    ]
    second = _resolve_ordinal_entity("what about the second one?", history)
    print(f"{label} second:", second)
    assert second and "Home 20" in second, second

    result = rewrite_query_if_needed("what about the second one?", history)
    print(f"{label} rewrite:", result["retrieval_query"])
    assert result["was_rewritten"] is True
    assert "Home 20" in result["retrieval_query"]
    assert "Home 10" not in result["retrieval_query"]


def main() -> None:
    _check("plain", ASSISTANT_PLAIN)
    _check("markdown", ASSISTANT_MARKDOWN)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
