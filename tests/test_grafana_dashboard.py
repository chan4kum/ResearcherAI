import json
from pathlib import Path


def test_grafana_dashboard_json_validity() -> None:
    """Ensure Grafana dashboard JSON is well-formed and adheres to requirements."""
    dashboard_path = Path("monitoring/grafana/agentic_platform_dashboard.json")
    assert dashboard_path.exists(), f"Dashboard JSON not found at {dashboard_path}"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("title") == "Enterprise Agentic Platform - Production Telemetry & Cost Governance"
    assert data.get("uid") == "agentic-platform-prod"
    assert data.get("schemaVersion") == 38

    panels = [p for p in data.get("panels", []) if p.get("type") != "row"]
    assert len(panels) == 9, f"Expected 9 core metric panels, found {len(panels)}"

    panel_titles = [p.get("title") for p in panels]
    assert any("Request Rate" in t for t in panel_titles)
    assert any("Error Rate" in t for t in panel_titles)
    assert any("Request Latency" in t for t in panel_titles)
    assert any("Agent Graph Execution" in t for t in panel_titles)
    assert any("Retrieval Iterations" in t for t in panel_titles)
    assert any("Tool Calls" in t for t in panel_titles)
    assert any("LLM Latency" in t for t in panel_titles)
    assert any("Token Consumption" in t for t in panel_titles)
    assert any("Hourly LLM Burn Rate" in t or "Cost" in t for t in panel_titles)

    # Verify all targets have valid PromQL queries and panel descriptions explaining operational value
    for panel in panels:
        desc = panel.get("description", "")
        assert desc.startswith("Answers:"), f"Panel '{panel.get('title')}' missing operational question description."
        assert len(panel.get("targets", [])) > 0, f"Panel '{panel.get('title')}' has no targets."
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            assert expr, f"Target in panel '{panel.get('title')}' has empty expr."
