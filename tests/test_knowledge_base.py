import json
from pathlib import Path

from knowledge_base.validate_knowledge_base import validate


ROOT = Path(__file__).resolve().parents[1]


def test_committed_compact_knowledge_base_is_valid():
    path = ROOT / "knowledge_base.json"
    assert validate(path) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert len(data["companies"]) == 6
    assert len(data["technologies"]) == 4
    assert data["certifications"] == {}


def test_every_compact_fact_is_actionable():
    data = json.loads(
        (ROOT / "knowledge_base.json").read_text(encoding="utf-8")
    )
    for entry in data["companies"].values():
        assert entry["founded_date"]
        assert entry["precision"]
        assert entry["source"].startswith("https://")
        assert entry["source_type"]
        assert None not in entry.values()
    for entry in data["technologies"].values():
        assert entry["released_date"]
        assert entry["patterns"]
        assert entry["source"].startswith("https://")
        assert entry["source_type"]
        assert None not in entry.values()
