import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "knowledge_base_builder", ROOT / "knowledge_base" / "build_knowledge_base.py"
)
validator = load_module(
    "knowledge_base_validator",
    ROOT / "knowledge_base" / "validate_knowledge_base.py",
)


def test_committed_knowledge_base_is_valid_and_complete():
    path = ROOT / "knowledge_base.json"
    assert validator.validate(path) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["companies"]) == 63
    assert len(data["technologies"]) == 133
    assert len(data["certifications"]) == 8
    assert data["metadata"]["source_candidate_count"] == 100000


def test_verified_facts_have_primary_source_metadata():
    data = json.loads((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
    for section in ("companies", "technologies", "certifications"):
        for entry in data[section].values():
            if entry["status"] != "verified":
                continue
            assert entry["date"]
            assert entry["date_precision"]
            assert entry["date_basis"]
            assert entry["sources"]
            assert all(source["url"].startswith("https://") for source in entry["sources"])


def test_fictional_and_not_dateable_entries_are_never_dated():
    data = json.loads((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
    for section in ("companies", "technologies"):
        for entry in data[section].values():
            if entry["status"] in {"fictional", "not_dateable"}:
                assert entry["date"] is None
                assert entry["date_precision"] is None
                assert entry["date_basis"] is None
