# Historical Knowledge Base

This directory owns the compact, actionable `knowledge_base.json`. It contains
only facts that can be used by integrity checks; it does not contain placeholder
or null inventory records.

Normalize an externally researched compact file:

```bash
python3 knowledge_base/build_knowledge_base.py \
  --input knowledge_base1.json \
  --output knowledge_base.json
```

Validate dates, sources, and technology regex patterns:

```bash
python3 knowledge_base/validate_knowledge_base.py
```

Add an entity only when it has a useful date, matching information, and a
reviewable source. Missing entities simply receive no historical check.
