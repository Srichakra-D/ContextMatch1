# Historical Knowledge Base

This directory owns the compact `knowledge_base.json`. It contains sourced
facts for auditability, without placeholder or null inventory records.
Company facts are used by integrity checks. Technology release dates are kept
as reference data only and are not used for honeypot detection or ranking.

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
