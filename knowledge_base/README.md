# Historical Knowledge Base

This directory owns creation and validation of `knowledge_base.json`. The
current 641-candidate ranking pipeline consumes this file through the
`scan-integrity` command.

Build from the complete 100,000-candidate dataset:

```bash
python3 knowledge_base/build_knowledge_base.py
```

Validate coverage and source requirements:

```bash
python3 knowledge_base/validate_knowledge_base.py
```

Status meanings:

- `verified`: dated fact backed by a primary source.
- `ambiguous`: a likely date exists but needs stronger or clearer evidence.
- `unknown`: no approved primary-source date is currently recorded.
- `fictional`: synthetic employer placeholder; not invalid by name.
- `not_dateable`: broad concept without one defensible release date.

Unknown and ambiguous entries are retained deliberately. They must never be
silently upgraded to verified facts.
