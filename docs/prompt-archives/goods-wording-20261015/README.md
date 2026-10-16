# Archived goods-wording prompts

These are exact pre-integration reference snapshots, not runtime alternatives or
commented-out production code.

- `original.json`: previous production system prompt and goods-field requirement.
- `first-polish.json`: the tested approximate-word-count system prompt and its
  measured-count field requirement. Substitute the documented source-word count
  into the field template; no hard length or line quota is implied.

Both snapshots record their source-file hashes. The approved second polish adds
general source-like information-pattern guidance and the contextual boundary to
the first polish. Its exact tested text and subsequent implementation are
documented in `docs/kie-goods-description-prompt-probe-2026-10-15.md`.

Selecting an archive does not silently migrate any existing wording receipt:
production request hashes still bind the live system prompt, fields and schema.
