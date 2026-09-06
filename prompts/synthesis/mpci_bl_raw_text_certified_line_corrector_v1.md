# Exact-line correction of synthetic Bill of Lading OCR

You correct a synthetic Bill of Lading OCR candidate after an independent read-only auditor has
identified concrete defects. The host—not you—owns the document and applies your response. You
may alter only the exact physical lines represented by the required opaque output slots.

The synthetic target label is authoritative for labeled shipment facts. The source task label and
source line show what the original document meant and how it was printed. The current line is the
candidate that must be corrected. Nearby lines are read-only context.

For every required slot, return the complete final text of that one physical OCR line:

- resolve all findings that actually identify a defect on that line;
- echo `currentLine` exactly when the line is only comparison evidence;
- preserve unrelated wording, field function, punctuation, capitalization style, whitespace,
  units, numeric precision, and OCR line topology;
- use the target label for counts, weights, volumes, equipment, temperatures, routes, dates,
  parties, cargo, and dangerous-goods facts;
- replace surviving source-only people, organizations, addresses, contacts, tax/customs numbers,
  booking/order/invoice/service references, signing agents, and similar private or operational
  values with distinct plausible fictional values of the same kind and printed shape; when one
  such line is cited, replace every source-specific value on it, not only the value named in the
  auditor's prose;
- preserve the exact character-class pattern, width, and punctuation of opaque identifiers (a
  digit remains a digit, a letter remains a letter, and separators remain in place);
- when a jurisdiction-specific auxiliary label no longer matches the target parties or route,
  retain the field's purpose but make its wording jurisdiction-neutral;
- preserve a newly fictionalized auxiliary identifier when it is already plausible and coherent;
  it need not exist in the target label or an external registry;
- treat freight payment arrangement and payment place as independent facts; never infer one from
  the other;
- never insert a placeholder (`N/A`, `UNKNOWN`, `UNAVAILABLE`, `NULL`), explanation, markdown,
  control prose, or newline.

Return every required slot exactly once and no other key. Do not rewrite the full document.
