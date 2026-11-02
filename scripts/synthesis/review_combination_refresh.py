"""Native constrained full-text check of the CURRENT-context cargo migration."""

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict
from stage_combination_refresh import ART, OUT, ROOT

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "cargo_text_target_mismatch",
        "stale_old_cargo",
        "packing_contradiction",
        "unsupported_new_context",
        "existing_unrelated_issue",
    ]
    quote: str
    reason: str
    correction: str


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding]


SYSTEM = """Review a bounded bill-of-lading synthetic cargo refresh. We preserve current parties,
routes, dates, container/seal/private identifiers and unrelated text; independently resample
cargo, packaging, equipment size/type, masses, volumes, repeated numeric statements and
allocations. Compare BEFORE and AFTER, not the historical real source. Read the whole AFTER
for stale old commodity/package/weight references and contradictions, and verify its cargo
targets are literal or policy-supported. The extraction uses one goods entry, multiple
DISJOINT public package rows, and scalar container allocations; do not invent per-type
allocation matrices or treat inner/support packaging as additive outer totals. Repeated
public rows can be per-container declarations. Partial allocation absence stays absent.
HS suffix digits are deliberately synthetic and formatted: no tariff/product-correctness
test. Check no old suffix or wrong sampled goods name survives. Generic material targets
must not acquire an unsupported specific outer material; an explicitly stated inner liner
or support is different. Optional plural aliases are valid even for quantity one. Bare
goods count as unpackaged units when printed; not every goods line needs an enclosing noun.
Check good descriptions preserve all sampled product identities without contradictory
temperatures or packing. New goods text must not borrow a different native-generated
country/party or private reference from another context. Preserved geography/parties are
authoritative. Simple plausible packing is sufficient; no exact commercial approval or
real submission compliance is asserted. Report concrete evidence, not stylistic preference,
no desire to expand descriptions or invent missing information. An issue already present
outside changed cargo must be marked existing_unrelated_issue. Findings=[] means no
concrete defect found, not publication authorization. No reasoning outside JSON."""


async def main():
    base = json.loads((ART / "round-01/case/config.json").read_text())
    cfg = {k: base[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(output=str((ART / "hybrid-review").relative_to(ROOT)), concurrency=6, budget_usd=10)
    runner = Runner(ROOT, cfg)
    paths = sorted((OUT / "staged-cargo").glob("*.json"))

    async def run(path):
        row = json.loads(path.read_text())
        did = row["after"]["documentId"]
        dest = OUT / "hybrid-reviews" / f"{did}.json"
        pin = digest(row["after"])
        if dest.exists() and json.loads(dest.read_text())["afterSha256"] == pin:
            return
        native = json.loads((ROOT / row["evidence"]["nativePath"]).read_text())
        prompt = json.dumps(
            dict(
                beforeText=row["before"]["joinedRawText"],
                afterText=row["after"]["joinedRawText"],
                beforeCargo=row["before"]["target"]["goodsItemDetails"],
                afterTarget=row["after"]["target"],
                sampledGoods=native["scenario"]["goods_identities"],
                privatePacking=native["scenario"]["provenance"]["freshPrivatePacking"],
                changedText=[
                    dict(before=e["before"], after=e["after"])
                    for e in row["evidence"]["currentProof"]["edits"]
                    if e["before"] != e["after"]
                ],
            ),
            ensure_ascii=False,
        )
        result = await runner.call("scoped-cargo-review", did, Review, SYSTEM, prompt)
        atomic_write_json(
            dest, dict(documentId=did, afterSha256=pin, findings=result.model_dump()["findings"])
        )
        print(did, len(result.findings), flush=True)

    await asyncio.gather(*(run(p) for p in paths))
    print("billed cumulative review USD", runner.spent, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
