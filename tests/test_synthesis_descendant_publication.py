import asyncio
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedRunError
from document_ocr.synthesis.template_compiler import descendant as renderer


def test_published_private_tares_survive_exact_decimal_roundtrip():
    numbers = {"tare:1": NS(contract=NS(mode="sampled_equipment_tare"))}
    assert renderer._published_equipment_tares(
        {"equipmentTareValues": {"tare:1": "3720"}}, numbers
    ) == {"tare:1": Decimal("3720")}
    assert renderer._published_equipment_tares({}, {}) == {}


@pytest.mark.parametrize(
    "payload", [{}, {"wrong": "3720"}, {"tare:1": 3720}, {"tare:1": "NaN"}, {"tare:1": "0"}]
)
def test_published_private_tares_cannot_be_missing_or_fabricated(payload):
    numbers = {"tare:1": NS(contract=NS(mode="sampled_equipment_tare"))}
    with pytest.raises(ValueError):
        renderer._published_equipment_tares({"equipmentTareValues": payload}, numbers)


def model(value):
    return NS(model_dump=lambda **kw: value)


def case(index, *, replay):
    prepared = NS(
        document_id=f"sample-{index}",
        source=b"source\r\n",
        source_target={"original": index},
        target={"new": index},
        auxiliary_values={"name": "new"},
        numeric_auxiliary={"sum": model({"n": 3})},
        dangerous_goods_facts=(model({"un": "1234"}),),
        target_receipt=model({"hash": index}),
        customs_presentation=NS(evidence={"neutral": True}, template=model({"slots": []})),
    )
    plan = NS(routes=(model({"route": index}),))
    execution = NS(
        stage=model({"requests": 0}),
        result=model({"passed": True}),
        slot_bindings={"slot": "new"},
        binding_outputs={"key": NS(canonical_value="new", replacements={"slot": "new"})},
        rendered=f"é new {index}\r\n".encode(),
        proof=model({"exact": True}),
    )
    receipt = model({"replay": index}) if replay else None
    return prepared, plan, execution, receipt


@pytest.mark.parametrize("replay", [False, True])
def test_parallel_descendant_publication_preserves_all_bytes(tmp_path, replay):
    data = [case(i, replay=replay) for i in range(9)]

    async def publish(run, workers):
        for offset in range(0, len(data), workers):
            await asyncio.gather(
                *(
                    asyncio.to_thread(renderer._publish_prepared_case, run, c, p)
                    for c, p, e, r in data[offset : offset + workers]
                )
            )
            await asyncio.gather(
                *(
                    asyncio.to_thread(renderer._publish_executed_case, run, c, e, r)
                    for c, p, e, r in data[offset : offset + workers]
                )
            )

    inventories = []
    for workers in (1, 4):
        run = StagedArtifactRun(
            output_parent=tmp_path, run_name=f"w{workers}", transaction_sha256="a" * 64
        )
        asyncio.run(publish(run, workers))
        before = run._scan_artifacts()
        asyncio.run(publish(run, workers))
        assert run._scan_artifacts() == before
        inventories.append(before)
        assert len(before) == len(data) * (18 if replay else 17)
        for c, _p, e, _r in data:
            folder = run.stage_root / "cases" / c.document_id
            assert (folder / "rendered.txt").read_bytes() == e.rendered
            assert (folder / "source.txt").read_bytes() == c.source
        run.commit(expected_artifacts=[r.relative_path for r in before], metadata={})
    assert inventories[0] == inventories[1]


def test_descendant_publication_propagates_conflicts_and_handles_rejection(tmp_path):
    run = StagedArtifactRun(
        output_parent=tmp_path, run_name="conflict", transaction_sha256="a" * 64
    )
    c, p, e, r = case(0, replay=False)
    c.customs_presentation = None
    e.rendered = e.proof = e.slot_bindings = e.binding_outputs = None
    renderer._publish_prepared_case(run, c, p)
    renderer._publish_executed_case(run, c, e, r)
    assert not (run.stage_root / "cases" / c.document_id / "rendered.txt").exists()
    c.target = {"different": True}
    with pytest.raises(StagedRunError, match="conflict"):
        renderer._publish_prepared_case(run, c, p)
    assert not run.completed
