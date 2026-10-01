from document_ocr.synthesis.template_compiler.host import (
    _repair_uniquely_bracketed_target_paths,
    _topology_target_draft,
)


def test_bracketed_repair_uses_path_local_ranking_when_later_path_has_no_candidate() -> None:
    raw = "--- PAGE 1 ---\nGOODS: THING\nHS: 123456\n"
    description_path = "documentPatch.cargoGroups[0].description"
    present_path = "documentPatch.cargoGroups[0].hsCodes[0]"
    absent_path = "documentPatch.cargoGroups[0].hsCodes[1]"
    context = _topology_target_draft(
        raw=raw,
        path=description_path,
        start=raw.index("THING"),
        end=raw.index("THING") + len("THING"),
    )
    source_target = {
        "documentPatch": {
            "cargoGroups": [
                {"description": "THING", "hsCodes": ["123456", "987654"]}
            ]
        }
    }

    repaired = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=(present_path, absent_path),
        occupied=(context,),
        source_target=source_target,
    )

    assert [(draft.target_paths, draft.source_text) for draft in repaired] == [
        ((present_path,), "123456")
    ]
