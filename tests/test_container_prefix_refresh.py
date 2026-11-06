from document_ocr.synthesis.container_prefix_refresh import refresh_prefix
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.generators import validate_container_number


def test_repeated_formatted_ids_and_links_only():
    raw = "ABCU 123456-7 / 80 BOXES\nTEMU7654321 / 80 BOXES\nABCU 123456-7"
    row = dict(
        documentId="sample",
        joinedRawText=raw,
        joinedRawTextSha256=digest(raw.encode()),
        positionedText=(
            "ABCU 123456-7 / 80 BOXES || 100,200\n"
            "TEMU7654321 / 80 BOXES || 100,300\nABCU 123456-7 ||"
        ),
        target=dict(
            containerInformation=[
                dict(equipmentIdentifier="ABCU1234567"),
                dict(equipmentIdentifier="TEMU7654321"),
            ],
            goodsItemDetails=[
                dict(
                    splitGoodsPlacement=[
                        dict(equipmentIdentifier="ABCU1234567", packageQuantity=80),
                        dict(equipmentIdentifier="TEMU7654321", packageQuantity=80),
                    ]
                )
            ],
        ),
    )
    after, receipt = refresh_prefix(row, prefix="ECMU", seed=123, excluded=set())
    assert len(receipt["spans"]) == 3
    assert all(
        validate_container_number(s) and s.startswith("ECMU")
        for s in receipt["identifiers"].values()
    )
    assert after["joinedRawText"].count("80 BOXES") == 2
    assert "|| 100,200\n" in after["positionedText"]
    assert "|| 100,300\n" in after["positionedText"]
    assert all(
        a["packageQuantity"] == 80
        for a in after["target"]["goodsItemDetails"][0]["splitGoodsPlacement"]
    )
    assert refresh_prefix(row, prefix="ECMU", seed=123, excluded=set()) == (after, receipt)
