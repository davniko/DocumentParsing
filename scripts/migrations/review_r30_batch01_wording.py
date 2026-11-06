"""Apply the main-agent readback of the first twenty additional-owner proposals."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin

from document_ocr.atomic import atomic_publish_json

REVIEWED = {
    "003c738e9cf8c6bda43ee207",
    "006912ce4856f0c46e4f00d3",
    "00da76e4fde563fc25fed1fd",
    "00efa251b0d9a243900afb67",
    "00f93f3b197ea99c1487aef5",
    "017b17474c6a0841d8d1ef7b",
    "01c4af12f09390240ffef869",
    "01d9c4683ff4018f5e2305b5",
    "02eccf7b97ad01fe6909f3de",
    "042081d81e00375b5697830f",
    "0458499ab3b0683ee7c5b584",
    "04ad0042be933277feb216e0",
    "04ddee68222be425b12c889c",
    "04e317812c2ae60ec21c9ad4",
    "04ef12f72900c5b8c3f7526b",
    "04efa668603695cbf4233409",
    "0555689f09bb3576dd54663c",
    "05be699447953eb76ea308da",
    "05f1bf84f674575d02ea1067",
    "063553e23391d747f070b709",
}


def main():
    decisions = []
    for short in sorted(REVIEWED):
        identity = "syn_full_v7_" + short
        path = ART / "wording-proposals" / (identity + ".json")
        result = read(path)
        if result["status"] != "PROPOSED_REQUIRES_REVIEW":
            raise ValueError("unexpected proposal state")
        result["status"] = "REVIEWED"
        result["review"] = {
            "proposal": pin(path),
            "scope": "Main agent inspected all requested values against frozen locality, company, contact, field boundary and source-style instructions. No goods/route changes. Subject to final rendered-fit and target-projection checks.",
        }
        output = ART / "wording" / (identity + ".json")
        atomic_publish_json(output, result)
        decisions.append(pin(output))
    atomic_publish_json(OUT / "batches/001-wording-review.json", {"reviewed": decisions})


if __name__ == "__main__":
    main()
