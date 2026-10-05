"""Freeze the complete reviewed 006-008 acceptance/exclusion inventory."""

from review_real_v7_batch import ROOT, read

from document_ocr.atomic import atomic_publish_json


def main():
    entries = []
    for tag, path in [
        ("006", "direct-real-batch006-20261005"),
        ("007", "direct-real-batch007-recovery-20261005"),
        ("008", "direct-real-batch008-20261005"),
    ]:
        batch = ROOT / "artifacts/kie-labeling" / path
        selection = read(batch / "selection.json")
        # Resolve the frozen assignment from the manifest, not an assumed filename.
        matches = [p for p in selection["sources"] if "assignment" in p]
        assert len(matches) == 1, matches
        entries.append(
            dict(
                batchTag=tag,
                path=str(batch.relative_to(ROOT)),
                assignments=matches[0],
                exclusions=read(batch / "manual-review/exclusions.json"),
            )
        )
    dest = ROOT / "artifacts/kie-labeling/batch006-screen-20261005/publication-spec.json"
    atomic_publish_json(dest, dict(batches=entries))
    print(dest)


if __name__ == "__main__":
    main()
