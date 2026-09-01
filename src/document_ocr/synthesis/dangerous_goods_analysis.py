"""Registry/corpus analysis for the dangerous-goods synthesis contract."""

from __future__ import annotations

import csv
import json
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis import dangerous_goods_registry as dangerous_goods_registry_module
from document_ocr.synthesis.config import SynthesisDangerousGoodsAnalysisConfig
from document_ocr.synthesis.dangerous_goods_registry import (
    DangerousGoodsRegistryReceipt,
    load_dangerous_goods_registry,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path


def _resolve_file(project_root: Path, path_value: str, expected_sha256: str) -> Path:
    path = resolve_config_path(project_root, path_value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"analysis input differs from its pin: {path}")
    return path.resolve(strict=True)


def _validate_committed_run(project_root: Path, configured: Any) -> Path:
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"committed analysis dependency differs from its pin: {root}")
    run = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    )
    run.validate_committed_run()
    return root.resolve(strict=True)


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _counter_rows[CounterKey: str](
    counter: Counter[CounterKey], *, key: str
) -> list[dict[str, Any]]:
    return [
        {key: name, "count": count, "fraction": count / counter.total()}
        for name, count in sorted(counter.items(), key=lambda row: (-row[1], row[0]))
    ]


def _plot_bytes(plotter: Any) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook")
    figure = plotter(plt, sns)
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    return stream.getvalue()


def _bar_plot(
    rows: list[dict[str, Any]],
    *,
    label: str,
    title: str,
    horizontal: bool = True,
) -> bytes:
    def draw(plt: Any, sns: Any) -> Any:
        import pandas as pd  # type: ignore[import-untyped]

        height = max(4.5, len(rows) * 0.38) if horizontal else 5.5
        figure, axis = plt.subplots(figsize=(11, height))
        frame = pd.DataFrame(rows)
        if horizontal:
            ordered = frame.iloc[::-1]
            sns.barplot(data=ordered, x="count", y=label, ax=axis, color="#2C7FB8")
        else:
            sns.barplot(data=frame, x=label, y="count", ax=axis, color="#2C7FB8")
            axis.tick_params(axis="x", rotation=45)
        axis.set_title(title)
        axis.set_xlabel("Records")
        figure.tight_layout()
        return figure

    return _plot_bytes(draw)


def _corpus_dg_rows(corpus_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    output: list[dict[str, Any]] = []
    documents = 0
    dg_documents = 0
    with corpus_path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                raise ValueError("DG analysis corpus contains a blank row")
            row = json.loads(raw)
            documents += 1
            target = row.get("target")
            groups = (
                target.get("documentPatch", {}).get("cargoGroups", [])
                if isinstance(target, dict)
                else []
            )
            document_has_dg = False
            for group in groups:
                hs_codes = tuple(group.get("hsCodes") or [])
                for dangerous in group.get("dangerousGoods") or []:
                    document_has_dg = True
                    flash = dangerous.get("flashPoint") or {}
                    output.append(
                        {
                            "document_id": row.get("documentId"),
                            "un_number": dangerous.get("unNumber"),
                            "hazard_category": dangerous.get("hazardCategory"),
                            "subsidiary_hazard_count": int(
                                dangerous.get("subsidiaryHazardCategory") is not None
                            ),
                            "packing_group": flash.get("packingGroupCategory"),
                            "flashpoint_present": "temperature" in flash,
                            "hs_codes": hs_codes,
                        }
                    )
            dg_documents += int(document_has_dg)
    return output, documents, dg_documents


def run_dangerous_goods_analysis(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisDangerousGoodsAnalysisConfig,
) -> dict[str, Any]:
    """Publish complete distributions plus explicit top-N visualization views."""

    _validate_committed_run(project_root, config.registry.run)
    receipt_path = _resolve_file(
        project_root,
        config.registry.receipt.path,
        config.registry.receipt.sha256,
    )
    receipt = DangerousGoodsRegistryReceipt.model_validate_json(
        read_regular_file_bytes(receipt_path), strict=True
    )
    hmt_path = _resolve_file(
        project_root,
        config.registry.hmt_records.path,
        config.registry.hmt_records.sha256,
    )
    ecics_path = _resolve_file(
        project_root,
        config.registry.ecics_links.path,
        config.registry.ecics_links.sha256,
    )
    registry = load_dangerous_goods_registry(
        hmt_path=hmt_path,
        hmt_sha256=config.registry.hmt_records.sha256,
        ecics_path=ecics_path,
        ecics_sha256=config.registry.ecics_links.sha256,
    )
    if (
        len(registry.hmt_records) != config.registry.hmt_records.records
        or len(registry.ecics_links) != config.registry.ecics_links.records
    ):
        raise ValueError("DG analysis registry row count differs from configuration")
    corpus_path = _resolve_file(project_root, config.corpus.path, config.corpus.sha256)
    corpus_rows, corpus_documents, corpus_dg_documents = _corpus_dg_rows(corpus_path)
    if corpus_documents != config.corpus.records:
        raise ValueError("DG analysis corpus row count differs from configuration")

    maritime = [row for row in registry.hmt_records if row.maritime_eligible]
    exact_links = [
        row for row in registry.ecics_links if row.disposition == "eligible_unique_maritime_hmt"
    ]
    category_counts = Counter(row.hazard_category for row in maritime)
    exact_class_counts = Counter(row.exact_hazard_class for row in maritime)
    packing_counts = Counter(row.packing_group_category or "ABSENT" for row in maritime)
    subsidiary_counts = Counter(str(len(row.exact_subsidiary_hazards)) for row in maritime)
    rows_by_un = Counter(row.un_number for row in maritime)
    disposition_counts = Counter(row.disposition for row in registry.ecics_links)
    hs_chapters = Counter(cast(str, row.hs6)[:2] for row in exact_links)
    links_by_un = Counter(row.un_number for row in exact_links)
    corpus_category = Counter(cast(str, row["hazard_category"] or "ABSENT") for row in corpus_rows)
    corpus_un = {cast(str, row["un_number"]) for row in corpus_rows if row["un_number"]}
    hmt_un = {row.un_number for row in maritime}
    exact_un = {row.un_number for row in exact_links}
    corpus_hs_lengths = Counter(
        str(len(code)) for row in corpus_rows for code in cast(tuple[str, ...], row["hs_codes"])
    )
    cross: Counter[tuple[str, str]] = Counter(
        (str(row.hazard_category), str(row.packing_group_category or "ABSENT")) for row in maritime
    )

    data_payloads: dict[str, bytes] = {
        "data/hazard-categories.csv": _csv_bytes(
            _counter_rows(category_counts, key="hazard_category"),
            ("hazard_category", "count", "fraction"),
        ),
        "data/exact-hazard-classes.csv": _csv_bytes(
            _counter_rows(exact_class_counts, key="exact_hazard_class"),
            ("exact_hazard_class", "count", "fraction"),
        ),
        "data/packing-groups.csv": _csv_bytes(
            _counter_rows(packing_counts, key="packing_group"),
            ("packing_group", "count", "fraction"),
        ),
        "data/ecics-link-dispositions.csv": _csv_bytes(
            _counter_rows(disposition_counts, key="disposition"),
            ("disposition", "count", "fraction"),
        ),
        "data/ecics-hs6-chapters.csv": _csv_bytes(
            _counter_rows(hs_chapters, key="hs_chapter"),
            ("hs_chapter", "count", "fraction"),
        ),
    }

    plot_payloads = {
        "plots/01-hmt-hazard-categories.png": _bar_plot(
            _counter_rows(category_counts, key="hazard_category"),
            label="hazard_category",
            title="Maritime-eligible PHMSA tuples by task-facing hazard category",
        ),
        "plots/02-hmt-exact-hazard-classes-top25.png": _bar_plot(
            _counter_rows(exact_class_counts, key="exact_hazard_class")[:25],
            label="exact_hazard_class",
            title="Top 25 exact regulatory hazard classes (full table in CSV)",
        ),
        "plots/03-hmt-packing-groups.png": _bar_plot(
            _counter_rows(packing_counts, key="packing_group"),
            label="packing_group",
            title="Packing-group availability in maritime-eligible tuples",
        ),
        "plots/04-hmt-subsidiary-count.png": _bar_plot(
            _counter_rows(subsidiary_counts, key="subsidiary_hazards"),
            label="subsidiary_hazards",
            title="Exact subsidiary hazards per regulatory tuple",
            horizontal=False,
        ),
        "plots/05-ecics-link-dispositions.png": _bar_plot(
            _counter_rows(disposition_counts, key="disposition"),
            label="disposition",
            title="ECICS exact-CUS join disposition",
        ),
        "plots/06-ecics-hs-chapters-top25.png": _bar_plot(
            _counter_rows(hs_chapters, key="hs_chapter")[:25],
            label="hs_chapter",
            title="Top 25 globally portable HS chapters in exact ECICS links",
        ),
        "plots/07-corpus-hazard-categories.png": _bar_plot(
            _counter_rows(corpus_category, key="hazard_category"),
            label="hazard_category",
            title="Observed DG hazard categories in the 1,157-label corpus",
        ),
        "plots/08-corpus-hs-code-lengths.png": _bar_plot(
            _counter_rows(corpus_hs_lengths, key="digits"),
            label="digits",
            title="Observed HS-code lengths on DG cargo groups",
            horizontal=False,
        ),
    }

    def histogram_plot(plt: Any, sns: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(list(rows_by_un.values()), discrete=True, ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Maritime HMT formulations per UN")
        axes[0].set_xlabel("Regulatory rows")
        sns.histplot(list(links_by_un.values()), bins=35, ax=axes[1], color="#7FCDBB")
        axes[1].set_title("Eligible ECICS chemicals per UN")
        axes[1].set_xlabel("Exact chemical links")
        figure.tight_layout()
        return figure

    plot_payloads["plots/09-records-per-un.png"] = _plot_bytes(histogram_plot)

    categories: tuple[str, ...] = tuple(sorted(category_counts))
    packings: tuple[str, ...] = (
        "HIGH_DANGER",
        "MEDIUM_DANGER",
        "LOW_DANGER",
        "ABSENT",
    )

    def heatmap_plot(plt: Any, sns: Any) -> Any:
        matrix = [[cross[(category, packing)] for packing in packings] for category in categories]
        figure, axis = plt.subplots(figsize=(13, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=packings,
            yticklabels=categories,
            ax=axis,
        )
        axis.set_title("Regulatory hazard category x packing group")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=25, ha="right")
        figure.tight_layout()
        return figure

    plot_payloads["plots/10-hazard-by-packing-group.png"] = _plot_bytes(heatmap_plot)

    summary = {
        "schemaVersion": 1,
        "registry": receipt.audit.model_dump(mode="json"),
        "corpus": {
            "documents": corpus_documents,
            "dangerousGoodsDocuments": corpus_dg_documents,
            "dangerousGoodsRows": len(corpus_rows),
            "distinctUnNumbers": len(corpus_un),
            "unNumbersCoveredByMaritimeHmt": len(corpus_un & hmt_un),
            "unNumbersCoveredByExactEcicsHsBranch": len(corpus_un & exact_un),
            "rowsWithFlashpoint": sum(bool(row["flashpoint_present"]) for row in corpus_rows),
            "rowsWithPackingGroup": sum(row["packing_group"] is not None for row in corpus_rows),
            "rowsWithHsCodes": sum(bool(row["hs_codes"]) for row in corpus_rows),
        },
        "generator": {
            "generalBranchMaritimeTuples": len(maritime),
            "generalBranchDistinctUnNumbers": len(hmt_un),
            "exactChemicalHsBranchLinks": len(exact_links),
            "exactChemicalHsBranchDistinctUnNumbers": len(exact_un),
            "exactChemicalHsBranchDistinctHs6": len({row.hs6 for row in exact_links}),
            "multipleSubsidiaryRowsRetainedInRegistry": sum(
                len(row.exact_subsidiary_hazards) > 1 for row in maritime
            ),
            "numericFlashpointsGenerated": 0,
        },
    }
    exact_hs6_count = len({row.hs6 for row in exact_links})
    report = f"""# Dangerous-goods synthesis registry analysis

This analysis is based on the pinned PHMSA HMT and exact-CUS ECICS snapshot. The
HS-linked branch additionally requires exact normalized agreement between an official
ECICS chemical name and the PHMSA proper shipping name. It is a synthetic-data source,
not an IMDG compliance engine.

## Coverage

- PHMSA source rows: **{receipt.audit.hmt_source_rows:,}**
- Valid compiled HMT tuples: **{receipt.audit.hmt_records:,}**
- Maritime-eligible tuples: **{len(maritime):,}** across **{len(hmt_un):,}** UN numbers
- Exact ECICS chemical/HS links: **{len(exact_links):,}** across
  **{len(exact_un):,}** UN numbers and **{exact_hs6_count:,}** HS6 codes
- Real corpus: **{corpus_dg_documents:,}** DG documents and **{len(corpus_rows):,}** DG rows

## Generation contract

The general branch samples an atomic PHMSA row and emits no HS code. The exact-chemical
branch samples an ECICS CUS record whose UN number has exactly one maritime HMT tuple
and whose official chemical name exactly matches that tuple's proper shipping name after
NFKD/ASCII/alphanumeric normalization, then emits the first six CN digits as portable
HS6. Sampling is hierarchical by semantic hazard category, UN number, then regulatory
row/chemical, so large ECICS buckets do not dominate.

Numeric flashpoint is never synthesized from hazard class or packing group: neither source
provides a formulation-specific flashpoint. Packing group is represented independently in
relation-v4. Exact classes, proper shipping names, CUS/CAS/CN identifiers, technical-name
requirements, and vessel stowage remain provenance/rendering metadata.

## Source boundaries

- PHMSA oCFR HMT: {_PHMSA_DOC_URL}
- ECICS: {_ECICS_DOC_URL}
- IMDG remains the maritime compliance authority; the compiled PHMSA source must not be
  represented as an IMDG registry.

All complete distributions are in `data/`; plots explicitly labeled `top25` are visual
views only and do not truncate the underlying tables.
"""
    summary_payload = json_artifact_bytes(summary)
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "configSha256": sha256_file(config_path),
                "registryReceiptSha256": config.registry.receipt.sha256,
                "hmtSha256": config.registry.hmt_records.sha256,
                "ecicsSha256": config.registry.ecics_links.sha256,
                "corpusSha256": config.corpus.sha256,
                "summarySha256": sha256_bytes(summary_payload),
                "implementationSha256": sha256_file(Path(__file__)),
                "registryImplementationSha256": sha256_file(
                    Path(dangerous_goods_registry_module.__file__)
                ),
            }
        )
    )
    output_parent = resolve_config_path(project_root, config.run.output_dir)
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    for relative_path, payload in sorted(data_payloads.items()):
        stage.publish_bytes(relative_path, payload)
    for relative_path, payload in sorted(plot_payloads.items()):
        stage.publish_bytes(relative_path, payload)
    stage.publish_bytes("summary.json", summary_payload)
    stage.publish_bytes("REPORT.md", report.encode("utf-8"))
    expected = tuple(
        sorted(
            (
                "REPORT.md",
                "config.yaml",
                "summary.json",
                *data_payloads,
                *plot_payloads,
            )
        )
    )
    committed = stage.commit(
        expected_artifacts=expected,
        metadata={
            "schema_version": 1,
            "plots": len(plot_payloads),
            "hmt_records": len(registry.hmt_records),
            "ecics_links": len(registry.ecics_links),
            "corpus_dg_rows": len(corpus_rows),
        },
    )
    return {
        "output_dir": str(stage.final_root),
        "created": committed.created,
        "plots": len(plot_payloads),
        "hmt_records": len(registry.hmt_records),
        "exact_hs_links": len(exact_links),
        "corpus_dg_rows": len(corpus_rows),
    }


_PHMSA_DOC_URL = "https://www.phmsa.dot.gov/standards-rulemaking/hazmat/phmsas-online-cfr-ocfr"
_ECICS_DOC_URL = (
    "https://taxation-customs.ec.europa.eu/online-services/"
    "online-services-and-databases-customs/"
    "european-customs-inventory-chemical-substances-ecics-0_en"
)
