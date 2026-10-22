"""Independent campaigns use registry geography without constructing cargo donors."""

from copy import deepcopy

import pytest
from test_curated_scenarios import catalog, config, source_row

from document_ocr.synthesis.curated_scenarios import ScenarioCatalog, SourceCapabilities


def registry_only(base, rows, **changes):
    return ScenarioCatalog(
        config=config(**changes),
        countries=base.countries,
        hs=base.hs,
        phrases=base.phrases,
        ports=base.ports,
        localities=base.localities,
        dg=base.dg,
        train_rows=rows,
        validation_ids=frozenset(),
        fit_observed=False,
    )


def test_registry_only_catalog_never_constructs_observed_commodity_bundles(monkeypatch):
    row = source_row()
    before = deepcopy(row)
    base = catalog([row])

    def forbidden(*args, **kwargs):
        raise AssertionError("observed cargo was accessed")

    monkeypatch.setattr("document_ocr.synthesis.curated_scenarios._observation", forbidden)
    support = registry_only(base, [row])
    assert support.observations == ()
    assert support.phrases == base.phrases
    assert support.ports == base.ports
    assert support.localities == base.localities
    assert support.train == base.train
    assert row == before
    with pytest.raises(ValueError, match="observed scenario sampling is disabled"):
        support.sample(row, seed=1, variant=1, capabilities=SourceCapabilities(family="ambient"))


@pytest.mark.parametrize(
    "changes",
    [
        {"load_profiles": {"path": "unused", "sha256": "0" * 64}},
        {"profile_weights": {"unused": 1}},
        {"package_weights": {"PACKAGE_BALE": 1}},
        {"equipment_weights": {"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1}},
    ],
)
def test_observed_profile_settings_cannot_be_silently_ignored(changes):
    row = source_row()
    base = catalog([row])
    with pytest.raises(ValueError, match="observed cargo/profile configuration"):
        registry_only(base, [row], **changes)
