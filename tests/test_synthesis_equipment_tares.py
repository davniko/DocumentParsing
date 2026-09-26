from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import equipment_tares as tares
from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract

PAIR = "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"


def document(tare="3700", *, observed=True, number="ABCU1234567"):
    text = (
        "GROSS WEIGHT\nCARGO\nKGS\nTARE\nKGS\nCBM\n\n"
        f"{number}\nSEAL SE123\n\n10 CARTONS\n10000.000\n{tare}\n30.000\n\n"
        f"Weight in Kgs Total: 1 CONTAINER(S)\n10000.000 {tare} 30.000\n"
    )
    container = {"containerNumber": number}
    if observed:
        container.update(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")
    return dict(
        documentId=number, joinedRawText=text, target={"documentPatch": {"containers": [container]}}
    )


def source(row):
    text = row["joinedRawText"]
    offsets = [text.index("3700"), text.rindex("3700")]
    binding = NS(
        logical_key="tare",
        derivation=None,
        occurrences=tuple(
            NS(byte_start=i, byte_end=i + 4, source_text="3700", slot_id=str(n))
            for n, i in enumerate(offsets)
        ),
    )
    return NS(source=text.encode(), target=row["target"], template=NS(bindings=[binding]))


def contract():
    return NumericContract(
        mode="source_fixed",
        role="tare",
        source_value="3700",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Reviewed printed tare in kilograms",
    )


def test_train_support_uses_exact_printed_pair_and_unambiguous_tare():
    support = tares.build_support([document()])
    assert support.pairs_by_kg == {Decimal(3700): frozenset({PAIR})}
    assert support.source_ids_by_kg_pair == {(Decimal(3700), PAIR): ("ABCU1234567",)}


def test_unobserved_equipment_does_not_establish_its_own_fit_pair():
    assert not tares.build_support([document(observed=False)]).pairs_by_kg


def test_captioned_total_tare_is_not_an_individual_observation_for_last_container():
    row = document()
    row["target"]["documentPatch"]["containers"].append(
        {
            "containerNumber": "DEFU9876543",
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        }
    )
    row["joinedRawText"] = (
        "ABCU1234567 / 40HC / SEAL1/3.890,00 KGS\n/FCL\n28 PALLET(S)\n"
        "DEFU9876543 / 40HC / SEAL2/3.700,00 KGS\n/FCL\n28 PALLET(S)\n"
        "TOTAL TARE WEIGHT 7,590.00 KGS\nTOTAL PALLET 56\n"
    )
    assert not tares.build_support([row]).pairs_by_kg
    occurrences = tares._occurrences(
        row["joinedRawText"], row["target"]["documentPatch"]["containers"]
    )
    assert occurrences[(7, "7,590.00")] == frozenset({0, 1})


@pytest.mark.parametrize("surface", ["3.700", "3,700"])
def test_ambiguous_numeric_surface_is_not_guessed_for_tare_fit(surface):
    assert not tares.build_support([document(surface)]).pairs_by_kg


def test_four_integer_digits_are_not_valid_thousands_grouping():
    assert tares.build_support([document("3700.000")]).pairs_by_kg == {
        Decimal(3700): frozenset({PAIR})
    }


def test_declared_kg_units_are_required():
    row = document()
    row["joinedRawText"] = row["joinedRawText"].replace("KGS", "LBS").replace("Kgs", "Lbs")
    assert not tares.build_support([row]).pairs_by_kg


@pytest.mark.parametrize("unit", ["LB", "LBS", "POUNDS", "TONNES"])
def test_aggregate_kg_caption_cannot_override_a_different_tare_column_unit(unit):
    row = document()
    row["joinedRawText"] = row["joinedRawText"].replace("TARE\nKGS", "TARE\n" + unit)
    assert not tares.build_support([row]).pairs_by_kg


def test_conflicting_repeated_tare_headers_are_not_unified_by_numeric_coincidence():
    row = document()
    row["joinedRawText"] += "\nGROSS WEIGHT\nCARGO\nKGS\nTARE\nLBS\nCBM\n"
    assert not tares.build_support([row]).pairs_by_kg


def test_latent_options_do_not_add_equipment_to_labels():
    original = document(observed=False)
    src = source(original)
    support = tares.build_support([document()])
    assert tares.compile_constraints(
        src, {"tare": contract()}, support, frozenset({0})
    ).options == {0: frozenset({PAIR})}
    assert src.target["documentPatch"]["containers"] == [{"containerNumber": "ABCU1234567"}]


def test_nearest_weight_is_never_used_as_support():
    with pytest.raises(ValueError, match="no exact train-only"):
        tares.compile_constraints(
            source(document(observed=False)),
            {"tare": contract()},
            tares.build_support([document("3710")]),
            frozenset({0}),
        )


def test_total_cannot_fabricate_missing_container_ownership():
    row = document(observed=False)
    row["target"]["documentPatch"]["containers"].append({"containerNumber": "XYZU7654321"})
    with pytest.raises(ValueError, match="inventory"):
        tares.compile_constraints(
            source(row), {"tare": contract()}, tares.build_support([document()]), frozenset({0, 1})
        )


def test_shared_source_scalar_requires_at_least_one_explicit_kg_row():
    row = document(observed=False)
    row["joinedRawText"] = row["joinedRawText"].replace("TARE", "NET")
    with pytest.raises(ValueError, match="ownership"):
        tares.compile_constraints(
            source(row), {"tare": contract()}, tares.build_support([document()]), frozenset({0})
        )


def test_known_cargo_line_and_ibc_count_do_not_hide_dense_tare_column():
    row = document()
    row["target"]["documentPatch"]["cargoGroups"] = [{"description": "LUBRICANTS"}]
    row["joinedRawText"] = row["joinedRawText"].replace(
        "10 CARTONS\n", "10 INTERMEDIATE BULK CONTAINERS\nLUBRICANTS\n\n"
    )
    # Remove the independently usable aggregate to test the individual row.
    row["joinedRawText"] = row["joinedRawText"].replace("10000.000 3700 30.000", "")
    observations = tares._occurrences(
        row["joinedRawText"], row["target"]["documentPatch"]["containers"], ("LUBRICANTS",)
    )
    assert len(observations) == 1
    assert next(iter(observations.values())) == frozenset({0})
    assert tares.build_support([row]).pairs_by_kg == {Decimal(3700): frozenset({PAIR})}


def aggregate_source(total="7400"):
    raw = ("ABCU1234567\nDEFU9876543\nTOTAL TARE WEIGHT " + total + " KGS\n").encode()
    start = raw.index(total.encode())
    return NS(
        source=raw,
        target={
            "documentPatch": {
                "containers": [
                    {"containerNumber": "ABCU1234567"},
                    {"containerNumber": "DEFU9876543"},
                ]
            }
        },
        template=NS(
            bindings=[
                NS(
                    logical_key="total",
                    occurrences=[
                        NS(
                            byte_start=start,
                            byte_end=start + len(total),
                            source_text=total,
                        )
                    ],
                )
            ]
        ),
    )


def test_aggregate_tare_can_condition_private_equipment_without_inventing_labels():
    src = aggregate_source()
    support = tares.build_support([document()])
    constraints = tares.compile_constraints(
        src,
        {"total": contract().model_copy(update={"source_value": "7400"})},
        support,
        frozenset({0, 1}),
    )
    assert constraints.total_kg == Decimal(7400)
    assert constraints.witness({0: frozenset({PAIR}), 1: frozenset({PAIR})}) == {
        0: Decimal(3700),
        1: Decimal(3700),
    }
    assert all(set(c) == {"containerNumber"} for c in src.target["documentPatch"]["containers"])


def test_independently_possible_pairs_cannot_violate_the_joint_tare_total():
    small = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    domain = ((PAIR, (Decimal(3700),)), (small, (Decimal(2200),)))
    constraints = tares.TareConstraints(((0, domain), (1, domain)), Decimal(5900))
    assert constraints.witness({0: frozenset({small}), 1: frozenset({PAIR, small})}) == {
        0: Decimal(2200),
        1: Decimal(3700),
    }
    assert constraints.witness({0: frozenset({small}), 1: frozenset({small})}) is None
    assert constraints.witness({0: frozenset({PAIR}), 1: frozenset({PAIR})}) is None


def test_unobserved_tare_is_not_rounded_to_make_an_aggregate_fit():
    with pytest.raises(ValueError, match="no exact joint"):
        tares.compile_constraints(
            aggregate_source("7401"),
            {"total": contract().model_copy(update={"source_value": "7401"})},
            tares.build_support([document()]),
            frozenset({0, 1}),
        )


def test_decimal_tare_solver_is_exact_and_rejects_impossible_domains():
    assert tares._sum_witness(((Decimal("2200.125"),), (Decimal("3700.875"),)), Decimal(5901)) == (
        Decimal("2200.125"),
        Decimal("3700.875"),
    )
    assert tares._sum_witness(((Decimal("2200.125"),), ()), Decimal(5901)) is None
    assert tares._sum_witness(((Decimal("2200.125"),),), Decimal("2200.124")) is None


def explicitly_owned_source(text="TARE (KGS)\n3700\nABCU1234567\nDEFU9876543\n"):
    start = text.index("3700")
    binding = NS(
        logical_key="tare",
        target_paths=(),
        dependency_paths=("documentPatch.containers[0]",),
        occurrences=(
            NS(byte_start=start, byte_end=start + 4, source_text="3700", slot_id="tare0"),
        ),
    )
    return NS(
        source=text.encode(),
        target={
            "documentPatch": {
                "containers": [
                    {"containerNumber": "ABCU1234567"},
                    {"containerNumber": "DEFU9876543"},
                ]
            }
        },
        template=NS(bindings=(binding,)),
    )


def test_reviewed_tare_row_can_precede_container_without_inventing_units_or_labels():
    src = explicitly_owned_source()
    original = src.source, repr(src.target)
    reviewed = NumericContract.model_validate(
        {**contract().model_dump(), "printed_unit_quote": "TARE (KGS)"}
    )
    constraints = tares.compile_constraints(
        src, {"tare": reviewed}, tares.build_support([document()]), frozenset({0, 1})
    )
    assert constraints.options == {0: frozenset({PAIR})}
    assert constraints.witness(constraints.options) == {0: Decimal(3700)}
    assert (src.source, repr(src.target)) == original


def test_tare_ownership_is_not_a_cargo_capacity_load():
    from document_ocr.synthesis.template_compiler.equipment_row_constraints import compile_rows

    src = explicitly_owned_source()
    reviewed = NumericContract.model_validate(
        {**contract().model_dump(), "printed_unit_quote": "TARE (KGS)"}
    )
    assert compile_rows(src, {"tare": reviewed}) == ()
    assert len(compile_rows(src, {"tare": reviewed}, include_tare=True)) == 1


def test_reviewed_tare_owner_cannot_override_a_printed_container_row():
    src = source(document(observed=False))
    src.source = src.source.replace(b"Total: 1 CONTAINER(S)", b"Total: 2 CONTAINER(S)")
    src.target["documentPatch"]["containers"].append({"containerNumber": "DEFU9876543"})
    src.template.bindings[0].target_paths = ()
    src.template.bindings[0].dependency_paths = ("documentPatch.containers[1]",)
    src.template.bindings[0].occurrences = src.template.bindings[0].occurrences[:1]
    reviewed = NumericContract.model_validate(
        {**contract().model_dump(), "printed_unit_quote": "TARE\nKGS"}
    )
    with pytest.raises(ValueError, match=r"contradicts.*container row"):
        tares.compile_constraints(
            src, {"tare": reviewed}, tares.build_support([document()]), frozenset({0, 1})
        )


@pytest.mark.parametrize("change", ["unowned", "wrong_unit", "wrong_row", "partial_repeat"])
def test_reviewed_tare_ownership_cannot_waive_missing_or_contradictory_evidence(change):
    src = explicitly_owned_source()
    quote = "TARE (KGS)"
    if change == "unowned":
        src.template.bindings[0].dependency_paths = ()
    elif change == "wrong_unit":
        src = explicitly_owned_source("TARE (KGS)\n3700 LBS\nABCU1234567\nDEFU9876543\n")
    elif change == "wrong_row":
        src.template.bindings[0].dependency_paths = ("documentPatch.containers[2]",)
    else:
        src.template.bindings[0].dependency_paths += ("documentPatch.containers[1]",)
    reviewed = NumericContract.model_validate(
        {**contract().model_dump(), "printed_unit_quote": quote}
    )
    with pytest.raises(ValueError):
        tares.compile_constraints(
            src, {"tare": reviewed}, tares.build_support([document()]), frozenset({0, 1})
        )


def test_sampled_tare_replaces_missing_exact_source_support_without_inventing_labels():
    from document_ocr.synthesis.generators import DeterministicStream

    src = source(document(observed=False))
    original = src.source, repr(src.target)
    c = contract().model_copy(update={"mode": "sampled_equipment_tare"})
    support = tares.build_support([document("3710"), document("3850")])
    constraints = tares.compile_constraints(src, {"tare": c}, support, frozenset({0}))
    assert constraints.options == {0: frozenset({PAIR})}
    stream = DeterministicStream(seed=71, namespace="test", identity="sample")
    private, values = constraints.sample(constraints.options, stream)
    assert values == {"tare": private[0]}
    assert private[0] in {Decimal(3710), Decimal(3850)}
    assert constraints.sample(constraints.options, stream) == (private, values)
    assert (src.source, repr(src.target)) == original


def test_sampled_aggregate_is_sum_of_private_train_supported_rows_not_original_total():
    from document_ocr.synthesis.generators import DeterministicStream

    src = aggregate_source("4060")
    support = tares.build_support([document("3710"), document("3850")])
    c = contract().model_copy(update={"mode": "sampled_equipment_tare", "source_value": "4060"})
    constraints = tares.compile_constraints(src, {"total": c}, support, frozenset({0, 1}))
    assert constraints.total_kg is None
    private, values = constraints.sample(
        constraints.options, DeterministicStream(seed=71, namespace="test", identity="sample")
    )
    assert values == {"total": sum(private.values(), Decimal(0))}
    assert values["total"] != 4060
    assert len(private) == 2


def test_sampled_tare_joint_solver_keeps_fixed_aggregate():
    from document_ocr.synthesis.generators import DeterministicStream

    domain = ((PAIR, (Decimal(3700), Decimal(3800))),)
    constraints = tares.TareConstraints(
        ((0, domain), (1, domain)), Decimal(7500),
        (("a", (0,)), ("b", (1,))), (("a", Decimal(1)), ("b", Decimal(1))),
    )
    for seed in range(8):
        private, values = constraints.sample(
            constraints.options, DeterministicStream(seed=seed, namespace="test", identity="sample")
        )
        assert sum(private.values()) == 7500
        assert values == {"a": private[0], "b": private[1]}


def test_private_kg_tare_requires_explicit_scope_and_records_audited_choice():
    src = explicitly_owned_source("TARE\n3700\nABCU1234567\nDEFU9876543\n")
    c = NumericContract.model_validate({**contract().model_dump(), "synthetic_unit": "kilogram"})
    constraints = tares.compile_constraints(
        src, {"tare": c}, tares.build_support([document()]), frozenset({0, 1})
    )
    assert constraints.private_unit_bindings == ("tare",)
    assert src.source.startswith(b"TARE\n3700\n")
    src.template.bindings[0].dependency_paths = ()
    with pytest.raises(ValueError, match="ownership"):
        tares.compile_constraints(
            src, {"tare": c}, tares.build_support([document()]), frozenset({0, 1})
        )


def partial_source(description="20' FLATRACK COLLAPSIBLE"):
    row = document(observed=False)
    row["target"]["documentPatch"]["containers"][0]["typeDescription"] = description
    src = source(row)
    start = len(src.source)
    src.source += description.encode() + b"\n"
    src.template.bindings.append(NS(
        logical_key="equipment", derivation=None,
        target_paths=("documentPatch.containers[0].typeDescription",),
        occurrences=(NS(byte_start=start, byte_end=start + len(description.encode()),
                        source_text=description),),
    ))
    return src


def test_source_observed_partial_tare_retains_private_height_without_fitted_claim():
    src = partial_source()
    original = src.source, repr(src.target)
    pair = "TWENTY_FOOT_STANDARD_HEIGHT|PLATFORM_COLLAPSIBLE"
    wrong = "FORTY_FOOT_STANDARD_HEIGHT|PLATFORM_COLLAPSIBLE"
    constraints = tares.compile_constraints(
        src, {"tare": contract()}, tares.TareSupport({}, {}), frozenset({0}),
        configured_pairs=frozenset({pair, wrong, PAIR}),
    )
    assert constraints.options == {0: frozenset({pair})}
    assert constraints.source_observed_partial_owners == (0,)
    assert constraints.witness(constraints.options) == {0: Decimal(3700)}
    assert (src.source, repr(src.target)) == original
    assert "sizeCategory" not in src.target["documentPatch"]["containers"][0]


def test_certified_ocr_joined_tank_word_proves_private_tare_pair():
    src = partial_source("20 FT ISO TANKCONTAINER(S)")
    src.target["documentPatch"]["containers"][0]["typeDescription"] = (
        "20 FT ISO TANK CONTAINER(S)"
    )
    pair = "TWENTY_FOOT_STANDARD_HEIGHT|PRESSURIZED_TANK"
    constraints = tares.compile_constraints(
        src, {"tare": contract()}, tares.TareSupport({}, {}), frozenset({0}),
        configured_pairs=frozenset({pair}),
    )
    assert constraints.options == {0: frozenset({pair})}
    assert constraints.source_observed_partial_owners == (0,)
    src.template.bindings[-1].occurrences[0].source_text = "20 FT ISO TANKTRAILER(S)"
    with pytest.raises(ValueError, match="no exact train-only"):
        tares.compile_constraints(
            src, {"tare": contract()}, tares.TareSupport({}, {}), frozenset({0}),
            configured_pairs=frozenset({pair}),
        )


@pytest.mark.parametrize("change", ["unowned", "wrong_span", "kind_only", "absent_domain"])
def test_partial_observation_cannot_invent_source_tare_compatibility(change):
    src = partial_source(
        "refrigerated container" if change == "kind_only" else "20' FLATRACK COLLAPSIBLE"
    )
    pair = "TWENTY_FOOT_STANDARD_HEIGHT|PLATFORM_COLLAPSIBLE"
    if change == "unowned":
        src.template.bindings[-1].target_paths = ()
    elif change == "wrong_span":
        src.template.bindings[-1].occurrences[0].byte_start = 0
    with pytest.raises(ValueError, match="no exact train-only"):
        tares.compile_constraints(
            src, {"tare": contract()}, tares.TareSupport({}, {}), frozenset({0}),
            configured_pairs=frozenset() if change == "absent_domain" else frozenset({pair}),
        )
