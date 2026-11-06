"""Recover private counts already implied by a retained uniform-unit ledger.

This is a synthetic receipt projection, never inference over a real document.
Fractional, conflicting or absent unit evidence must be handled by a separately
reviewed cargo refresh, not rounded into apparently valid counts.
"""

import copy
from decimal import Decimal

from r30_missing_numeric import amount


def complete_profile_counts(candidate):
    provenance = candidate["scenario"]["provenance"]
    if provenance["method"] != "registry_goods_train_physical_support_geography_v2":
        raise ValueError("profile projection requires the retained sampling contract")
    profile = provenance["registryLoadProfile"]
    goods = candidate["target"]["documentPatch"]["goodsItemDetails"]
    if len(goods) != 1:
        raise ValueError("profile projection requires one goods item")
    packages = goods[0]["numberAndTypeOfPackages"]
    if len(packages) != 1 or packages[0]["typeCategory"] != profile["package_category"]:
        raise ValueError("profile category differs from declared packages")
    rows = provenance["physicalRows"]
    source_ids = [r["equipmentIdentifier"] for r in candidate["scenario"]["cargo"]["containerInformation"]]
    ids = [r["equipmentIdentifier"] for r in rows]
    if not rows or len(ids) != len(source_ids) or len(set(ids)) != len(ids) or set(ids) != set(source_ids):
        raise ValueError("profile allocation inventory differs")
    fields = {"grossWeight": ("unit_gross_kg", "kilogram"),
              "netWeight": ("unit_net_kg", "kilogram"),
              "volume": ("unit_volume_m3", "cubic_metre")}
    counts = {}
    for row in rows:
        inferred = set()
        for field, measure in row["measures"].items():
            key, unit = fields[field]
            per_unit = Decimal(str(profile[key]))
            if not per_unit.is_finite() or per_unit <= 0:
                raise ValueError("profile unit fact must be finite and positive")
            value = amount(measure["value"], measure["unit"], unit)
            total = amount(measure["total"], measure["unit"], unit)
            if total != per_unit * packages[0]["packageQuantity"]:
                raise ValueError("profile unit fact disagrees with retained total")
            quantity = value / per_unit
            if quantity != quantity.to_integral_value() or quantity <= 0:
                raise ValueError("profile row implies a fractional or nonpositive count")
            inferred.add(int(quantity))
        if len(inferred) != 1:
            raise ValueError("profile row counts lack unique agreeing measures")
        quantity = inferred.pop()
        if row["quantity"] is not None and (type(row["quantity"]) is not int or row["quantity"] != quantity):
            raise ValueError("profile count disagrees with existing private count")
        counts[row["equipmentIdentifier"]] = quantity
    if sum(counts.values()) != packages[0]["packageQuantity"]:
        raise ValueError("profile row counts do not conserve packages")
    result = copy.deepcopy(candidate)
    for row in result["scenario"]["provenance"]["physicalRows"]:
        if row["quantity"] is None:
            row["quantity"] = counts[row["equipmentIdentifier"]]
    return result
