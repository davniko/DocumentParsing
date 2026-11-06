"""Reconcile an old private row projection with its final physical certificate.

The scenario sampler rounds public values before the physical compiler sees all
printed repetitions. The latter can recover finer precision from a registry
profile. This migration changes only the stale private projection, not cargo,
printed values, targets or the authoritative profile.
"""

import copy
from decimal import Decimal, ROUND_HALF_UP

from document_ocr.synthesis.paddle_render import _UNITS


def reconcile_profile_rows(candidate):
    p = candidate['scenario']['provenance']
    if p['method'] != 'registry_goods_train_physical_support_geography_v2':
        raise ValueError('requires retained registry sampling contract')
    profile = p['registryLoadProfile']
    final, = candidate['target']['documentPatch']['goodsItemDetails']
    initial, = candidate['scenario']['cargo']['goodsItemDetails']
    package, = final['numberAndTypeOfPackages']
    count = package['packageQuantity']
    rows = p['physicalRows']
    source_ids = [x['equipmentIdentifier'] for x in candidate['scenario']['cargo']['containerInformation']]
    if (package['typeCategory'] != profile['package_category'] or type(count) is not int
            or count <= 0 or not rows or len(set(source_ids)) != len(source_ids)
            or sorted(source_ids) != sorted(r['equipmentIdentifier'] for r in rows)
            or any(type(r['quantity']) is not int or r['quantity'] <= 0 for r in rows)
            or sum(r['quantity'] for r in rows) != count):
        raise ValueError('profile inventory or packages disagree')
    result = copy.deepcopy(candidate)
    changed = False
    for field, key, canonical_unit in [('grossWeight', 'unit_gross_kg', 'kilogram'),
                                      ('netWeight', 'unit_net_kg', 'kilogram'),
                                      ('volume', 'unit_volume_m3', 'cubic_metre')]:
        present = [field in r['measures'] for r in rows]
        if not any(present):
            continue
        if not all(present):
            raise ValueError('partial retained measure projection')
        certificate = candidate['physical']['measures'][field]
        per_unit = Decimal(str(profile[key]))
        exact = per_unit * count
        fact = p['explicitPhysicalFacts'][field]
        if not per_unit.is_finite() or per_unit <= 0 or fact['unit'] != canonical_unit or Decimal(str(fact['value'])) != exact:
            raise ValueError('profile disagrees with explicit authority')
        initial_measure, final_measure = initial[field], final[field]
        unit = certificate['unit']
        if any(m['unit'] != unit for m in [initial_measure, final_measure, *[r['measures'][field] for r in rows]]):
            raise ValueError('retained unit ownership differs')
        if _UNITS[unit][0] != _UNITS[canonical_unit][0]:
            raise ValueError('retained measure changes dimension')
        factor = _UNITS[unit][1] / _UNITS[canonical_unit][1]
        quantum = Decimal(certificate['quantum'])
        before = Decimal(certificate['beforeRounding'])
        total = Decimal(certificate['total'])
        if (before * factor != exact or not quantum.is_finite() or quantum <= 0
                or before.quantize(quantum, rounding=ROUND_HALF_UP) != total
                or total != Decimal(str(final_measure['value'])) or total * factor != exact):
            raise ValueError('final physical certificate does not preserve exact profile')
        previous_total = Decimal(str(initial_measure['value']))
        for old, new in zip(rows, result['scenario']['provenance']['physicalRows'], strict=True):
            m = old['measures'][field]
            # Verify the original sampler's actual expression, including its
            # Decimal operation order, rather than allowing a numeric tolerance.
            if (old['shareNumerator'] != old['quantity'] or old['shareDenominator'] != count
                    or Decimal(m['total']) != previous_total
                    or Decimal(m['value']) != previous_total * old['shareNumerator'] / old['shareDenominator']):
                raise ValueError('row is not the old proportional scenario projection')
            replacement = {'value': str(per_unit * old['quantity'] / factor),
                           'total': str(total), 'unit': unit}
            if any(Decimal(m[k]) != Decimal(replacement[k]) for k in ('value', 'total')):
                new['measures'][field] = replacement
                changed = True
    if not changed:
        raise ValueError('no stale row projection to reconcile')
    return result
