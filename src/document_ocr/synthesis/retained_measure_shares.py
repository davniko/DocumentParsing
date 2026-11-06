"""Exact verification of proportional measures serialized as Decimal strings."""

from decimal import Decimal
from fractions import Fraction


def has_exact_proportional_total(rows, field, total):
    """Verify the original sampler expression and the exact rational share sum.

    A finite Decimal representation of one third cannot sum to exactly one.
    Recorded integer shares establish conservation without a numeric tolerance.
    This must never validate newly proposed rows or arbitrary near-equal totals.
    """
    shares = []
    if not rows or not total.is_finite() or total <= 0:
        return False
    for row in rows:
        numerator, denominator = row.get('shareNumerator'), row.get('shareDenominator')
        if (type(numerator) is not int or type(denominator) is not int
                or numerator <= 0 or denominator <= 0):
            return False
        measure = row['measures'][field]
        if (Decimal(str(measure['total'])) != total
                or Decimal(str(measure['value'])) != total * numerator / denominator):
            return False
        shares.append(Fraction(numerator, denominator))
    return sum(shares, Fraction(0)) == 1
