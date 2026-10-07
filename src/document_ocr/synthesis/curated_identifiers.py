"""Explicit fictional identifier formats for reviewed synthesis regions.

VIN characters and check-digit arithmetic follow 49 CFR 565.15. Generated
values are fictional format examples, not manufacturer-issued identities or
certification of a make, model, plant, country, or actual model year.
Reference: https://www.ecfr.gov/current/title-49/section-565.15
"""

from __future__ import annotations

import re

from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.generators import DeterministicStream

_VIN_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
_YEAR_LETTERS = "ABCDEFGHJKLMNPRSTVWXY"
_VIN_PATTERN = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
_VALUES = dict(
    zip(
        _VIN_LETTERS,
        (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9),
        strict=True,
    )
)


def vin_check_digit(value: str) -> str:
    """Compute the ninth character; reject malformed 17-character inputs."""
    if not _VIN_PATTERN.fullmatch(value):
        raise ValueError("VIN requires 17 uppercase characters without I, O, or Q")
    total = sum(
        (int(char) if char.isdigit() else _VALUES[char]) * weight
        for char, weight in zip(value, _WEIGHTS, strict=True)
    )
    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def generate_fictional_vin(source: str, stream: DeterministicStream) -> str:
    """Generate a new VIN-shaped value with a valid checksum and year code.

    Preserve the observed letter/digit layout except the check-digit position,
    which may become X. Deliberately change the final serial character, so a
    source identifier cannot be copied even when other random choices coincide.
    No manufacturer or vehicle-attribute correspondence is asserted.
    """
    vin_check_digit(source)
    generated = []
    for index, char in enumerate(source):
        if index == 8:
            generated.append("0")
            continue
        alphabet = "0123456789" if char.isdigit() else _VIN_LETTERS
        if index == 9:
            alphabet = "123456789" if char.isdigit() else _YEAR_LETTERS
        if index == 16:
            alphabet = alphabet.replace(char, "")
        generated.append(alphabet[stream.derive(f"vin:{index}").randbelow(len(alphabet))])
    generated[8] = vin_check_digit("".join(generated))
    return "".join(generated)


def declared_identifier_values(
    blueprint: SamplingBlueprint, stream: DeterministicStream
) -> tuple[dict[str, str], list[dict]]:
    """Return only explicitly declared identifier values and their scope receipts."""
    variables = {variable.key: variable for variable in blueprint.contract.variables}
    values, receipts = {}, []
    for key, policy in blueprint.ownership_data.get("identifier_policies", {}).items():
        if policy != "fictional_vin_v1":
            raise ValueError(f"unknown declared identifier policy: {key}/{policy}")
        variable = variables.get(key)
        if variable is None or variable.kind != "identifier":
            raise ValueError(f"VIN policy requires an existing identifier variable: {key}")
        if any(occurrence.text != variable.value for occurrence in variable.occurrences):
            raise ValueError(f"VIN variable does not own its exact printed source: {key}")
        value = generate_fictional_vin(variable.value, stream.derive(key))
        values[key] = value
        receipts.append(
            {
                "key": key,
                "policy": policy,
                "before": variable.value,
                "after": value,
                "validCharacterSet": True,
                "validCheckDigit": True,
                "manufacturerIssuedIdentity": False,
                "vehicleAttributeCertification": False,
            }
        )
    return values, receipts
