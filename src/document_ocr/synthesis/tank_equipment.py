"""Installed-tank synthesis contracts; never dry-box or flexitank defaults.

Conservative planning bounds, not universal ratings or DG packing approvals.
20-foot liquid tanks: Eurotainer's standard-liquid fleet; 20/40-foot, 2,591-mm
models: Chart's ISO brochure. BIC separates KL, NH and NN from numeric pressure.
https://www.eurotainer.com/containers/tank-containers-for-liquids/standard-liquid/
https://files.chartindustries.com/chartferox_ISO_Containers.pdf
https://www.bic-code.org/type-code-designation/
"""

from decimal import Decimal
from types import MappingProxyType

TANK_TYPES = frozenset({"PRESSURIZED_TANK", "DRY_HOPPER_TANK", "DRY_REAR_DISCHARGE_TANK"})
LIQUID_TANK_TYPE = "PRESSURIZED_TANK"
TANK_COUNTED_PACKAGE_TYPES = frozenset({"PACKAGE_PACKAGE", "PACKAGE_LOT"})
TANK_SIZES = ("TWENTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_STANDARD_HEIGHT")

# Below the documented 20-foot 26.975 t / 20.080 m3 and 40-foot 22.5 t /
# 43.5 m3 reference pairs. Not a fabricated tank-specific nameplate rating.
TANK_CAPACITIES = MappingProxyType(
    {
        "TWENTY_FOOT_STANDARD_HEIGHT": (Decimal("26000"), Decimal("20")),
        "FORTY_FOOT_STANDARD_HEIGHT": (Decimal("22000"), Decimal("40")),
    }
)


def tank_capacity(size: str) -> tuple[Decimal, Decimal]:
    try:
        return TANK_CAPACITIES[size]
    except KeyError as error:
        raise ValueError(
            "tank size has no reviewed synthesis capacity contract: " + str(size)
        ) from error
