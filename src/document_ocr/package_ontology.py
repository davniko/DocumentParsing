"""Reviewed internal package-code definitions, separate from printed evidence.

Loading this reference does not change extraction categories or enable aliases.
A code definition can assert more than its ordinary-language name: callers must
not treat a recognized noun as if the formal code itself were printed. Omitted
properties are unasserted, not errors, negatives, or inferred child defaults.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

Token = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")]
Code = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{2}$")]
Category = Annotated[str, StringConstraints(pattern=r"^PACKAGE_[A-Z0-9]+(?:_[A-Z0-9]+)*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
_CODE_ADAPTER = TypeAdapter(Code)
_CATEGORY_ADAPTER = TypeAdapter(Category)
_TEXT_ADAPTER = TypeAdapter(NonEmpty)
Kind = Literal[
    "PACKAGE", "CARGO_FORM", "COUNTING_UNIT", "PRESENTATION", "EQUIPMENT_FORM", "ADMINISTRATIVE"
]
ComponentRole = Literal[
    "INNER_RECEPTACLE", "OUTER_ENCLOSURE", "PALLET_BASE", "LINER", "STRUCTURAL_SUPPORT"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class PackageMeasurement(_Strict):
    """A code-definition value, not a measured shipment/equipment setting."""

    property: Literal["DIMENSIONS", "PRESSURE", "TEMPERATURE", "CAPACITY"]
    values: Annotated[tuple[float, ...], Field(min_length=1)]
    unit: Literal["CENTIMETRE", "KILOPASCAL", "MILLIBAR", "CELSIUS", "GALLON_UNSPECIFIED"] | None
    relation: Literal["EQ", "GT", "LT", "GE", "LE", "UNSPECIFIED"]

    @model_validator(mode="after")
    def valid_definition_measurement(self) -> PackageMeasurement:
        units = {
            "DIMENSIONS": {"CENTIMETRE", None},
            "PRESSURE": {"KILOPASCAL", "MILLIBAR", None},
            "TEMPERATURE": {"CELSIUS", None},
            "CAPACITY": {"GALLON_UNSPECIFIED", None},
        }
        if self.unit not in units[self.property]:
            raise ValueError("package measurement property and unit disagree")
        if self.property == "DIMENSIONS":
            if len(self.values) not in {2, 3} or any(v <= 0 for v in self.values):
                raise ValueError("package dimensions require two or three positive values")
        elif len(self.values) != 1:
            raise ValueError("scalar package measurement requires one value")
        elif self.property != "TEMPERATURE" and self.values[0] <= 0:
            raise ValueError("package capacity/pressure must be positive")
        return self


class _Component(_Strict):
    role: ComponentRole
    family: Token | None = None
    material: Token | None = None
    attributes: dict[Token, Token] = Field(default_factory=dict)


class _Properties(_Strict):
    kind: Kind
    family: Token | None
    material: Token | None = None
    materialAlternatives: tuple[Token, ...] = ()
    attributes: dict[Token, Token] = Field(default_factory=dict)
    components: tuple[_Component, ...] = ()
    measurements: tuple[PackageMeasurement, ...] = ()
    standards: tuple[NonEmpty, ...] = ()
    definitionSources: tuple[NonEmpty, ...] = ()
    notes: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def meaningful_definition(self) -> _Properties:
        if self.material is not None and self.materialAlternatives:
            raise ValueError("a definite body material cannot also have alternatives")
        if self.materialAlternatives and (
            len(self.materialAlternatives) < 2
            or len(set(self.materialAlternatives)) != len(self.materialAlternatives)
        ):
            raise ValueError("material alternatives require distinct multiple possibilities")
        roles = [component.role for component in self.components]
        if len(set(roles)) != len(roles):
            raise ValueError("component roles must be unique in a code definition")
        if len(set(self.measurements)) != len(self.measurements):
            raise ValueError("package measurements must not be duplicated")
        if len(set(self.standards)) != len(self.standards):
            raise ValueError("package standards must not be duplicated")
        if len(set(self.definitionSources)) != len(self.definitionSources):
            raise ValueError("definition sources must not be duplicated")
        if self.kind == "ADMINISTRATIVE":
            if any(
                (
                    self.family,
                    self.material,
                    self.materialAlternatives,
                    self.components,
                    self.measurements,
                    self.standards,
                )
            ):
                raise ValueError("administrative definitions cannot invent physical properties")
            if set(self.attributes) != {"STATUS"}:
                raise ValueError("administrative definitions require an explicit status")
        elif self.family is None and not self.attributes:
            raise ValueError("a nonphysical presentation must state its explicit treatment")
        if self.attributes.get("LINING") == "ABSENT" and "LINER" in roles:
            raise ValueError("an explicitly absent liner cannot also be a component")
        if self.attributes.get("CONSTRUCTION") == "COMPOSITE" and self.material is not None:
            raise ValueError("composite materials require component ownership")
        return self


class _OfficialEntry(_Properties):
    code: Code


class _SemanticEntry(_Properties):
    categoryToken: Category
    displayName: NonEmpty


_OFFICIAL_ENTRIES = TypeAdapter(tuple[_OfficialEntry, ...])
_SEMANTIC_ENTRIES = TypeAdapter(tuple[_SemanticEntry, ...])


class _Pin(_Strict):
    path: NonEmpty
    sha256: Sha256


class _PrimarySource(_Strict):
    id: NonEmpty
    url: Annotated[str, StringConstraints(pattern=r"^https://")]
    sha256: Sha256


class _Manifest(_Strict):
    schemaVersion: Literal[1]
    status: Literal["internal_reviewed_reference_not_runtime_enabled"]
    reviewedAt: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    sourceRegistry: _Pin
    officialEntryCount: Annotated[int, Field(gt=0)]
    entryFiles: Annotated[tuple[_Pin, ...], Field(min_length=1)]
    semanticExtensions: _Pin
    requiredActiveTask: _Pin
    requiredAdditionalFamilies: tuple[Token, ...]
    primarySources: tuple[_PrimarySource, ...]
    attributeDefinitions: dict[Token, NonEmpty]
    attributeVocabulary: dict[Token, tuple[Token, ...]]
    notes: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def inventories_are_unique(self) -> _Manifest:
        if len({p.path for p in self.entryFiles}) != len(self.entryFiles):
            raise ValueError("crosswalk shard paths must be unique")
        if len({s.id for s in self.primarySources}) != len(self.primarySources):
            raise ValueError("primary source identities must be unique")
        if self.attributeDefinitions.keys() != self.attributeVocabulary.keys():
            raise ValueError("every attribute needs a reviewed definition and vocabulary")
        if any(not vs or len(set(vs)) != len(vs) for vs in self.attributeVocabulary.values()):
            raise ValueError("attribute value inventories must be nonempty and unique")
        return self


@dataclass(frozen=True, slots=True)
class PackageComponent:
    role: ComponentRole
    family: str | None
    material: str | None
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PackageProperties:
    kind: Kind
    family: str | None
    material: str | None
    material_alternatives: tuple[str, ...]
    attributes: Mapping[str, str]
    components: tuple[PackageComponent, ...]
    measurements: tuple[PackageMeasurement, ...]
    standards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PackageDefinition:
    code: str | None
    category_token: str
    display_name: str
    properties: PackageProperties
    definition_sources: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PackageOntology:
    """Immutable reference; lookups are exact codes/categories, never text aliases."""

    by_code: Mapping[str, PackageDefinition]
    by_category: Mapping[str, PackageDefinition]
    reviewed_at: str
    manifest_sha256: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property: {key}")
        result[key] = value
    return result


def _json_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    json.loads(data, object_pairs_hook=_unique_object)
    return data


def _pinned_bytes(root: Path, pin: _Pin) -> bytes:
    path = (root / pin.path).resolve()
    if Path(pin.path).is_absolute() or not path.is_relative_to(root):
        raise ValueError("crosswalk reference must remain inside its declared root")
    data = _json_bytes(path)
    if hashlib.sha256(data).hexdigest() != pin.sha256:
        raise ValueError(f"crosswalk input hash mismatch: {pin.path}")
    return data


def _properties(row: _Properties) -> PackageProperties:
    return PackageProperties(
        row.kind,
        row.family,
        row.material,
        row.materialAlternatives,
        MappingProxyType(dict(row.attributes)),
        tuple(
            PackageComponent(c.role, c.family, c.material, MappingProxyType(dict(c.attributes)))
            for c in row.components
        ),
        row.measurements,
        row.standards,
    )


def load_package_ontology(manifest_path: Path, *, project_root: Path) -> PackageOntology:
    """Validate all definitions, provenance and required coverage; perform no writes."""
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_bytes = _json_bytes(manifest_path)
    manifest = _Manifest.model_validate_json(manifest_bytes)
    registry = json.loads(_pinned_bytes(project_root, manifest.sourceRegistry))
    if registry.get("registryKind") != "package":
        raise ValueError("crosswalk authority must be the package registry")
    authority: dict[str, dict[str, str]] = {}
    for record in registry["entries"]:
        code = _CODE_ADAPTER.validate_python(record["applicationCode"])
        _CATEGORY_ADAPTER.validate_python(record["categoryToken"])
        _TEXT_ADAPTER.validate_python(record["displayName"])
        if code in authority:
            raise ValueError("duplicate code in crosswalk authority")
        authority[code] = record
    if len(authority) != manifest.officialEntryCount:
        raise ValueError("authoritative crosswalk code count changed")
    sources = {source.id for source in manifest.primarySources}
    definitions: dict[str, PackageDefinition] = {}

    def add(category: str, name: str, row: _Properties, code: str | None) -> None:
        if category in definitions:
            raise ValueError("duplicate crosswalk category")
        if set(row.definitionSources) - sources:
            raise ValueError("crosswalk entry uses an undeclared primary source")
        for attributes in (row.attributes, *(c.attributes for c in row.components)):
            for key, value in attributes.items():
                if (
                    key not in manifest.attributeVocabulary
                    or value not in manifest.attributeVocabulary[key]
                ):
                    raise ValueError(f"unreviewed crosswalk attribute: {key}={value}")
        definitions[category] = PackageDefinition(
            code, category, name, _properties(row), row.definitionSources, row.notes
        )

    visited: set[str] = set()
    for pin in manifest.entryFiles:
        payload = _pinned_bytes(manifest_path.parent, pin)
        for row in _OFFICIAL_ENTRIES.validate_json(payload):
            if row.code not in authority or row.code in visited:
                raise ValueError(f"unknown or duplicate crosswalk code: {row.code}")
            visited.add(row.code)
            source = authority[row.code]
            add(source["categoryToken"], source["displayName"], row, row.code)
    if visited != authority.keys():
        raise ValueError(f"crosswalk coverage missing codes: {sorted(authority.keys() - visited)}")
    semantic = _pinned_bytes(manifest_path.parent, manifest.semanticExtensions)
    for semantic_row in _SEMANTIC_ENTRIES.validate_json(semantic):
        add(semantic_row.categoryToken, semantic_row.displayName, semantic_row, None)
    constraints = json.loads(_pinned_bytes(project_root, manifest.requiredActiveTask))
    if set(constraints["packageCategoryTokens"]) - definitions.keys():
        raise ValueError("crosswalk omits a current active package category")
    families = {d.properties.family for d in definitions.values()}
    if set(manifest.requiredAdditionalFamilies) - families:
        raise ValueError("crosswalk omits a required additional pilot family")
    return PackageOntology(
        MappingProxyType({d.code: d for d in definitions.values() if d.code is not None}),
        MappingProxyType(definitions),
        manifest.reviewedAt,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _material_supports(
    material: str | None, alternatives: tuple[str, ...], required: PackageProperties
) -> bool:
    if required.material is not None:
        return material == required.material
    if not required.material_alternatives:
        return True
    allowed = set(required.material_alternatives)
    if material is not None:
        return material in allowed
    return bool(alternatives) and set(alternatives) <= allowed


def definition_includes(evidence: PackageProperties, required: PackageProperties) -> bool:
    """Compare explicit assertions, not aliases, document links or physical defaults.

    Omitted required properties impose no requirement. Omitted evidence cannot
    establish a specific value. No implicit material taxonomy, unit conversion,
    numeric inequality theorem proving or cross-component material transfer occurs.
    The caller must establish same-package ownership before applying this relation
    to any printed evidence; this function itself compares definition objects only.
    """
    if evidence.kind != required.kind:
        return False
    if required.family is not None and evidence.family != required.family:
        return False
    if not _material_supports(evidence.material, evidence.material_alternatives, required):
        return False
    if any(evidence.attributes.get(k) != v for k, v in required.attributes.items()):
        return False
    available = {component.role: component for component in evidence.components}
    for component in required.components:
        actual = available.get(component.role)
        if actual is None:
            return False
        if component.family is not None and actual.family != component.family:
            return False
        if component.material is not None and actual.material != component.material:
            return False
        if any(actual.attributes.get(k) != v for k, v in component.attributes.items()):
            return False
    return set(required.measurements) <= set(evidence.measurements) and set(
        required.standards
    ) <= set(evidence.standards)
