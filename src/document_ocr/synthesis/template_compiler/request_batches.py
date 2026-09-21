"""Lossless shared-context batching with named fields and attributable receipts."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby
from typing import Annotated, Any, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt


class BatchMemberError(RuntimeError):
    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(receipt["error"])


_BATCH_RULES = """
Complete each independent case below using the original rules. sharedContext is a
base context. Replace the paths listed in varyingPaths with the corresponding
case.values (same order). Paths are arrays of object keys or zero-based indexes.
Return one object per case, keyed by its exact case key. Each object must use the
named field keys; never exchange party roles or case facts. Cases must have
independently varied identities, not numbered copies of the same organization.
No leading/trailing whitespace or newline characters in generated field values.
All names/addresses must use Latin-script letters (transliterate if necessary).
When structured party locality is absent, preserve the locality/country explicitly
present in that party's source address; a loading port is not its postal locality.
Structured {generate: key} entries refer to the requested new field, whose source
is in requestedFields. Missing rendering-constraint keys impose no extra restriction.
Numeric auxiliary entries contain host-computed values; never recompute them.
All additionalInformation and marksAndNumbers entries within a cargo group must
remain distinct. Preserve each mark's separate role; do not collapse different
source marks or spelling variants into identical output strings.
Do not add package counts, weights or volumes to a field that did not express
them in the source. They are already represented in their own host-owned slots.
Do not copy other requested source fields when repairing just one rejected field.
When a field has hostAssembly, return ONLY its new mutable portion, with the
specified minimumWords. The host inserts the exact immutable prefix and suffix;
do not include, rewrite, or duplicate those fixed components in your answer.
"""
_TEXT = Annotated[str, Field(min_length=1, pattern=r"^\S(?:[^\r\n]*\S)?$")]


class _WaveReadiness:
    """Flush when all siblings are waiting or finished, not on a timing guess."""

    def __init__(self, members: set[str]) -> None:
        self.active = set(members)
        self.finished: set[str] = set()
        self.flushes: dict[tuple[int, str], Callable[[], None]] = {}

    def flush_if_ready(self) -> None:
        if not self.active:
            pending, self.flushes = self.flushes, {}
            for flush in pending.values():
                flush()

    def ready(self, member: str) -> None:
        if member not in self.finished:
            self.active.add(member)

    def finish(self, member: str) -> None:
        self.finished.add(member)
        self.active.discard(member)
        self.flush_if_ready()


_WAVE: ContextVar[tuple[_WaveReadiness, str] | None] = ContextVar("batch_wave", default=None)


async def run_template_waves(
    samples: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    workers: int,
    execute: Callable[[dict[str, Any]], Awaitable[None]],
    should_stop: Callable[[], bool],
) -> None:
    """Admit bounded sibling groups together, even after unequal request latency.

    Independent document workers fragment later sibling groups as they finish at
    different times. A wave retains their admission boundary without delaying the
    batcher's partial flush or exceeding workers * batch_size live documents.
    """
    if batch_size < 1 or workers < 1:
        raise ValueError("template waves require positive batch size and worker count")
    pending: asyncio.Queue[list[dict[str, Any]]] = asyncio.Queue()
    for _source, siblings in groupby(
        sorted(samples, key=lambda row: row["sourceDocumentId"]),
        key=lambda row: row["sourceDocumentId"],
    ):
        group = list(siblings)
        for offset in range(0, len(group), batch_size):
            pending.put_nowait(group[offset : offset + batch_size])

    async def worker() -> None:
        while not pending.empty() and not should_stop():
            wave = pending.get_nowait()
            readiness = _WaveReadiness({row["sampleId"] for row in wave})

            async def member(row: dict[str, Any], readiness: _WaveReadiness = readiness) -> None:
                token = _WAVE.set((readiness, row["sampleId"]))
                try:
                    await execute(row)
                finally:
                    readiness.finish(row["sampleId"])
                    _WAVE.reset(token)

            await asyncio.gather(*(member(row) for row in wave))
            pending.task_done()

    await asyncio.gather(*(worker() for _ in range(min(workers, pending.qsize()))))


def validate_batch_member(
    *, provenance: Mapping[str, Any], output: Any, messages: Any, usage: Any
) -> None:
    """Prove the member projection and its share of the real provider receipt."""
    required = {
        "requestSha256",
        "outputSha256",
        "member",
        "members",
        "usageAllocation",
        "fieldAliases",
        "parentUsage",
    }
    if set(provenance) != required:
        raise ValueError("batch provenance fields are incomplete or unknown")
    count = provenance["members"]
    member = provenance["member"]
    if not 1 <= count <= 16 or member not in {f"s{i}" for i in range(count)}:
        raise ValueError("invalid batch member identity")
    if provenance["usageAllocation"] != "equal_share":
        raise ValueError("unknown batch usage allocation")
    responses = [message for message in messages if message["kind"] == "response"]
    if len(responses) != 1:
        raise ValueError("batched receipt must contain exactly one provider response")
    response = responses[0]
    text = "".join(p["content"] for p in response["parts"] if p["part_kind"] == "text")
    parent = json.loads(text)
    if (
        set(parent) != {f"s{i}" for i in range(count)}
        or sha256_bytes(canonical_json_bytes(parent)) != provenance["outputSha256"]
    ):
        raise ValueError("batch response hash or membership differs")
    aliases = provenance["fieldAliases"]
    if (
        set(aliases.values()) != set(parent[member])
        or {key: parent[member][alias] for key, alias in aliases.items()} != output
    ):
        raise ValueError("batch member output differs from provider response")
    parent_usage = provenance["parentUsage"]
    LinguisticUsageReceipt.model_validate_json(canonical_json_bytes(parent_usage), strict=True)
    if parent_usage["requests"] != 1 or parent_usage["providerResponseIds"] != [
        response["provider_response_id"]
    ]:
        raise ValueError("batch parent request identity differs")
    if allocated_usage(parent_usage, int(member[1:]), count) != usage:
        raise ValueError("batch member usage allocation differs")


def _leaves(value: Any, path: tuple[str | int, ...] = ()) -> dict[tuple[str | int, ...], Any]:
    if isinstance(value, dict) and value:
        return {p: v for k, child in value.items() for p, v in _leaves(child, (*path, k)).items()}
    if isinstance(value, list) and value:
        return {
            p: v for i, child in enumerate(value) for p, v in _leaves(child, (*path, i)).items()
        }
    return {path: value}


def shared_context(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("cannot batch empty requests")
    flat = [_leaves(p) for p in payloads]
    if any(set(row) != set(flat[0]) for row in flat[1:]):
        raise ValueError("batch inputs have different topologies")
    paths = [p for p in flat[0] if any(row[p] != flat[0][p] for row in flat[1:])]
    result: dict[str, Any] = dict(
        sharedContext=deepcopy(payloads[0]),
        varyingPaths=[list(p) for p in paths],
        cases={f"s{i}": dict(values=[row[p] for p in paths]) for i, row in enumerate(flat)},
    )
    # This is a cheap exact check, not an inference about which context is relevant.
    for case, expected in zip(result["cases"].values(), payloads, strict=True):
        actual = deepcopy(result["sharedContext"])
        for path, value in zip(paths, case["values"], strict=True):
            node = actual
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
        if actual != expected:
            raise ValueError("shared-context roundtrip changed a scenario")
    return result


def semantic_aliases(payload: Mapping[str, Any], fields: Sequence[str]) -> dict[str, str]:
    if "residualBindings" in payload:
        # Repeated physical occurrences of the same semantic owner and exact
        # formatting contract need one generated value, not N repeated strings.
        # Never deduplicate across owners or distinct source/format contracts.
        aliases: dict[str, str] = {}
        for binding in payload["residualBindings"]:
            unique: dict[bytes, str] = {}
            for slot in binding["slots"]:
                signature = canonical_json_bytes({k: v for k, v in slot.items() if k != "slotId"})
                aliases[slot["slotId"]] = unique.setdefault(signature, slot["slotId"])
        if set(aliases) != set(fields):
            raise ValueError("residual alias ownership differs from requested fields")
        return aliases
    requests = {r["key"]: r for r in payload.get("requestedFields", ())}
    aliases = {}
    for key in fields:
        if key in requests:
            row = requests[key]
            path = (
                (
                    row["cargoFragment"]["targetPaths"][0]
                    + "_part_"
                    + str(row["cargoFragment"]["index"])
                )
                if "cargoFragment" in row
                else row["paths"][0]
                if row["paths"]
                else row["auxiliaryKey"]
            )
            path = path.removeprefix("documentPatch.").removeprefix("parties.")
            for original, short in (
                ("cargoGroups", "cargo"),
                ("notifyParties", "notify"),
                ("additionalInformation", "info"),
                ("marksAndNumbers", "marks"),
                ("forwardingAndExportReferences", "exportReference"),
                ("contactDetails.", ""),
                ("phoneNumbers", "phone"),
                ("emailAddresses", "email"),
                ("websiteUrls", "website"),
            ):
                path = path.replace(original, short)
            aliases[key] = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")
        else:
            aliases[key] = key
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("semantic response field aliases collide")
    return aliases


def lexical_payload(payload: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Send lexical facts and rendering requirements, not host audit internals.

    Generated target values are references to their complete source-field entries,
    not duplicated facts. No numeric scenario value, relationship, required literal,
    slot surface or minimum-length requirement is dropped.
    """
    result = deepcopy(dict(payload))
    for requirement in result.get("repairRequirements", ()):
        if "key" in requirement:
            requirement["key"] = aliases[requirement["key"]]
    if "residualBindings" in result:
        for binding in result["residualBindings"]:
            unique_slots = {}
            for slot in binding["slots"]:
                alias = aliases[slot["slotId"]]
                if alias not in unique_slots:
                    unique_slots[alias] = {**slot, "slotId": alias}
            binding["slots"] = list(unique_slots.values())
    fragment_fields: dict[str, list[str]] = {}
    for field in result.get("requestedFields", ()):
        field["key"] = aliases[field["key"]]
        if "cargoFragment" in field:
            for path in field["cargoFragment"]["targetPaths"]:
                fragment_fields.setdefault(path, []).append(field["key"])
        if "hostAssembly" in field:
            field["constraints"] = [{"minimumWords": field["hostAssembly"]["minimumWords"]}]
            field["source"] = field["hostAssembly"]["mutableSource"]
        unique = {}
        for constraint in field["constraints"]:
            trimmed = {k: v for k, v in constraint.items() if v not in (None, [], {})}
            unique[canonical_json_bytes(trimmed)] = trimmed
        field["constraints"] = list(unique.values())
        for path in field["paths"]:
            node = result["structuredScenario"]
            keys = re.findall(r"[^.\[\]]+", path)
            for key in keys[:-1]:
                node = node[int(key) if isinstance(node, list) else key]
            key = int(keys[-1]) if isinstance(node, list) else keys[-1]
            node[key] = {"generate": field["key"]}
    for path, fields in fragment_fields.items():
        node = result["structuredScenario"]
        keys = re.findall(r"[^.\[\]]+", path)
        for key in keys[:-1]:
            node = node[int(key) if isinstance(node, list) else key]
        key = int(keys[-1]) if isinstance(node, list) else keys[-1]
        node[key] = {"generateParts": fields}
    if "numericAuxiliary" in result:
        result["numericAuxiliary"] = {
            key: {
                "value": row["value"],
                "role": row["contract"]["role"],
                "mode": row["contract"]["mode"],
                "sourceValue": row["contract"]["source_value"],
                "targetPaths": row["contract"]["target_paths"],
            }
            for key, row in result["numericAuxiliary"].items()
        }
    if "priorOutput" in result:
        aliased_output: dict[str, str] = {}
        for key, value in result["priorOutput"].items():
            alias = aliases[key]
            if alias in aliased_output and aliased_output[alias] != value:
                raise ValueError("repeated residual occurrences disagree in prior output")
            aliased_output[alias] = value
        result["priorOutput"] = aliased_output
    return result


def allocated_usage(usage: Mapping[str, Any], index: int, count: int) -> dict[str, Any]:
    """Equal cost attribution; totals equal the single real provider receipt exactly."""

    def share(n: int) -> int:
        quotient, remainder = divmod(n, count)
        return quotient + int(index < remainder)

    result = dict(usage)
    for key in ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens", "visibleOutputTokens"):
        result[key] = share(usage[key])
    result["inputTokens"] = (
        result["cacheReadTokens"]
        + result["cacheWriteTokens"]
        + share(usage["inputTokens"] - usage["cacheReadTokens"] - usage["cacheWriteTokens"])
    )
    result["outputTokens"] = result["reasoningTokens"] + result["visibleOutputTokens"]
    if usage["providerTokenAccountingAnomaly"]:
        result["outputTokens"] = share(usage["outputTokens"])
    for key in ("estimatedCostUsd", "providerReportedCostUsd"):
        if usage[key] is not None:
            result[key] = str(Decimal(share(int(Decimal(usage[key]) * 10**12))) / 10**12)
    result["requests"] = usage["requests"] if index == 0 else 0
    result["providerResponseIds"] = usage["providerResponseIds"] if index == 0 else []
    result["finishReasons"] = usage["finishReasons"] if index == 0 else []
    return result


@dataclass
class _Pending:
    payload: Mapping[str, Any]
    output_type: type[BaseModel]
    future: asyncio.Future[dict[str, Any]]
    wave: tuple[_WaveReadiness, str] | None


class RequestBatcher:
    def __init__(self, *, size: int, submit: Callable[..., Awaitable[dict[str, Any]]]) -> None:
        if size < 2:
            raise ValueError("batch size must be at least two")
        self.size = size
        self.submit = submit
        self.pending: dict[str, list[_Pending]] = {}
        self.timers: dict[str, asyncio.Handle] = {}
        self.tasks: set[asyncio.Task[None]] = set()

    def member_contract(
        self, payload: Mapping[str, Any], output_type: type[BaseModel]
    ) -> dict[str, Any]:
        # Cache identity describes the full semantic slot request, not wire-level
        # deduplication. Existing paid responses still face full schema/provenance
        # checks and current materialization validation when reused.
        aliases = (
            {key: key for key in output_type.model_fields}
            if "residualBindings" in payload
            else semantic_aliases(payload, tuple(output_type.model_fields))
        )
        return {
            "receiptVersion": 2,
            "maximumBatchSize": self.size,
            "rules": _BATCH_RULES,
            "aliases": aliases,
            "payload": lexical_payload(payload, aliases),
        }

    async def request(
        self,
        *,
        group: str,
        payload: Mapping[str, Any],
        output_type: type[BaseModel],
        system_prompt: str,
    ) -> dict[str, Any]:
        key = sha256_bytes(
            canonical_json_bytes(
                [group, system_prompt, output_type.model_json_schema(), list(_leaves(payload))]
            )
        )
        rows = self.pending.setdefault(key, [])
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        wave = _WAVE.get()
        rows.append(_Pending(payload, output_type, future, wave))
        if len(rows) == self.size:
            self._flush(key, system_prompt)
        if wave is not None:
            readiness, member = wave
            readiness.flushes[(id(self), key)] = lambda: self._flush(key, system_prompt)
            readiness.active.discard(member)
            readiness.flush_if_ready()
        elif key not in self.timers and key in self.pending:
            # A zero-delay timer gathers the current ready-worker wave without
            # waiting for a full batch or deadlocking the final partial group.
            self.timers[key] = asyncio.get_running_loop().call_later(
                0, self._flush, key, system_prompt
            )
        return await future

    def _flush(self, key: str, system_prompt: str) -> None:
        timer = self.timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        rows = self.pending.pop(key, [])
        if not rows:
            return
        task = asyncio.create_task(self._execute(rows, system_prompt))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _execute(self, rows: list[_Pending], system_prompt: str) -> None:
        try:
            fields = tuple(rows[0].output_type.model_fields)
            aliases = semantic_aliases(rows[0].payload, fields)
            payloads = []
            for row in rows:
                if semantic_aliases(row.payload, fields) != aliases:
                    raise ValueError("batch semantic field ownership differs")
                payload = lexical_payload(row.payload, aliases)
                payloads.append(payload)
            member_fields: dict[str, Any] = {
                alias: (
                    Annotated[
                        str, *get_args(_TEXT)[1:], *rows[0].output_type.model_fields[key].metadata
                    ],
                    ...,
                )
                for key, alias in aliases.items()
            }
            members = create_model(
                "NamedScenarioValues",
                __config__=ConfigDict(extra="forbid", strict=True),
                **member_fields,
            )
            batch_fields: dict[str, Any] = {f"s{i}": (members, ...) for i in range(len(rows))}
            schema = create_model(
                "IndependentScenarios",
                __config__=ConfigDict(extra="forbid", strict=True),
                **batch_fields,
            )
            receipt = await self.submit(
                payload=shared_context(payloads),
                output_type=schema,
                system_prompt=system_prompt + _BATCH_RULES,
            )
            # Every child is validated before resolving any future, so a malformed
            # batch cannot make some members look complete and lose the others.
            outputs = [
                {key: receipt["output"][f"s{i}"][alias] for key, alias in aliases.items()}
                for i in range(len(rows))
            ]
            for row, output in zip(rows, outputs, strict=True):
                row.output_type.model_validate_json(canonical_json_bytes(output), strict=True)
            # Mark the entire returned batch runnable before waking any sibling;
            # otherwise its first fast preparation could prematurely flush alone.
            for row in rows:
                if row.wave is not None:
                    row.wave[0].ready(row.wave[1])
            for i, (row, output) in enumerate(zip(rows, outputs, strict=True)):
                child = {
                    **receipt,
                    "output": output,
                    "outputSha256": sha256_bytes(canonical_json_bytes(output)),
                    "usage": allocated_usage(receipt["usage"], i, len(rows)),
                    "batchProvenance": {
                        "requestSha256": receipt["requestSha256"],
                        "outputSha256": receipt["outputSha256"],
                        "member": f"s{i}",
                        "members": len(rows),
                        "usageAllocation": "equal_share",
                        "fieldAliases": aliases,
                        "parentUsage": receipt["usage"],
                    },
                }
                # The original batch transcript is retained verbatim. Do not invent
                # a per-member provider response or bill its full cost per member.
                if not row.future.cancelled():
                    row.future.set_result(child)
        except Exception as error:
            for row in rows:
                if row.wave is not None:
                    row.wave[0].ready(row.wave[1])
            for i, row in enumerate(rows):
                if not row.future.done():
                    failure_receipt = getattr(error, "receipt", None)
                    if failure_receipt is not None:
                        row.future.set_exception(
                            BatchMemberError(
                                {
                                    **failure_receipt,
                                    "usage": allocated_usage(
                                        failure_receipt["usage"], i, len(rows)
                                    ),
                                }
                            )
                        )
                    else:
                        row.future.set_exception(error)

    async def drain(self) -> None:
        while self.tasks:
            await asyncio.gather(*tuple(self.tasks))
