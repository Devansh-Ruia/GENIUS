"""Data contracts shared by tools, agents, the orchestrator, eval and the API.

Every model forbids extra fields and is frozen, so an agent can't add or
change fields on a record it was handed.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

GenomeBuild = Literal["GRCh37", "GRCh38"]
FindlayClass = Literal["FUNC", "INT", "LOF"]
Status = Literal["ok", "unavailable", "error"]
Verdict = Literal["damaging", "tolerated", "uncertain"]
AgentName = Literal["evo2", "sae", "conservation", "population", "single_agent"]
ToolName = Literal["evo2", "sae", "conservation", "population"]
Call = Literal["likely_damaging", "likely_tolerated", "unresolved"]

EVIDENCE_AGENTS: tuple[str, ...] = ("evo2", "sae", "conservation", "population")
ANSWER_KEY_FIELDS: tuple[str, ...] = ("findlay_function_class", "findlay_score")

_BASE_RE = re.compile(r"^[ACGT]$")
_CHROM_RE = re.compile(r"^(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$")
_VARIANT_ID_RE = re.compile(r"^(GRCh37|GRCh38)-([0-9XYMT]+)-([0-9]+)-([ACGT])-([ACGT])$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Variant(_Contract):
    chrom: str
    pos: int = Field(gt=0)
    ref: str
    alt: str
    genome_build: GenomeBuild
    hgvs_c: str | None = None
    findlay_function_class: FindlayClass | None = None
    findlay_score: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("chrom", mode="before")
    @classmethod
    def _normalize_chrom(cls, value: Any) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            raise ValueError("chrom must be a string such as '17' or 'chr17'")
        chrom = value.strip()
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        chrom = chrom.upper()
        if chrom == "M":
            chrom = "MT"
        if not _CHROM_RE.match(chrom):
            raise ValueError(f"unknown chromosome {value!r}")
        return chrom

    @field_validator("ref", "alt")
    @classmethod
    def _single_base(cls, value: str) -> str:
        if not _BASE_RE.match(value):
            raise ValueError("must be a single uppercase base from A, C, G, T")
        return value

    @model_validator(mode="after")
    def _ref_differs_from_alt(self) -> "Variant":
        if self.ref == self.alt:
            raise ValueError("ref and alt must differ")
        return self

    @property
    def variant_id(self) -> str:
        return f"{self.genome_build}-{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    def without_answer_key(self) -> "Variant":
        """Return a copy with the Findlay fields removed, safe to hand to an agent."""
        return Variant(**self.model_dump(exclude={"findlay_function_class", "findlay_score"}))


class AnswerKeyError(ValueError):
    """A Variant carrying the Findlay answer key reached an agent or a tool."""


def reject_answer_key(variant: Variant, receiver: str) -> None:
    """Raise AnswerKeyError if any Findlay field on the variant is set.

    Every agent run() and tool entry point calls this first, so a labeled
    variant fails loudly on any call path, not only through the orchestrator.
    """
    carried = [name for name, value in variant.model_dump(include=set(ANSWER_KEY_FIELDS)).items() if value is not None]
    if carried:
        raise AnswerKeyError(
            f"{receiver} was handed a Variant carrying the Findlay answer key ({', '.join(carried)}). "
            "Call variant.without_answer_key() first."
        )


def parse_variant_id(variant_id: str) -> Variant:
    """Reverse Variant.variant_id. Raises ValueError on a malformed ID."""
    match = _VARIANT_ID_RE.match(variant_id)
    if match is None:
        raise ValueError(f"not a variant ID of the form BUILD-CHROM-POS-REF-ALT: {variant_id!r}")
    build, chrom, pos, ref, alt = match.groups()
    return Variant(chrom=chrom, pos=int(pos), ref=ref, alt=alt, genome_build=build)


class ToolResult(_Contract):
    tool: ToolName
    status: Status
    values: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _values_only_when_ok(self) -> "ToolResult":
        if self.status != "ok" and self.values:
            raise ValueError("values must be empty unless status is 'ok'")
        return self


class AgentFinding(_Contract):
    agent: AgentName
    variant_id: str = Field(min_length=1)
    status: Status
    verdict: Verdict | None = None
    raw_values: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    # Set only by the rebuttal round, when the agent revises its verdict.
    revision_reason: str | None = None

    @model_validator(mode="after")
    def _status_rules(self) -> "AgentFinding":
        if self.status != "ok":
            if self.verdict is not None:
                raise ValueError("verdict must be null unless status is 'ok'")
            if self.raw_values:
                raise ValueError("raw_values must be empty unless status is 'ok'")
        elif self.verdict is None:
            raise ValueError("a finding with status 'ok' needs a verdict ('uncertain' is allowed)")
        return self


def _check_unique_agents(findings: list[AgentFinding], label: str) -> None:
    agents = [f.agent for f in findings]
    if len(agents) != len(set(agents)):
        raise ValueError(f"{label} lists the same agent more than once")


class PanelReport(_Contract):
    variant_id: str = Field(min_length=1)
    call: Call
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    calibrated: bool
    driving_agents: list[AgentName] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    what_would_settle_it: str
    counselor_summary: str
    patient_summary: str
    findings_initial: list[AgentFinding]
    findings_after_rebuttal: list[AgentFinding]
    mock: bool

    @model_validator(mode="after")
    def _report_rules(self) -> "PanelReport":
        if self.confidence is None and self.calibrated:
            raise ValueError("calibrated must be false when confidence is null")
        if self.call == "unresolved":
            if self.confidence is not None or self.calibrated:
                raise ValueError("an unresolved report has confidence null and calibrated false")
        elif self.confidence is None:
            raise ValueError("a call other than 'unresolved' needs a confidence")

        for finding in [*self.findings_initial, *self.findings_after_rebuttal]:
            if finding.variant_id != self.variant_id:
                raise ValueError(
                    f"finding from {finding.agent} is for {finding.variant_id}, not {self.variant_id}"
                )

        _check_unique_agents(self.findings_initial, "findings_initial")
        _check_unique_agents(self.findings_after_rebuttal, "findings_after_rebuttal")
        initial_by_agent = {f.agent: f for f in self.findings_initial}
        after_by_agent = {f.agent: f for f in self.findings_after_rebuttal}
        if set(initial_by_agent) != set(after_by_agent):
            raise ValueError("findings_initial and findings_after_rebuttal must cover the same agents")

        for finding in self.findings_initial:
            if finding.revision_reason is not None:
                raise ValueError(f"initial finding from {finding.agent} can't carry a revision_reason")
        for agent, after in after_by_agent.items():
            before = initial_by_agent[agent]
            if before.status != "ok" and after.status == "ok":
                raise ValueError(f"{agent} had no evidence before rebuttal, so it can't gain a verdict in rebuttal")
            if before.status == "ok" and after.status == "ok":
                if after.raw_values != before.raw_values or after.sources != before.sources:
                    raise ValueError(f"{agent} changed raw_values or sources in rebuttal, which has no tool access")
            if before.status == "ok" and after.status != "ok":
                if not (after.revision_reason and after.revision_reason.strip()):
                    raise ValueError(f"{agent} withdrew an ok finding in rebuttal without a revision_reason")
            revised = before.status == "ok" and after.status == "ok" and before.verdict != after.verdict
            if revised and not (after.revision_reason and after.revision_reason.strip()):
                raise ValueError(f"{agent} changed its verdict in rebuttal without a revision_reason")

        if len(self.driving_agents) != len(set(self.driving_agents)):
            raise ValueError("driving_agents lists the same agent more than once")
        ok_after = {f.agent for f in self.findings_after_rebuttal if f.status == "ok"}
        not_ok = [a for a in self.driving_agents if a not in ok_after]
        if not_ok:
            raise ValueError(f"driving_agents must have status 'ok' after rebuttal: {not_ok}")
        if self.call != "unresolved" and not self.driving_agents:
            raise ValueError("a call other than 'unresolved' needs at least one driving agent")
        return self


class RunRecord(_Contract):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    config: dict[str, Any]
    seed: int
    started_at: AwareDatetime
    variant_ids: list[str]
    outputs_path: str = Field(min_length=1)
