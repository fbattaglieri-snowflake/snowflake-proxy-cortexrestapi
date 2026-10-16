"""Strict data-only contracts shared by administration and serving."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Protocol = Literal["chat", "responses", "messages"]
Capability = Literal["text", "tools", "parallel_tools", "streaming", "structured_output", "reasoning"]
OPTION_PATHS = {
    "chat": {"max_tokens", "max_completion_tokens", "reasoning_effort", "temperature", "top_p", "parallel_tool_calls", "stream_options.include_usage"},
    "responses": {"max_output_tokens", "reasoning.effort", "store", "include", "temperature", "top_p", "parallel_tool_calls"},
    "messages": {"max_tokens", "thinking.type", "thinking.budget_tokens", "output_config.effort", "temperature", "top_p", "tool_choice.type", "tool_choice.disable_parallel_tool_use"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result.astimezone(timezone.utc)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON value")
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical(result)
    return result


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def document(self):
        return self.model_dump(mode="json")


class Rule(Contract):
    id: Name
    path: str
    operation: Literal["set", "remove"]
    when: Literal["missing", "equals"]
    expected: object = None
    value: object = None
    reason: Annotated[str, Field(min_length=1, max_length=512)]
    effects: Annotated[str, Field(min_length=1, max_length=512)]

    @model_validator(mode="after")
    def safe_values(self):
        for value in (self.expected, self.value):
            if not (value is None or type(value) in (str, bool, int, float) or
                    isinstance(value, list) and all(isinstance(item, str) for item in value)):
                raise ValueError("Rules accept scalar options or string lists only")
            if len(canonical(value)) > 512:
                raise ValueError("Rule value too large")
        if self.operation == "remove" and self.when == "missing":
            raise ValueError("Removing a missing value is meaningless")
        return self


class Profile(Contract):
    id: Name
    model: Name
    protocol: Protocol
    account: Name
    backend: Literal["cortex"] = "cortex"
    artifact: Digest
    revision: Name
    capabilities: list[Capability]
    options: dict[str, list[object]] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)
    created_at: str
    expires_at: str
    retire_at: str | None = None

    @model_validator(mode="after")
    def validate_contract(self):
        if "text" not in self.capabilities or len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Unique capabilities including text required")
        if "parallel_tools" in self.capabilities and "tools" not in self.capabilities:
            raise ValueError("Parallel tools requires tools")
        allowed = OPTION_PATHS[self.protocol]
        if not set(self.options) <= allowed or any(not values for values in self.options.values()):
            raise ValueError("Invalid option constraints")
        if len(self.rules) > 8:
            raise ValueError("At most eight bounded option rules per profile")
        if len(canonical([rule.document() for rule in self.rules])) > 3000:
            raise ValueError("Rule metadata must fit bounded response headers")
        if any(rule.path not in allowed for rule in self.rules):
            raise ValueError("Rule may not modify protected fields")
        if len({rule.path for rule in self.rules}) != len(self.rules) or len({rule.id for rule in self.rules}) != len(self.rules):
            raise ValueError("Duplicate rule path or ID")
        if timestamp(self.created_at) >= timestamp(self.expires_at):
            raise ValueError("Invalid validity interval")
        if self.retire_at is not None:
            timestamp(self.retire_at)
        if len(canonical(self.document())) > 32768:
            raise ValueError("Profile too large")
        return self

    def valid(self, now):
        return (timestamp(self.created_at) <= now < timestamp(self.expires_at) and
                (self.retire_at is None or now < timestamp(self.retire_at)))


class Evidence(Contract):
    kind: Literal["kit-proxy", "reviewed-coverage", "experimental-record"]
    sha256: Digest
    profile_sha256: Digest
    target: str
    expires_at: str
    coverage: list[str]
    synthetic: bool = False


class OptionConstraint(Contract):
    """A reviewed policy domain, not a claim that every value was live-tested."""
    kind: Literal["integer", "number", "enum", "string"]
    minimum: int | float | None = None
    maximum: int | float | None = None
    values: list[object] = Field(default_factory=list)
    max_length: Annotated[int, Field(ge=1, le=256)] | None = None

    @model_validator(mode="after")
    def bounded(self):
        if self.kind in {"integer", "number"}:
            if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
                raise ValueError("Finite ordered option bounds required")
            canonical([self.minimum, self.maximum])
            if self.values or self.max_length is not None:
                raise ValueError("Unexpected numeric constraint fields")
            if self.kind == "integer" and any(type(v) is not int for v in (self.minimum, self.maximum)):
                raise ValueError("Integer bounds required")
        elif self.kind == "enum":
            if not self.values or len(self.values) > 32 or self.minimum is not None or self.maximum is not None or self.max_length is not None:
                raise ValueError("Bounded enum required")
            if any(not (v is None or type(v) in (str, int, float, bool) or
                        isinstance(v, list) and all(isinstance(item, str) for item in v)) for v in self.values):
                raise ValueError("Enum values must be scalar options or string lists")
            if len(canonical(self.values)) > 2048:
                raise ValueError("Enum too large")
        elif self.max_length is None or self.minimum is not None or self.maximum is not None or self.values:
            raise ValueError("String length bound required")
        return self

    def accepts(self, value):
        if self.kind == "enum":
            return any(canonical(value) == canonical(item) for item in self.values)
        if self.kind == "string":
            return isinstance(value, str) and 0 < len(value) <= self.max_length and all(ord(char) >= 32 for char in value)
        valid_type = type(value) is int if self.kind == "integer" else type(value) in (int, float)
        return valid_type and self.minimum <= value <= self.maximum


V2_EXTRA_PATHS = {"chat": set(), "responses": {"prompt_cache_key", "reasoning.summary"}, "messages": set()}
INTEGER_PATHS = {"max_tokens", "max_completion_tokens", "max_output_tokens", "thinking.budget_tokens"}


class ProfileV2(Profile):
    schema_version: Literal[2]
    response_stream: Literal["native", "ordered-items"] = "native"
    normalize_client_options: bool = False
    constraints: dict[str, OptionConstraint] = Field(default_factory=dict)
    requires: dict[str, list[str]] = Field(default_factory=dict)
    conflicts: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_v2(self):
        if self.response_stream != "native" and (self.protocol != "responses" or "streaming" not in self.capabilities):
            raise ValueError("Ordered items require Responses streaming")
        allowed = OPTION_PATHS[self.protocol] | V2_EXTRA_PATHS[self.protocol]
        if not set(self.constraints) <= allowed or set(self.constraints) & set(self.options):
            raise ValueError("Invalid or overlapping v2 option constraints")
        for path, constraint in self.constraints.items():
            expected = ("integer" if path in INTEGER_PATHS else "number" if path in {"temperature", "top_p"}
                        else "string" if path == "prompt_cache_key" else "enum")
            if constraint.kind not in {expected, "enum"} or path == "prompt_cache_key" and constraint.kind != "string":
                raise ValueError("Constraint type does not match protocol field")
            if path in INTEGER_PATHS and (constraint.kind == "enum" and any(type(v) is not int or v <= 0 for v in constraint.values)
                                         or constraint.kind == "integer" and constraint.minimum <= 0):
                raise ValueError("Positive integer token domain required")
            if constraint.kind == "enum":
                for value in constraint.values:
                    if path in {"store", "parallel_tool_calls", "stream_options.include_usage", "tool_choice.disable_parallel_tool_use"}:
                        valid = type(value) is bool
                    elif path == "include":
                        valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
                    elif path in INTEGER_PATHS:
                        valid = type(value) is int
                    elif path in {"temperature", "top_p"}:
                        valid = type(value) in (int, float)
                    else:
                        valid = isinstance(value, str) and bool(value)
                    if not valid:
                        raise ValueError("Enum type does not match protocol field")
        declared = set(self.constraints) | set(self.options)
        if len(self.requires) > 16 or len(self.conflicts) > 16:
            raise ValueError("Too many cross-option conditions")
        for path, dependencies in self.requires.items():
            if path not in declared or not dependencies or len(dependencies) > 8 or not set(dependencies) <= declared or path in dependencies:
                raise ValueError("Invalid option dependencies")
        for pair in self.conflicts:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= declared:
                raise ValueError("Invalid option conflict")
        for rule in self.rules:
            if rule.operation == "set" and rule.path in self.constraints and not self.constraints[rule.path].accepts(rule.value):
                raise ValueError("Default/override outside option domain")
        return self


def parse_profile(document):
    return (ProfileV2 if document.get("schema_version") == 2 else Profile).model_validate(document)


class Proposal(Contract):
    profile: ProfileV2 | Profile
    upstream: str
    expected_revision: Digest
    evidence: list[Evidence]
    synthetic: bool = False


class Approval(Contract):
    key_id: Name
    approver: Name
    proposal_sha256: Digest
    account: Name
    decision: Literal["approve"]
    issued_at: str
    expires_at: str
    synthetic: bool = False
    signature: str


class ExperimentalApproval(Contract):
    method: Literal["administrative-experimental-v1", "administrative-standard-v1"]
    approver: Name
    proposal_sha256: Digest
    account: Name
    decision: Literal["approve-experimental"]
    issued_at: str
    expires_at: str
    reason: Annotated[str, Field(min_length=1, max_length=1024)]
    limitations: Annotated[list[str], Field(min_length=1, max_length=32)]
    synthetic: Literal[False] = False


class RevisionProposal(Proposal):
    schema_version: Literal[2]
    previous_entry: Digest
    change: Literal["compatible", "rollback", "artifact"]


class Entry(Contract):
    proposal: RevisionProposal | Proposal
    approval: Approval | ExperimentalApproval


class Snapshot(Contract):
    schema_version: Literal[1] = 1
    generation: Annotated[int, Field(ge=0)] = 0
    entries: dict[str, Entry] = Field(default_factory=dict)
    revoked: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def identities(self):
        if self.schema_version == 1 and any(isinstance(entry.proposal, RevisionProposal) for entry in self.entries.values()):
            raise ValueError("Revisions require snapshot v2 ancestry")
        if any(key != entry.proposal.profile.id for key, entry in self.entries.items()):
            raise ValueError("Entry identity mismatch")
        if len(set(self.revoked)) != len(self.revoked):
            raise ValueError("Duplicate revocation")
        return self


class RevisionSnapshot(Snapshot):
    schema_version: Literal[2]
    history: dict[str, Entry]
    activations: list[Digest] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_chain(self):
        if len(self.history) > 1000 or len(self.activations) > 1000 or len(set(self.activations)) != len(self.activations):
            raise ValueError("Bounded unique revision chain required")
        active = {}
        revisions = set()
        for key, entry in self.history.items():
            if key != digest(entry.document()):
                raise ValueError("History digest mismatch")
            identity = (entry.proposal.profile.id, entry.proposal.profile.revision)
            if isinstance(entry.proposal, RevisionProposal):
                revisions.add(key)
            else:
                if identity[0] in active:
                    raise ValueError("Duplicate revision root")
                active[identity[0]] = key
        if set(self.activations) != revisions:
            raise ValueError("Unaccounted history activation")
        activated = set(active.values())
        for key in self.activations:
            entry = self.history[key]
            proposal = entry.proposal
            identifier = proposal.profile.id
            if active.get(identifier) != proposal.previous_entry:
                raise ValueError("Broken revision ancestry")
            prior = self.history[proposal.previous_entry]
            validate_revision(prior, entry, self.history, activated=activated)
            active[identifier] = key
            activated.add(key)
        if {identifier: self.history[key] for identifier, key in active.items()} != self.entries:
            raise ValueError("Active pointer does not match revision chain")
        return self


def parse_snapshot(document):
    return (RevisionSnapshot if document.get("schema_version") == 2 else Snapshot).model_validate(document)


def snapshot_history(snapshot):
    if isinstance(snapshot, RevisionSnapshot):
        return snapshot.history
    return {digest(entry.document()): entry for entry in snapshot.entries.values()}


def validate_successor(previous, current):
    if (current.generation < previous.generation or
            current.generation == previous.generation and current != previous or
            not set(previous.revoked) <= set(current.revoked)):
        raise ValueError("Registry regression or forgotten revocation")
    old_history, new_history = snapshot_history(previous), snapshot_history(current)
    if any(new_history.get(key) != value for key, value in old_history.items()):
        raise ValueError("Immutable registry entry changed")
    if isinstance(previous, RevisionSnapshot):
        if not isinstance(current, RevisionSnapshot) or current.activations[:len(previous.activations)] != previous.activations:
            raise ValueError("Activation history regression")


def compatible_profiles(previous, current):
    """Conservative widening only; ambiguous changes require an explicit new ID."""
    for field in ("id", "model", "protocol", "account", "backend", "artifact", "rules", "retire_at"):
        if getattr(previous, field) != getattr(current, field):
            return False
    if not set(previous.capabilities) <= set(current.capabilities):
        return False
    if not getattr(current, "normalize_client_options", False):
        if getattr(previous, "response_stream", "native") != getattr(current, "response_stream", "native"):
            return False
    if timestamp(current.expires_at) < timestamp(previous.expires_at):
        return False
    if isinstance(previous, ProfileV2):
        if not isinstance(current, ProfileV2) or previous.requires != current.requires or previous.conflicts != current.conflicts:
            return False
    elif isinstance(current, ProfileV2) and (current.requires or current.conflicts):
        return False
    for path, values in previous.options.items():
        constraint = current.constraints.get(path) if isinstance(current, ProfileV2) else None
        if any(not (constraint.accepts(value) if constraint else
                    any(canonical(value) == canonical(candidate) for candidate in current.options.get(path, []))) for value in values):
            return False
    if isinstance(previous, ProfileV2):
        for path, old in previous.constraints.items():
            new = current.constraints.get(path)
            if new is None or old.kind != new.kind:
                return False
            if old.kind == "enum" and any(not new.accepts(value) for value in old.values):
                return False
            if old.kind in {"integer", "number"} and (new.minimum > old.minimum or new.maximum < old.maximum):
                return False
            if old.kind == "string" and new.max_length < old.max_length:
                return False
    return True


def validate_revision(prior, entry, history, *, activated=None):
    proposal = entry.proposal
    previous, current = prior.proposal.profile, proposal.profile
    if any(item.proposal.profile.id == current.id and item.proposal.profile.revision == current.revision
           and item.proposal.profile != current for item in history.values()):
        raise ValueError("Profile revision identity reused")
    if proposal.change == "compatible":
        if current.revision == previous.revision or not compatible_profiles(previous, current):
            raise ValueError("Incompatible profile requires a new public ID")
    elif proposal.change == "artifact":
        before, after = previous.document(), current.document()
        for field in ("artifact", "revision", "created_at", "expires_at"):
            before.pop(field, None)
            after.pop(field, None)
        if before != after or current.artifact == previous.artifact or current.revision == previous.revision:
            raise ValueError("Artifact activation must preserve the exact behavioral profile")
    else:
        candidates = (history[key] for key in activated) if activated is not None else history.values()
        if not any(item.proposal.profile == current for item in candidates):
            raise ValueError("Rollback must select a previously activated exact profile")
        if current.id != previous.id or current == previous:
            raise ValueError("Invalid rollback target")


def required_coverage(profile):
    required = {"native-sdk", "history", "proxy-options", "error-propagation"}
    if "tools" in profile.capabilities:
        required |= {"tool-pairing", "tool-error"}
    if "streaming" in profile.capabilities:
        required |= {"incremental-sse", "disconnect", "backpressure", "terminal"}
    if "reasoning" in profile.capabilities:
        required |= {"reasoning-output", "reasoning-replay"}
    if profile.protocol == "responses":
        required |= {"stateless-history", "opaque-reasoning"}
    if profile.rules:
        required |= {"rule:" + rule.id for rule in profile.rules}
    if isinstance(profile, ProfileV2):
        required |= {"constraint:" + path for path in profile.constraints}
        if profile.response_stream == "ordered-items":
            required.add("ordered-items-stream")
        if profile.requires or profile.conflicts:
            required.add("option-dependencies")
    return required