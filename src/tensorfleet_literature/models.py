"""Data model and validation rules for the R-01 literature matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

_REQUIRED_FIELDS = (
    "citation",
    "title",
    "year",
    "hardware",
    "model",
    "quantization",
    "adapter_placement",
    "adapter_rank",
    "training_budget",
    "seeds",
    "memory_methodology",
    "evaluation_metrics",
    "released_artifacts",
    "findings",
    "limitations",
)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One comparable paper or implementation in the related-work matrix."""

    citation: str
    title: str
    year: int
    url: str
    hardware: str
    model: str
    quantization: str
    adapter_placement: str
    adapter_rank: str
    training_budget: str
    seeds: str
    memory_methodology: str
    evaluation_metrics: str
    released_artifacts: str
    findings: str
    limitations: str
    study_type: str = "empirical study"
    notes: str = ""

    def validate(self) -> list[str]:
        """Return actionable validation errors; an empty list means valid."""
        errors: list[str] = []
        for name in _REQUIRED_FIELDS:
            value = getattr(self, name)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"{name} is required")
        if not isinstance(self.year, int) or not 1900 <= self.year <= 2100:
            errors.append("year must be an integer between 1900 and 2100")
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("url must be an absolute http(s) URL")
        return errors


@dataclass(slots=True)
class LiteratureMatrix:
    """Collection-level checks required before R-01 can be marked complete."""

    records: list[EvidenceRecord] = field(default_factory=list)

    def validate(self, *, minimum: int = 12, maximum: int = 15) -> list[str]:
        errors: list[str] = []
        if not minimum <= len(self.records) <= maximum:
            errors.append(
                f"matrix must contain {minimum}-{maximum} records; found {len(self.records)}"
            )

        seen_citations: set[str] = set()
        for index, record in enumerate(self.records, start=1):
            for error in record.validate():
                errors.append(f"record {index}: {error}")
            key = record.citation.casefold().strip()
            if key in seen_citations:
                errors.append(f"record {index}: duplicate citation {record.citation!r}")
            seen_citations.add(key)
        return errors

    def require_valid(self, *, minimum: int = 12, maximum: int = 15) -> None:
        errors = self.validate(minimum=minimum, maximum=maximum)
        if errors:
            raise ValueError("literature matrix is invalid:\n- " + "\n- ".join(errors))

    @classmethod
    def from_records(cls, records: Iterable[EvidenceRecord]) -> "LiteratureMatrix":
        return cls(list(records))
