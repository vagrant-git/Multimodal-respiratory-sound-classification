import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional


class DropLabel(Exception):
    """Raised when a label is configured to be dropped."""


# Canonical -> normalized class token (string). Use "drop" to remove samples.
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "no secretion": "no_secretion",
    "no secretion sound": "no_secretion",
    "3ml secretion": "secretion_3ml",
    "3ml secretion m4": "secretion_3ml_m4",
    "5ml secretion m4": "secretion_5ml_m4",
    "3ml secretion (no hemf)": "drop",  # example: drop no-HEMF samples
}


def _canonicalize(raw: str, drop_no_hemf: bool = False) -> str:
    s = str(raw).strip()
    if drop_no_hemf:
        s = re.sub(r"\s*\(no\s*hemf\)\s*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


@dataclass
class LabelProcessor:
    """
    Rule-based label processor with an explicit mapping table.

    - Provide a raw->class mapping (string tokens such as "no_secretion", "m4", "3ml_m4"...).
    - Use value "drop" (configurable) to discard samples.
    - Unknown labels fail fast (KeyError) unless fail_on_unknown=False.
    """

    raw_to_norm: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_MAP))
    drop_token: str = "drop"
    drop_no_hemf: bool = False  # default to manual matching; set True if you still want to strip suffix
    fail_on_unknown: bool = True

    def __call__(self, raw: Optional[str]) -> str:
        if raw is None:
            raise ValueError("LabelProcessor: raw label is None")
        canon = _canonicalize(raw, drop_no_hemf=self.drop_no_hemf)
        if canon in self.raw_to_norm:
            norm = self.raw_to_norm[canon]
            if norm == self.drop_token:
                raise DropLabel(canon)
            return norm
        if self.fail_on_unknown:
            known = ", ".join(sorted(self.raw_to_norm.keys()))
            raise KeyError(
                f"Unrecognized label '{raw}' (canonical='{canon}'). "
                f"Known labels: {known}"
            )
        return canon

    def describe(self) -> str:
        """
        Return a human-readable table of the mapping.
        """
        lines = ["Label mapping (canonical -> class_token):"]
        for k in sorted(self.raw_to_norm.keys()):
            v = self.raw_to_norm[k]
            lines.append(f"  '{k}' -> '{v}'")
        return "\n".join(lines)

    def summarize_counts(self, raw_labels) -> str:
        """
        Given an iterable of raw labels, return a summary count per normalized class.
        Drop-marked samples are counted separately.
        """
        counter = Counter()
        dropped = 0
        for r in raw_labels:
            try:
                norm = self(r)
                counter[norm] += 1
            except DropLabel:
                dropped += 1
        lines = ["Label counts (normalized):"]
        for cls, cnt in counter.most_common():
            lines.append(f"  {cls}: {cnt}")
        lines.append(f"  dropped: {dropped}")
        return "\n".join(lines)
