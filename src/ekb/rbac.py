"""Document-level role-based access control, enforced as a metadata filter.

Every chunk carries an ``acl_tag``:
  * ``public``                              — any employee
  * ``<business_unit>:<access_level>``      — e.g. ``compliance:confidential``

A user's entitlements (business units + clearance level) expand into the list of tags
they may read. That list is pushed down into the Vector Search query as a filter, so
unauthorized chunks are never retrieved — they cannot leak into the prompt, the
citations or the logs. A second, post-retrieval check (``is_authorized``) is kept as
defence in depth.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ["public", "internal", "confidential", "restricted"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


def acl_tag(business_unit: str, access_level: str) -> str:
    if access_level not in LEVEL_RANK:
        raise ValueError(f"unknown access level {access_level!r}")
    return "public" if access_level == "public" else f"{business_unit}:{access_level}"


@dataclass(frozen=True)
class UserContext:
    user_id: str
    business_units: tuple[str, ...] = ()
    clearance: str = "internal"
    roles: tuple[str, ...] = field(default_factory=tuple)

    def allowed_tags(self) -> list[str]:
        max_rank = LEVEL_RANK[self.clearance]
        tags = ["public"]
        for bu in self.business_units:
            for lvl in LEVELS[1 : max_rank + 1]:
                tags.append(f"{bu}:{lvl}")
        return tags

    def vector_search_filter(self) -> dict:
        """Filter dict for ``VectorSearchIndex.similarity_search(filters=...)``."""
        return {"acl_tag": self.allowed_tags()}

    def sql_predicate(self) -> str:
        quoted = ", ".join(f"'{t}'" for t in self.allowed_tags())
        return f"acl_tag IN ({quoted})"

    def is_authorized(self, chunk: dict) -> bool:
        return chunk.get("acl_tag") in set(self.allowed_tags())


def load_entitlements(path: str | Path) -> dict[str, UserContext]:
    """Read ``user_id,business_units,clearance,roles`` (``;``-separated lists)."""
    users = {}
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            users[row["user_id"]] = UserContext(
                user_id=row["user_id"],
                business_units=tuple(b for b in row["business_units"].split(";") if b),
                clearance=row["clearance"],
                roles=tuple(r for r in row.get("roles", "").split(";") if r),
            )
    return users
