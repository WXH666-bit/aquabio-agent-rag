from .models import AquaBioState


def _append_trace(state: AquaBioState, value: str) -> list[str]:
    return [*state.get("trace", []), value]


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            result.append(row)
    return result


