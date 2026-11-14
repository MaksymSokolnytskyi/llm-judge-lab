"""SQLite cache of winning routes, with an in-memory mirror.

Lookup order used by the router:

1. exact context match,
2. fallback: best-scoring route for the same node and persona (outcome and style may
   differ),
3. nothing -- the router serves a hard-coded baseline and queues a micro-arena.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from judgelab.dialogues.routing.models import ArenaContext, RegistryEntry, WinningRoute


class ArenaRegistry:
    def __init__(self, db_path: Path | str = ":memory:"):
        # One connection: ``:memory:`` databases vanish when their connection closes.
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registry (
                key TEXT PRIMARY KEY,
                node_name TEXT NOT NULL,
                persona TEXT NOT NULL,
                judge_score REAL NOT NULL,
                run_count INTEGER NOT NULL DEFAULT 1,
                entry_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._cache: dict[str, RegistryEntry] = {
            key: RegistryEntry.model_validate_json(entry_json)
            for key, entry_json in self._conn.execute(
                "SELECT key, entry_json FROM registry"
            )
        }

    def close(self) -> None:
        self._conn.close()

    def get(self, ctx: ArenaContext) -> RegistryEntry | None:
        """Exact match, served from memory."""
        return self._cache.get(ctx.key)

    def get_fallback(self, ctx: ArenaContext) -> RegistryEntry | None:
        """Best route for the same node and persona, ignoring outcome and style."""
        row = self._conn.execute(
            """SELECT entry_json FROM registry WHERE node_name = ? AND persona = ?
               ORDER BY judge_score DESC LIMIT 1""",
            (ctx.node_name, ctx.persona),
        ).fetchone()
        return RegistryEntry.model_validate_json(row[0]) if row else None

    def save(self, ctx: ArenaContext, route: WinningRoute) -> RegistryEntry:
        """Store ``route`` unless a strictly better one is already cached.

        Re-exploration may find a worse route than the incumbent; keeping the best
        makes exploration safe. ``run_count`` counts every arena run for the context.
        """
        incumbent = self._cache.get(ctx.key)
        if incumbent is not None and incumbent.route.judge_score > route.judge_score:
            entry = incumbent
        else:
            entry = RegistryEntry(context=ctx, route=route)
        self._conn.execute(
            """
            INSERT INTO registry
                (key, node_name, persona, judge_score, run_count, entry_json)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(key) DO UPDATE SET
                judge_score = excluded.judge_score,
                entry_json = excluded.entry_json,
                run_count = run_count + 1
            """,
            (
                ctx.key,
                ctx.node_name,
                ctx.persona,
                entry.route.judge_score,
                entry.model_dump_json(),
            ),
        )
        self._conn.commit()
        self._cache[ctx.key] = entry
        return entry

    def stats(self) -> dict[str, float]:
        total, avg = self._conn.execute(
            "SELECT COUNT(*), AVG(judge_score) FROM registry"
        ).fetchone()
        return {"cached_routes": total, "avg_score": round(avg or 0.0, 2)}
