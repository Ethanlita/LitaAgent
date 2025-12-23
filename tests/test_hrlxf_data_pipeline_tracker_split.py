import json
import tempfile
import unittest
from pathlib import Path

from litaagent_std.hrl_xf import data_pipeline


def _write_min_tracker_file(path: Path, *, world_id: str, days: list[int]) -> None:
    data = {
        "agent_id": "00Pe@0",
        "agent_type": "PenguinAgent",
        "world_id": world_id,
        "entries": [
            {
                "category": "state",
                "event": "daily_status",
                "day": int(d),
                "data": {"n_steps": 10, "n_lines": 1},
            }
            for d in days
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestHRLXFTrackerLogsSplit(unittest.TestCase):
    def test_tracker_logs_directory_is_split_by_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracker_dir = root / "tracker_logs"
            tracker_dir.mkdir(parents=True, exist_ok=True)

            # 两个不同 world/run，但 agent_id 相同（与 hrl_data_runner 输出一致）
            _write_min_tracker_file(
                tracker_dir / "agent_00Pe_at_0_world_A.00__aaa.json",
                world_id="worldA.00",
                days=[0, 1],
            )
            _write_min_tracker_file(
                tracker_dir / "agent_00Pe_at_0_world_A.01__bbb.json",
                world_id="worldA.01",
                days=[0, 1],
            )

            macro, micro = data_pipeline.load_tournament_data(
                str(root),
                agent_name="Pe",
                strict_json_only=True,
                num_workers=1,
                goal_backfill="none",
            )

            self.assertEqual(len(micro), 0)
            self.assertEqual(len(macro), 4)  # 2 files × 2 days


if __name__ == "__main__":
    unittest.main()
