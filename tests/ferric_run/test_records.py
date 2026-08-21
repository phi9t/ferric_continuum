import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from tools.ferric_run.records import create_run_dir, new_run_id, write_json, write_text


class RecordsTests(unittest.TestCase):
    def test_new_run_id_is_utc_sortable(self):
        run_id = new_run_id(datetime(2026, 8, 21, 1, 2, 3, 456789, tzinfo=UTC))
        self.assertEqual(run_id, "20260821T010203456789Z")

    def test_create_run_dir_and_write_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_dir = create_run_dir(repo, "run-1")

            write_json(run_dir / "result.json", {"ok": True})
            write_text(run_dir / "stdout.log", "hello\n")

            self.assertEqual(run_dir, repo / ".ferric" / "runs" / "run-1")
            self.assertEqual(json.loads((run_dir / "result.json").read_text()), {"ok": True})
            self.assertEqual((run_dir / "stdout.log").read_text(), "hello\n")


if __name__ == "__main__":
    unittest.main()
