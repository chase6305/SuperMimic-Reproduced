"""Exercise the public assessor's resource options before heavy SDK imports."""

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


class GuardReached(Exception):
    pass


class AssessmentResourceOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "scripts/assess_skill.py"
        spec = importlib.util.spec_from_file_location("assessment_options", script)
        cls.assessor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.assessor)

    def argv(self, budget=None):
        args = [
            "assess_skill.py", "--bundle", "/unused", "--motion", "/unused",
            "--run", "/unused", "--skill", "fight", "--output", "/unused",
        ]
        return args if budget is None else args + [f"--max-rss-gib={budget}"]

    def test_budget_reaches_guard_with_system_reserves_unchanged(self):
        for argument, expected in [(None, 2.0), ("0.25", .25), ("1", 1.), ("1.5", 1.5), ("2", 2.)]:
            with self.subTest(argument=argument), patch.object(sys, "argv", self.argv(argument)):
                # Stop at the real guard boundary: no SDK, model or unused path is accessed.
                with patch("supermimic.cpu_resources.cpu_guard", side_effect=GuardReached) as guard:
                    with self.assertRaises(GuardReached):
                        self.assessor.main()
                guard.assert_called_once()
                limits = guard.call_args.kwargs["limits"]
                self.assertEqual(limits.max_rss_gib, expected)
                self.assertEqual(limits.reserve_ram_gib, 8.)
                self.assertEqual(limits.reserve_disk_gib, 20.)
                self.assertEqual(limits.max_gpu_mib, 2304)
                self.assertEqual(limits.reserve_gpu_mib, 4096)

    def test_invalid_budget_is_rejected_before_resource_work(self):
        for argument in ["nan", "inf", "-inf", "0", "-1", ".249", "2.001"]:
            with self.subTest(argument=argument), patch.object(sys, "argv", self.argv(argument)):
                with patch("supermimic.cpu_resources.cpu_guard") as guard, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.assessor.main()
                self.assertEqual(caught.exception.code, 2)
                guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
