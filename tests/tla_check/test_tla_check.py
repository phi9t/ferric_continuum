import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tools import tla_check


class TlaCheckTests(unittest.TestCase):
    def test_summarize_tlc_output_classifies_common_results(self):
        success = "Model checking completed. No error has been found.\n"
        invariant = "Error: Invariant Inv is violated.\n"
        temporal = "Error: Temporal properties were violated.\n"

        self.assertEqual(tla_check.summarize_tlc_output(success), "success")
        self.assertEqual(tla_check.summarize_tlc_output(invariant), "invariant_violation")
        self.assertEqual(tla_check.summarize_tlc_output(temporal), "temporal_violation")
        self.assertEqual(tla_check.summarize_tlc_output("Semantic errors:\n"), "checker_error")

    def test_discover_checker_prefers_explicit_jar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jar = root / "tla2tools.jar"
            jar.write_text("fake jar\n")

            checker = tla_check.discover_checker(jar_arg=str(jar), environ={}, path=[])

            self.assertEqual(checker.kind, "tlc")
            self.assertEqual(checker.jar, jar.resolve())
            self.assertEqual(
                checker.command_prefix,
                ["java", "-XX:+UseParallelGC", "-cp", str(jar.resolve()), "tlc2.TLC"],
            )

    def test_discover_checker_uses_explicit_java_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jar = root / "tla2tools.jar"
            jar.write_text("fake jar\n")

            checker = tla_check.discover_checker(
                jar_arg=str(jar), java_arg="/hermetic/jdk/bin/java", environ={}, path=[]
            )

            self.assertEqual(checker.kind, "tlc")
            self.assertEqual(checker.command_prefix[0], "/hermetic/jdk/bin/java")
            self.assertEqual(checker.command_prefix[-1], "tlc2.TLC")

    def test_discover_checker_reports_missing_tool(self):
        checker = tla_check.discover_checker(jar_arg=None, environ={}, path=[])

        self.assertEqual(checker.kind, "missing")
        self.assertIn("TLA_TOOLS_JAR", checker.message)

    def test_dry_run_writes_evidence_without_executing_checker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "MeshPlan.tla"
            config = root / "GoodSchedule.cfg"
            out_dir = root / "evidence"
            jar = root / "tla2tools.jar"
            model.write_text("---- MODULE MeshPlan ----\n====\n")
            config.write_text("SPECIFICATION Spec\n")
            jar.write_text("fake jar\n")

            exit_code = tla_check.main([
                "--model",
                str(model),
                "--config",
                str(config),
                "--tla-tools-jar",
                str(jar),
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ])

            self.assertEqual(exit_code, 0)
            evidence = json.loads((out_dir / "tla-check.json").read_text())
            self.assertEqual(evidence["checker"], "tlc")
            self.assertEqual(evidence["executed"], False)
            self.assertTrue(evidence["ok"])
            self.assertEqual(evidence["model"], str(model.resolve()))
            self.assertEqual(evidence["config"], str(config.resolve()))
            self.assertIn("-metadir", evidence["argv"])
            self.assertIn(str(out_dir.resolve() / "states"), evidence["argv"])
            self.assertEqual(evidence["argv"][-2:], ["-config", str(config.resolve())])

    def test_missing_checker_writes_unavailable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "MeshPlan.tla"
            config = root / "GoodSchedule.cfg"
            out_dir = root / "evidence"
            model.write_text("---- MODULE MeshPlan ----\n====\n")
            config.write_text("SPECIFICATION Spec\n")

            with mock.patch.dict(os.environ, {}, clear=True):
                exit_code = tla_check.main([
                    "--model",
                    str(model),
                    "--config",
                    str(config),
                    "--out-dir",
                    str(out_dir),
                ])

            self.assertEqual(exit_code, 2)
            evidence = json.loads((out_dir / "tla-check.json").read_text())
            self.assertEqual(evidence["checker"], "missing")
            self.assertFalse(evidence["ok"])
            self.assertFalse(evidence["executed"])

    def test_dry_run_without_checker_records_unavailable_tooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "MeshPlan.tla"
            config = root / "GoodSchedule.cfg"
            out_dir = root / "evidence"
            model.write_text("---- MODULE MeshPlan ----\n====\n")
            config.write_text("SPECIFICATION Spec\n")

            with mock.patch.dict(os.environ, {}, clear=True):
                exit_code = tla_check.main([
                    "--model",
                    str(model),
                    "--config",
                    str(config),
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ])

            self.assertEqual(exit_code, 0)
            evidence = json.loads((out_dir / "tla-check.json").read_text())
            self.assertEqual(evidence["checker"], "missing")
            self.assertFalse(evidence["executed"])
            self.assertTrue(evidence["ok"])
            self.assertIn("pass --tla-tools-jar", evidence["message"])

    def test_counterexample_path_is_recorded_for_failed_executed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checker = tla_check.Checker(kind="tlc", command_prefix=["python3", "-c", "print('Error: Invariant Inv is violated.')"])
            model = root / "Model.tla"
            config = root / "Model.cfg"
            out_dir = root / "evidence"
            model.write_text("---- MODULE Model ----\n====\n")
            config.write_text("SPECIFICATION Spec\n")

            exit_code = tla_check.run_checker(checker, model.resolve(), config.resolve(), out_dir.resolve())

            self.assertEqual(exit_code, 0)
            evidence = json.loads((out_dir / "tla-check.json").read_text())
            self.assertFalse(evidence["ok"])
            self.assertEqual(evidence["result"], "invariant_violation")
            self.assertEqual(evidence["counterexample"], str(out_dir.resolve() / "stdout.log"))

    def test_mesh_plan_fixtures_are_present(self):
        root = Path(__file__).resolve().parents[2]
        model = root / "formal" / "distributed_training" / "MeshPlan.tla"
        good_model = root / "formal" / "distributed_training" / "MeshPlanGood.tla"
        bad_model = root / "formal" / "distributed_training" / "MeshPlanBad.tla"
        good = root / "formal" / "distributed_training" / "MeshPlanGood.cfg"
        bad = root / "formal" / "distributed_training" / "MeshPlanBad.cfg"
        readme = root / "formal" / "distributed_training" / "README.org"

        for path in [model, good_model, bad_model, good, bad, readme]:
            self.assertTrue(path.is_file(), path)

    @unittest.skipUnless(os.environ.get("TLA_TOOLS_JAR"), "TLA_TOOLS_JAR is required for TLC fixture smoke tests")
    def test_mesh_plan_good_passes_and_bad_finds_blocked_cycle(self):
        root = Path(__file__).resolve().parents[2]
        jar = Path(os.environ["TLA_TOOLS_JAR"])
        checker = tla_check.discover_checker(jar_arg=str(jar))

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            good_exit = tla_check.run_checker(
                checker,
                root / "formal" / "distributed_training" / "MeshPlanGood.tla",
                root / "formal" / "distributed_training" / "MeshPlanGood.cfg",
                out_root / "good",
            )
            bad_exit = tla_check.run_checker(
                checker,
                root / "formal" / "distributed_training" / "MeshPlanBad.tla",
                root / "formal" / "distributed_training" / "MeshPlanBad.cfg",
                out_root / "bad",
            )

            good = json.loads((out_root / "good" / "tla-check.json").read_text())
            bad = json.loads((out_root / "bad" / "tla-check.json").read_text())

            self.assertEqual(good_exit, 0)
            self.assertTrue(good["ok"])
            self.assertEqual(good["result"], "success")
            self.assertEqual(bad_exit, 12)
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["result"], "invariant_violation")
            self.assertIn("counterexample", bad)


if __name__ == "__main__":
    unittest.main()
