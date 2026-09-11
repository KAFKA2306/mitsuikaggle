import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).parent / "scripts" / "submit_to_kaggle.py"
spec = importlib.util.spec_from_file_location("submit_to_kaggle", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
KaggleSubmissionManager = module.KaggleSubmissionManager


def completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class KaggleSubmissionContractTests(unittest.TestCase):
    def manager(self, **kwargs):
        return KaggleSubmissionManager(
            verification_timeout=kwargs.pop("verification_timeout", 20),
            poll_interval=0,
            **kwargs,
        )

    def test_submit_failure_returns_no_identity(self):
        manager = self.manager()
        with patch.object(module.subprocess, "run", return_value=completed(stderr="upload failed", returncode=1)):
            self.assertIsNone(manager.submit_file())

    def test_submit_requires_stable_submission_ref(self):
        manager = self.manager()
        with patch.object(module.subprocess, "run", return_value=completed(stdout="Successfully submitted file")):
            self.assertIsNone(manager.submit_file())

    def test_current_submission_absent_then_pending_then_complete(self):
        manager = self.manager()
        results = [
            completed(stderr="not found", returncode=1),
            completed(stdout="Submission Ref:  12345\nStatus:          PENDING\n"),
            completed(stdout="Submission Ref:  12345\nStatus:          COMPLETE\nPublic Score:    0.1\n"),
        ]
        with patch.object(module.subprocess, "run", side_effect=results) as run, patch.object(module.time, "sleep"):
            self.assertTrue(manager.verify_submission("12345"))
        for call in run.call_args_list:
            self.assertEqual(call.args[0], ['kaggle', 'competitions', 'submission', '12345'])

    def test_current_submission_terminal_failure_fails(self):
        manager = self.manager()
        with patch.object(
            module.subprocess,
            "run",
            return_value=completed(stdout="Submission Ref:  12345\nStatus:          ERROR\n"),
        ):
            self.assertFalse(manager.verify_submission("12345"))

    def test_verification_timeout_fails_closed(self):
        manager = self.manager(verification_timeout=0)
        with patch.object(module.subprocess, "run", return_value=completed(stderr="not found", returncode=1)) as run:
            self.assertFalse(manager.verify_submission("12345"))
        self.assertEqual(run.call_count, 1)

    def test_unrelated_submission_history_cannot_satisfy_current_verification(self):
        manager = self.manager(verification_timeout=0)
        unrelated_history = "old.csv  COMPLETE  0.99"
        with patch.object(module.subprocess, "run", return_value=completed(stdout=unrelated_history)) as run:
            self.assertFalse(manager.verify_submission("777"))
        self.assertEqual(run.call_args.args[0], ['kaggle', 'competitions', 'submission', '777'])

    def test_run_submission_succeeds_only_after_exact_ref_completion(self):
        manager = self.manager()
        manager.validate_environment = Mock(return_value=True)
        manager.validate_submission_file = Mock(return_value=True)
        manager.check_competition_status = Mock(return_value=True)
        manager.submit_file = Mock(return_value="42")
        manager.verify_submission = Mock(return_value=True)
        self.assertTrue(manager.run_submission())
        manager.verify_submission.assert_called_once_with("42")

    def test_run_submission_fails_when_exact_ref_does_not_complete(self):
        manager = self.manager()
        manager.validate_environment = Mock(return_value=True)
        manager.validate_submission_file = Mock(return_value=True)
        manager.check_competition_status = Mock(return_value=True)
        manager.submit_file = Mock(return_value="42")
        manager.verify_submission = Mock(return_value=False)
        self.assertFalse(manager.run_submission())

    def test_main_exit_code_reflects_exact_verification(self):
        with patch.object(module.KaggleSubmissionManager, "run_submission", return_value=False):
            self.assertEqual(module.main(), 1)
        with patch.object(module.KaggleSubmissionManager, "run_submission", return_value=True):
            self.assertEqual(module.main(), 0)


if __name__ == "__main__":
    unittest.main()
