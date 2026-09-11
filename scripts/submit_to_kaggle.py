#!/usr/bin/env python3
"""Mitsui Commodity Prediction Challenge - fail-closed Kaggle submission workflow."""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('kaggle_submission.log'), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SUBMISSION_REF_RE = re.compile(r"Submission ref:\s*(\d+)", re.IGNORECASE)
SUBMISSION_STATUS_RE = re.compile(r"^Status:\s*([A-Za-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)
PENDING_STATUSES = {"PENDING", "QUEUED", "RUNNING", "PROCESSING"}
FAILURE_STATUSES = {"ERROR", "FAILED", "FAILURE", "CANCELLED", "CANCELED"}
SUCCESS_STATUSES = {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}


class KaggleSubmissionManager:
    def __init__(
        self,
        competition_name="mitsui-commodity-prediction-challenge",
        submission_file="submission_final_424.csv",
        kaggle_config_dir=".env",
        verification_timeout=600,
        poll_interval=5,
    ):
        self.competition = competition_name
        self.submission_file = submission_file
        self.config_dir = kaggle_config_dir
        self.submission_message = "Production Neural Network - 1.1912 Sharpe Score"
        self.verification_timeout = verification_timeout
        self.poll_interval = poll_interval

    def _kaggle_env(self):
        env = os.environ.copy()
        env['KAGGLE_CONFIG_DIR'] = self.config_dir
        return env

    def validate_environment(self):
        """Validate Kaggle API setup and credentials."""
        logger.info("Validating environment...")
        try:
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True, check=True)
            logger.info("Kaggle CLI version: %s", result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Kaggle CLI not installed. Install with: pip install kaggle")
            return False

        kaggle_json = Path(self.config_dir) / "kaggle.json"
        if not kaggle_json.exists():
            logger.error("Kaggle credentials not found at %s", kaggle_json)
            return False
        try:
            with open(kaggle_json) as f:
                creds = json.load(f)
            if 'username' not in creds or 'key' not in creds:
                logger.error("Invalid kaggle.json format")
                return False
            logger.info("Kaggle credentials found for user: %s", creds['username'])
        except Exception as exc:
            logger.error("Error reading kaggle.json: %s", exc)
            return False
        if kaggle_json.stat().st_mode & 0o077:
            logger.warning("Fixing kaggle.json permissions...")
            kaggle_json.chmod(0o600)
        return True

    def validate_submission_file(self):
        """Comprehensive validation of submission file."""
        logger.info("Validating submission file...")
        if not Path(self.submission_file).exists():
            logger.error("Submission file not found: %s", self.submission_file)
            return False
        try:
            import numpy as np
            import pandas as pd

            df = pd.read_csv(self.submission_file)
            logger.info("File shape: %s", df.shape)
            if df.shape != (90, 425):
                logger.error("Wrong dimensions. Expected (90, 425), got %s", df.shape)
                return False
            expected_cols = ['date_id'] + [f'target_{i}' for i in range(424)]
            if df.columns.tolist() != expected_cols:
                logger.error("Column names don't match expected format")
                return False
            expected_dates = list(range(1827, 1917))
            if df['date_id'].tolist() != expected_dates:
                logger.error("Date IDs don't match expected sequence")
                return False
            numeric_data = df.iloc[:, 1:]
            nan_count = numeric_data.isna().sum().sum()
            inf_count = np.isinf(numeric_data).sum().sum()
            if nan_count > 0:
                logger.error("Found %s NaN values", nan_count)
                return False
            if inf_count > 0:
                logger.error("Found %s infinite values", inf_count)
                return False
            logger.info("Value range: %.4f to %.4f", numeric_data.min().min(), numeric_data.max().max())
            logger.info("Total predictions: %s", numeric_data.size)
            logger.info("Submission file validation passed")
            return True
        except Exception as exc:
            logger.error("Error validating submission file: %s", exc)
            return False

    def check_competition_status(self):
        """Check if competition is accessible before submitting."""
        logger.info("Checking competition status...")
        try:
            result = subprocess.run(
                ['kaggle', 'competitions', 'list', '-s', 'mitsui'],
                capture_output=True,
                text=True,
                env=self._kaggle_env(),
                check=True,
            )
            if self.competition in result.stdout:
                logger.info("Competition found and accessible")
                return True
            logger.warning("Competition status unclear")
            return True
        except Exception as exc:
            logger.warning("Could not verify competition status: %s", exc)
            return True

    @staticmethod
    def _submission_ref(output):
        match = SUBMISSION_REF_RE.search(output or "")
        return match.group(1) if match else None

    @staticmethod
    def _submission_status(output):
        match = SUBMISSION_STATUS_RE.search(output or "")
        return match.group(1).upper() if match else None

    def submit_file(self):
        """Submit once and return the exact Kaggle submission ref."""
        logger.info("Starting submission process...")
        cmd = [
            'kaggle', 'competitions', 'submit', self.competition,
            '-f', self.submission_file, '-m', self.submission_message,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._kaggle_env(),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            logger.error("Submission command timed out")
            return None
        except Exception as exc:
            logger.error("Submission command failed: %s", exc)
            return None
        if result.returncode != 0:
            logger.error("Submission failed: %s", result.stderr.strip())
            return None
        submission_ref = self._submission_ref(result.stdout)
        if submission_ref is None:
            logger.error("Submission upload returned success but no submission ref was reported")
            return None
        logger.info("Submission created with ref %s", submission_ref)
        return submission_ref

    def verify_submission(self, submission_ref):
        """Poll the exact submission ref until a terminal status or timeout."""
        logger.info("Verifying submission ref %s...", submission_ref)
        deadline = time.monotonic() + self.verification_timeout
        while True:
            result = subprocess.run(
                ['kaggle', 'competitions', 'submission', str(submission_ref)],
                capture_output=True,
                text=True,
                env=self._kaggle_env(),
            )
            status = self._submission_status(result.stdout) if result.returncode == 0 else None
            if status in SUCCESS_STATUSES:
                logger.info("Submission ref %s completed with status %s", submission_ref, status)
                return True
            if status in FAILURE_STATUSES:
                logger.error("Submission ref %s failed with status %s", submission_ref, status)
                return False
            if time.monotonic() >= deadline:
                logger.error(
                    "Timed out waiting for submission ref %s; last status=%s",
                    submission_ref,
                    status or "NOT_VISIBLE",
                )
                return False
            if status is None:
                logger.info("Submission ref %s is not visible yet; retrying", submission_ref)
            elif status in PENDING_STATUSES:
                logger.info("Submission ref %s is %s; retrying", submission_ref, status)
            else:
                logger.info("Submission ref %s has non-terminal status %s; retrying", submission_ref, status)
            time.sleep(self.poll_interval)

    def run_submission(self):
        """Run workflow and fail closed until exact completion is verified."""
        logger.info("Starting Kaggle submission workflow...")
        logger.info("=" * 60)
        if not self.validate_environment():
            logger.error("Environment validation failed")
            return False
        if not self.validate_submission_file():
            logger.error("Submission file validation failed")
            return False
        if not self.check_competition_status():
            logger.error("Competition status check failed")
            return False
        submission_ref = self.submit_file()
        if submission_ref is None:
            logger.error("Submission failed")
            return False
        if not self.verify_submission(submission_ref):
            logger.error("Exact submission verification failed")
            return False
        logger.info("SUBMISSION WORKFLOW COMPLETED")
        logger.info("=" * 60)
        return True


def main():
    """Main entry point."""
    print("Mitsui Commodity Prediction Challenge - Kaggle Submission")
    print("=" * 60)
    manager = KaggleSubmissionManager()
    success = manager.run_submission()
    if success:
        print("\nSUCCESS! Your model has been submitted to Kaggle.")
        print(f"https://www.kaggle.com/competitions/{manager.competition}/submissions")
    else:
        print("\nFAILED! Check the logs above for detailed error information.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
