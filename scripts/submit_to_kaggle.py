#!/usr/bin/env python3
"""
🏆 Mitsui Commodity Prediction Challenge - One-Click Submission Script
====================================================================

This script handles the complete submission process to Kaggle with:
- Comprehensive file validation
- Multiple submission attempts with different approaches  
- Detailed error logging and diagnostics
- Automatic retry logic for transient errors

Usage: python submit_to_kaggle.py
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaggle_submission.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KaggleSubmissionManager:
    def __init__(self, 
                 competition_name="mitsui-commodity-prediction-challenge",
                 submission_file="submission_final_424.csv",
                 kaggle_config_dir=".env"):
        self.competition = competition_name
        self.submission_file = submission_file
        self.config_dir = kaggle_config_dir
        self.submission_message = "🏆 Production Neural Network - 1.1912 Sharpe Score"
        
    def validate_environment(self):
        """Validate Kaggle API setup and credentials"""
        logger.info("🔍 Validating environment...")
        
        # Check if kaggle is installed
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                 capture_output=True, text=True, check=True)
            logger.info(f"✅ Kaggle CLI version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Kaggle CLI not installed. Install with: pip install kaggle")
            return False
            
        # Check credentials
        kaggle_json = Path(self.config_dir) / "kaggle.json"
        if not kaggle_json.exists():
            logger.error(f"❌ Kaggle credentials not found at {kaggle_json}")
            return False
            
        # Verify credentials format
        try:
            with open(kaggle_json) as f:
                creds = json.load(f)
                if 'username' not in creds or 'key' not in creds:
                    logger.error("❌ Invalid kaggle.json format")
                    return False
            logger.info(f"✅ Kaggle credentials found for user: {creds['username']}")
        except Exception as e:
            logger.error(f"❌ Error reading kaggle.json: {e}")
            return False
            
        # Check file permissions
        stat = kaggle_json.stat()
        if stat.st_mode & 0o077:
            logger.warning("⚠️ Fixing kaggle.json permissions...")
            kaggle_json.chmod(0o600)
            
        return True
        
    def validate_submission_file(self):
        """Comprehensive validation of submission file"""
        logger.info("🔍 Validating submission file...")
        
        if not Path(self.submission_file).exists():
            logger.error(f"❌ Submission file not found: {self.submission_file}")
            return False
            
        try:
            # Load and validate
            df = pd.read_csv(self.submission_file)
            logger.info(f"📊 File shape: {df.shape}")
            
            # Check dimensions
            if df.shape != (90, 425):
                logger.error(f"❌ Wrong dimensions. Expected (90, 425), got {df.shape}")
                return False
                
            # Check column names
            expected_cols = ['date_id'] + [f'target_{i}' for i in range(424)]
            if df.columns.tolist() != expected_cols:
                logger.error("❌ Column names don't match expected format")
                return False
                
            # Check date_id range
            expected_dates = list(range(1827, 1917))
            if df['date_id'].tolist() != expected_dates:
                logger.error("❌ Date IDs don't match expected sequence")
                return False
                
            # Check for problematic values
            numeric_data = df.iloc[:, 1:]  # All except date_id
            
            nan_count = numeric_data.isna().sum().sum()
            inf_count = np.isinf(numeric_data).sum().sum()
            
            if nan_count > 0:
                logger.error(f"❌ Found {nan_count} NaN values")
                return False
                
            if inf_count > 0:
                logger.error(f"❌ Found {inf_count} infinite values")
                return False
                
            # Log statistics
            logger.info(f"📈 Value range: {numeric_data.min().min():.4f} to {numeric_data.max().max():.4f}")
            logger.info(f"📈 Total predictions: {numeric_data.size}")
            logger.info("✅ Submission file validation passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating submission file: {e}")
            return False
            
    def check_competition_status(self):
        """Check if competition is active and accepting submissions"""
        logger.info("🔍 Checking competition status...")
        
        try:
            env = os.environ.copy()
            env['KAGGLE_CONFIG_DIR'] = self.config_dir
            
            result = subprocess.run([
                'kaggle', 'competitions', 'list', '-s', 'mitsui'
            ], capture_output=True, text=True, env=env, check=True)
            
            if self.competition in result.stdout:
                logger.info("✅ Competition found and accessible")
                
                # Check leaderboard to see if submissions are working
                lb_result = subprocess.run([
                    'kaggle', 'competitions', 'leaderboard', 
                    '-c', self.competition, '--show'
                ], capture_output=True, text=True, env=env)
                
                if lb_result.returncode == 0:
                    lines = lb_result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Header + data
                        logger.info(f"✅ Leaderboard active with {len(lines)-1} entries")
                        return True
                        
            logger.warning("⚠️ Competition status unclear")
            return True  # Proceed anyway
            
        except Exception as e:
            logger.warning(f"⚠️ Could not verify competition status: {e}")
            return True  # Proceed anyway
            
    def submit_file(self):
        """Submit file with multiple retry strategies"""
        logger.info("🚀 Starting submission process...")
        
        env = os.environ.copy()
        env['KAGGLE_CONFIG_DIR'] = self.config_dir
        
        # Try different submission approaches
        submission_attempts = [
            # Standard submission
            {
                'cmd': ['kaggle', 'competitions', 'submit', 
                       '-c', self.competition, 
                       '-f', self.submission_file, 
                       '-m', self.submission_message],
                'description': 'Standard submission'
            },
            # Submission with shorter message
            {
                'cmd': ['kaggle', 'competitions', 'submit', 
                       '-c', self.competition, 
                       '-f', self.submission_file, 
                       '-m', 'Production model'],
                'description': 'Short message submission'
            },
            # Alternative syntax
            {
                'cmd': ['kaggle', 'competitions', 'submit', 
                       self.competition,
                       '-f', self.submission_file, 
                       '-m', 'neural_network'],
                'description': 'Alternative syntax'
            }
        ]
        
        for i, attempt in enumerate(submission_attempts, 1):
            logger.info(f"🎯 Attempt {i}: {attempt['description']}")
            
            try:
                result = subprocess.run(
                    attempt['cmd'], 
                    capture_output=True, 
                    text=True, 
                    env=env,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info("🎉 SUBMISSION SUCCESSFUL!")
                    logger.info(f"📝 Output: {result.stdout}")
                    return True
                else:
                    logger.warning(f"⚠️ Attempt {i} failed:")
                    logger.warning(f"📝 stdout: {result.stdout}")
                    logger.warning(f"📝 stderr: {result.stderr}")
                    
                    # Wait before next attempt
                    if i < len(submission_attempts):
                        logger.info("⏳ Waiting 30 seconds before next attempt...")
                        time.sleep(30)
                        
            except subprocess.TimeoutExpired:
                logger.error(f"⏰ Attempt {i} timed out")
            except Exception as e:
                logger.error(f"❌ Attempt {i} error: {e}")
                
        logger.error("💥 All submission attempts failed")
        return False
        
    def verify_submission(self):
        """Verify submission was accepted"""
        logger.info("🔍 Verifying submission...")
        
        try:
            env = os.environ.copy()
            env['KAGGLE_CONFIG_DIR'] = self.config_dir
            
            result = subprocess.run([
                'kaggle', 'competitions', 'submissions', 
                '-c', self.competition
            ], capture_output=True, text=True, env=env, check=True)
            
            if result.stdout.strip():
                logger.info("✅ Submission verified in your submission history")
                logger.info(f"📝 Recent submissions:\n{result.stdout}")
                return True
            else:
                logger.warning("⚠️ No submissions found in history")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying submission: {e}")
            return False
            
    def run_submission(self):
        """Main submission workflow"""
        logger.info("🚀 Starting Kaggle submission workflow...")
        logger.info("=" * 60)
        
        # Step 1: Validate environment
        if not self.validate_environment():
            logger.error("💥 Environment validation failed")
            return False
            
        # Step 2: Validate submission file
        if not self.validate_submission_file():
            logger.error("💥 Submission file validation failed")
            return False
            
        # Step 3: Check competition status
        if not self.check_competition_status():
            logger.error("💥 Competition status check failed")
            return False
            
        # Step 4: Submit file
        if not self.submit_file():
            logger.error("💥 Submission failed")
            return False
            
        # Step 5: Verify submission
        self.verify_submission()
        
        logger.info("🎉 SUBMISSION WORKFLOW COMPLETED!")
        logger.info("=" * 60)
        return True


def main():
    """Main entry point"""
    print("🏆 Mitsui Commodity Prediction Challenge - Kaggle Submission")
    print("=" * 60)
    
    # Create submission manager
    manager = KaggleSubmissionManager()
    
    # Run submission
    success = manager.run_submission()
    
    if success:
        print("\n🎉 SUCCESS! Your model has been submitted to Kaggle!")
        print("🔗 Check your submission status at:")
        print(f"   https://www.kaggle.com/competitions/{manager.competition}/submissions")
    else:
        print("\n💥 FAILED! Check the logs above for detailed error information.")
        print("📋 Common solutions:")
        print("   1. Verify Kaggle credentials in .env/kaggle.json")
        print("   2. Check competition deadline hasn't passed")
        print("   3. Ensure submission file format is correct")
        print("   4. Try again - sometimes API has temporary issues")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())