"""
Local Mock for Kaggle Evaluation Module
Provides minimal interface for local development without grpc dependency
"""
import pandas as pd
import polars as pl
import os
from pathlib import Path

class MockMitsuiInferenceServer:
    """
    Mock version of kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer
    for local development and testing
    """
    
    def __init__(self, predict_function):
        self.predict_function = predict_function
        print("🧪 Mock Mitsui Inference Server initialized for local testing")
        print("   Note: This is a simplified version for development")
        print("   Upload to Kaggle to use official inference server")
    
    def serve(self):
        """Mock serve method for competition environment"""
        print("🏃 Mock serve() called - simulating competition environment")
        print("   In actual Kaggle environment, this would handle live inference")
        return self._local_test()
    
    def run_local_gateway(self, data_paths=None):
        """Mock local gateway for testing"""
        print("🧪 Mock run_local_gateway() called")
        print("   Simulating local test data processing...")
        return self._local_test()
    
    def _local_test(self):
        """
        Simple local test using a few sample data points
        """
        print("\n🔬 Running local mock test...")
        
        # Create sample test data (mimicking what Kaggle would provide)
        sample_dates = [1500, 1501, 1502]  # Sample date_ids
        
        for i, date_id in enumerate(sample_dates):
            print(f"\n📅 Testing prediction for date_id: {date_id}")
            
            # Create mock test DataFrame
            test_data = pl.DataFrame({
                'date_id': [date_id],
                'feature_1': [0.1 * i],
                'feature_2': [0.2 * i],
                # Add more mock features as needed
            })
            
            # Create mock lag DataFrames (empty for now)
            empty_df = pl.DataFrame({'date_id': [date_id]})
            
            try:
                # Call the prediction function
                result = self.predict_function(
                    test=test_data,
                    label_lags_1_batch=empty_df,
                    label_lags_2_batch=empty_df,
                    label_lags_3_batch=empty_df,
                    label_lags_4_batch=empty_df
                )
                
                print(f"   ✅ Prediction successful: {result.shape}")
                print(f"   📊 Sample predictions: {result.select(pl.all()[:5]).to_pandas().iloc[0].to_dict()}")
                
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        
        print("\n🏁 Mock testing complete!")
        print("   Upload notebook to Kaggle for actual submission")

# Mock module structure to match kaggle_evaluation
class MockKaggleEvaluation:
    class mitsui_inference_server:
        MitsuiInferenceServer = MockMitsuiInferenceServer

# Create the mock module
import sys
sys.modules['kaggle_evaluation'] = MockKaggleEvaluation()
sys.modules['kaggle_evaluation.mitsui_inference_server'] = MockKaggleEvaluation.mitsui_inference_server()

print("📦 Local Kaggle Evaluation Mock installed successfully!")
print("   You can now import kaggle_evaluation.mitsui_inference_server")
print("   This provides local testing capabilities without grpc dependency")