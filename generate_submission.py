#!/usr/bin/env python3
"""
Generate final submission using trained 424 targets model
"""

import pandas as pd
import torch
import torch.nn as nn

class ProductionModel(nn.Module):
    """Production model for 424 targets."""
    
    def __init__(self, input_dim, n_targets=424):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, n_targets)
        )
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def apply_standardization(X, means, stds):
    """Apply standardization parameters."""
    X_std = []
    
    for row in range(X.shape[0]):
        std_row = []
        for col in range(min(X.shape[1], len(means))):  # Safety check
            std_val = (X[row, col] - means[col]) / stds[col]
            std_row.append(std_val)
        X_std.append(std_row)
    
    return X_std

def get_standardization_params():
    """Get standardization parameters from training data."""
    print("📊 Computing standardization parameters...")
    
    train_data = pd.read_csv('input/train.csv')
    feature_cols = [col for col in train_data.columns if col != 'date_id']
    
    X_train = train_data[feature_cols].fillna(method='ffill').fillna(0).values
    
    # Split to match training
    split_idx = int(0.8 * len(X_train))
    X_train_part = X_train[:split_idx]
    
    # Calculate means and stds
    means = []
    stds = []
    
    for col in range(X_train_part.shape[1]):
        col_data = X_train_part[:, col]
        mean_val = sum(col_data) / len(col_data)
        var_val = sum((x - mean_val)**2 for x in col_data) / (len(col_data) - 1)
        std_val = var_val**0.5 if var_val > 0 else 1.0
        
        means.append(mean_val)
        stds.append(std_val)
    
    print(f"   Computed {len(means)} feature parameters")
    return means, stds

def generate_final_submission():
    """Generate final competition submission."""
    print("🔮 Generating Final Submission")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Load model
        print("📦 Loading trained model...")
        checkpoint = torch.load('production_424_model.pth', map_location=device)
        
        # Get standardization parameters
        means, stds = get_standardization_params()
        
        # Load test data
        print("📁 Loading test data...")
        test_data = pd.read_csv('input/test.csv')
        print(f"   Test shape: {test_data.shape}")
        
        # Get train features to match column order
        train_data = pd.read_csv('input/train.csv')
        train_feature_cols = [col for col in train_data.columns if col != 'date_id']
        
        # Prepare test features (only columns that exist in both)
        available_cols = [col for col in train_feature_cols if col in test_data.columns]
        print(f"   Matching features: {len(available_cols)}")
        
        X_test_raw = test_data[available_cols].fillna(method='ffill').fillna(0).values
        
        # Apply standardization
        print("⚡ Standardizing features...")
        X_test_std = apply_standardization(X_test_raw, means[:len(available_cols)], stds[:len(available_cols)])
        
        # Create model
        model = ProductionModel(len(available_cols), 424).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"   Model loaded with Sharpe score: {checkpoint['sharpe_score']:.4f}")
        
        # Generate predictions
        print("🧠 Generating predictions...")
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_std).to(device)
            predictions = model(X_tensor).cpu()
        
        print(f"   Predictions shape: {predictions.shape}")
        
        # Create submission
        print("📝 Creating submission file...")
        submission = pd.DataFrame({'date_id': test_data['date_id']})
        
        for i in range(424):
            pred_values = []
            for j in range(predictions.shape[0]):
                pred_values.append(float(predictions[j, i].item()))
            submission[f'target_{i}'] = pred_values
        
        # Save submission
        submission.to_csv('submission_final_424.csv', index=False)
        
        print(f"✅ Submission saved: submission_final_424.csv")
        print(f"   Shape: {submission.shape}")
        print(f"   Columns: {len(submission.columns)} (1 date_id + 424 targets)")
        
        # Show sample predictions
        print(f"\n📋 Sample predictions:")
        print(f"   Date ID range: {submission['date_id'].min()} - {submission['date_id'].max()}")
        print(f"   Target_0 range: {submission['target_0'].min():.6f} - {submission['target_0'].max():.6f}")
        print(f"   Target_423 range: {submission['target_423'].min():.6f} - {submission['target_423'].max():.6f}")
        
        return submission
        
    except Exception as e:
        print(f"❌ Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    submission = generate_final_submission()
    
    if submission is not None:
        print(f"\n🎉 Final submission ready!")
        print(f"   File: submission_final_424.csv")
        print(f"   Ready for competition upload")
    else:
        print(f"\n❌ Submission generation failed")

if __name__ == "__main__":
    main()