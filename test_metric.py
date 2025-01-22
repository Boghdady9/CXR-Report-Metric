import config
from CXRMetric.run_eval import calc_metric
from CXRMetric.run_eval import CompositeMetric
import pandas as pd
import os
import shutil

gt_reports = config.GT_REPORTS
predicted_reports = config.PREDICTED_REPORTS
out_file = config.OUT_FILE
use_idf = config.USE_IDF

# Create necessary directories
os.makedirs('cache', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Verify model files exist
if not os.path.exists(config.CHEXBERT_PATH):
    raise FileNotFoundError(f"CheXbert model not found at {config.CHEXBERT_PATH}")
if not os.path.exists(config.RADGRAPH_PATH):
    raise FileNotFoundError(f"RadGraph model not found at {config.RADGRAPH_PATH}")

# Clean cache directory
if os.path.exists('cache'):
    shutil.rmtree('cache')
os.makedirs('cache')

# Create sample ground truth reports
gt_data = {
    'study_id': ['001', '002', '003'],
    'report': [
        'The chest x-ray shows clear lungs with no infiltrates. Heart size is normal.',
        'There is a small pleural effusion in the right lung base. No pneumothorax.',
        'Clear lungs bilaterally. No acute cardiopulmonary abnormality.'
    ]
}

# Create sample predicted reports
pred_data = {
    'study_id': ['001', '002', '003'],
    'report': [
        'Chest x-ray demonstrates clear lung fields. Cardiac silhouette is normal.',
        'Right-sided pleural effusion noted. No evidence of pneumothorax.',
        'Lungs are clear. No acute cardiopulmonary findings.'
    ]
}

# Save to CSV files
gt_reports = 'reports/gt_reports.csv'
predicted_reports = 'reports/predicted_reports.csv'
out_file = 'reports/metric_scores.txt'

pd.DataFrame(gt_data).to_csv(gt_reports, index=False)
pd.DataFrame(pred_data).to_csv(predicted_reports, index=False)

if __name__ == "__main__":

    calc_metric(gt_reports, predicted_reports, out_file, use_idf)
