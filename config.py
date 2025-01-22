import os

# Paths to your data
GT_REPORTS = 'reports/gt_reports.csv'
PREDICTED_REPORTS = 'reports/predicted_reports.csv'
OUT_FILE = 'reports/metric_scores.txt'

# Use the actual path where you downloaded chexbert.pth
CHEXBERT_PATH = '/Users/boghdady/Documents/GitHub/CXR-Report-Metric/chexbert.pth'

# Use the actual path where you downloaded the RadGraph model
RADGRAPH_PATH = '/Users/boghdady/Documents/GitHub/CXR-Report-Metric/radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0/models/model_checkpoint/model.tar.gz'

# Whether to use inverse document frequency (idf) for BERTScore
USE_IDF = False
