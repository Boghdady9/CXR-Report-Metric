import pandas as pd
from torch.utils.data import Dataset

class UnlabeledDataset(Dataset):
    """Dataset class for unlabeled reports."""
    
    def __init__(self, reports_path, report_col):
        """
        Args:
            reports_path (string): Path to csv containing reports
            report_col (string): Name of column containing reports
        """
        self.df = pd.read_csv(reports_path)
        self.report_col = report_col
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        report = str(self.df.iloc[idx][self.report_col])
        return report
