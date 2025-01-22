import json
import numpy as np
import os
import re
import pandas as pd
import pickle
import torch
import subprocess

from bert_score import BERTScorer
import evaluate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import config
from CXRMetric.radgraph_evaluate_model import run_radgraph

"""Computes 4 individual metrics and a composite metric on radiology reports."""


CHEXBERT_PATH = config.CHEXBERT_PATH
RADGRAPH_PATH = config.RADGRAPH_PATH

NORMALIZER_PATH = "CXRMetric/normalizer.pkl"
COMPOSITE_METRIC_V0_PATH = "CXRMetric/composite_metric_model.pkl"
COMPOSITE_METRIC_V1_PATH = "CXRMetric/radcliq-v1.pkl"

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]

cache_path = "cache/"
pred_embed_path = os.path.join(cache_path, "pred_embeddings.pt")
gt_embed_path = os.path.join(cache_path, "gt_embeddings.pt")
weights = {"bigram": (1/2., 1/2.)}
composite_metric_col_v0 = "RadCliQ-v0"
composite_metric_col_v1 = "RadCliQ-v1"


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

def add_bleu_col(gt_df, pred_df):
    """Computes BLEU-2 and adds scores as a column to prediction df."""
    # Initialize the BLEU metric from evaluate
    bleu = evaluate.load('bleu')
    
    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = str(row[REPORT_COL_NAME])  # Convert to string format
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        predicted_report = str(pred_row[REPORT_COL_NAME].values[0])  # Convert to string format
        
        if len(pred_row) == 1:
            # Calculate BLEU score using evaluate
            score = bleu.compute(
                predictions=[predicted_report],  # Pass full string
                references=[[gt_report]],        # References need to be nested list
                max_order=2                      # For BLEU-2
            )['bleu']
            
            _index = pred_df.index[
                pred_df[STUDY_ID_COL_NAME]==row[STUDY_ID_COL_NAME]].tolist()[0]
            pred_df.at[_index, "bleu_score"] = score
            
    return pred_df

def add_bertscore_col(gt_df, pred_df, use_idf):
    """Computes BERTScore and adds scores as a column to prediction df."""
    test_reports = gt_df[REPORT_COL_NAME].tolist()
    test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
    method_reports = pred_df[REPORT_COL_NAME].tolist()
    method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
        idf=use_idf,
        idf_sents=test_reports)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    
    list_label_embeds = []
    list_pred_embeds = []
    
    for data_idx in sorted(label_embeds.keys()):
        embed = label_embeds[data_idx]
        if isinstance(embed, list):
            embed = torch.tensor(embed)
        list_label_embeds.append(embed)
        
        pred_embed = pred_embeds[data_idx]
        if isinstance(pred_embed, list):
            pred_embed = torch.tensor(pred_embed)
        list_pred_embeds.append(pred_embed)
    
    try:
        np_label_embeds = torch.stack(list_label_embeds, dim=0).cpu().numpy()
        np_pred_embeds = torch.stack(list_pred_embeds, dim=0).cpu().numpy()
    except:
        # If stacking fails, try converting individual elements
        np_label_embeds = np.array([x.cpu().numpy() for x in list_label_embeds])
        np_pred_embeds = np.array([x.cpu().numpy() for x in list_pred_embeds])
    
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    
    pred_df["semb_score"] = scores
    return pred_df

def add_radgraph_col(pred_df, entities_path, relations_path):
    """Computes RadGraph scores and adds them as a column to prediction df."""
    # Load the entities and relations files
    with open(entities_path, "r") as f:
        entities_dict = json.load(f)
    with open(relations_path, "r") as f:
        relations_dict = json.load(f)
        
    # Create a mapping from study_id to scores
    study_id_to_radgraph = {}
    
    # Process each study in the predictions
    for study_id in entities_dict['pred'].keys():
        # Get predictions and ground truth
        pred_entities = entities_dict['pred'].get(study_id, [])
        gt_entities = entities_dict['gt'].get(study_id, [])
        pred_relations = relations_dict['pred'].get(study_id, [])
        gt_relations = relations_dict['gt'].get(study_id, [])
        
        # Calculate F1 scores
        entity_f1 = calculate_entity_f1(pred_entities, gt_entities)
        relation_f1 = calculate_relation_f1(pred_relations, gt_relations)
        
        # Combine scores (you can adjust the weights if needed)
        combined_score = (entity_f1 + relation_f1) / 2
        study_id_to_radgraph[str(study_id)] = combined_score
    
    # Add scores to dataframe
    radgraph_scores = []
    for _, row in pred_df.iterrows():
        study_id = str(row[STUDY_ID_COL_NAME])  # Convert to string
        if study_id in study_id_to_radgraph:
            radgraph_scores.append(study_id_to_radgraph[study_id])
        else:
            print(f"Warning: No RadGraph score for study_id {study_id}")
            radgraph_scores.append(0.0)  # Default score
    
    pred_df["radgraph_combined"] = radgraph_scores
    return pred_df

def calculate_entity_f1(pred_entities, gt_entities):
    """Calculate F1 score for entities."""
    if not gt_entities and not pred_entities:
        return 1.0
    if not gt_entities or not pred_entities:
        return 0.0
        
    # Convert entities to hashable format
    def entity_to_hashable(entity):
        # Convert entity to a tuple of (start, end, label)
        # Assuming entity format is [start_idx, end_idx, label]
        return tuple(entity)
    
    try:
        # Convert to sets of hashable entities
        pred_set = {entity_to_hashable(entity) for entity in pred_entities}
        gt_set = {entity_to_hashable(entity) for entity in gt_entities}
        
        # Calculate precision and recall
        true_positives = len(pred_set.intersection(gt_set))
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gt_set) if gt_set else 0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    except Exception as e:
        print(f"Warning: Error calculating entity F1: {e}")
        print(f"Pred entities: {pred_entities}")
        print(f"GT entities: {gt_entities}")
        return 0.0

def calculate_relation_f1(pred_relations, gt_relations):
    """Calculate F1 score for relations."""
    if not gt_relations and not pred_relations:
        return 1.0
    if not gt_relations or not pred_relations:
        return 0.0
        
    # Convert relations to hashable format
    def relation_to_hashable(relation):
        # Convert relation to a tuple of (head_start, head_end, tail_start, tail_end, label)
        # Assuming relation format is [head_start, head_end, tail_start, tail_end, label]
        return tuple(relation)
    
    try:
        # Convert to sets of hashable relations
        pred_set = {relation_to_hashable(relation) for relation in pred_relations}
        gt_set = {relation_to_hashable(relation) for relation in gt_relations}
        
        # Calculate precision and recall
        true_positives = len(pred_set.intersection(gt_set))
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gt_set) if gt_set else 0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    except Exception as e:
        print(f"Warning: Error calculating relation F1: {e}")
        print(f"Pred relations: {pred_relations}")
        print(f"GT relations: {gt_relations}")
        return 0.0

def calc_metric(gt_csv, pred_csv, out_csv, use_idf):
    """Computes four metrics and composite metric scores."""
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_path, exist_ok=True)
    
    cache_gt_csv = os.path.join(
        os.path.dirname(gt_csv), f"cache_{os.path.basename(gt_csv)}")
    cache_pred_csv = os.path.join(
        os.path.dirname(pred_csv), f"cache_{os.path.basename(pred_csv)}")
    
    print(f"Loading CSV files...")
    gt = pd.read_csv(gt_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)
    pred = pd.read_csv(pred_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)

    # Keep intersection of study IDs
    gt_study_ids = set(gt[STUDY_ID_COL_NAME])
    pred_study_ids = set(pred[STUDY_ID_COL_NAME])
    shared_study_ids = gt_study_ids.intersection(pred_study_ids)
    print(f"Number of shared study IDs: {len(shared_study_ids)}")
    
    gt = gt.loc[gt[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()
    pred = pred.loc[pred[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()

    gt.to_csv(cache_gt_csv)
    pred.to_csv(cache_pred_csv)

    # Check that length and study IDs are the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)
    assert (gt[STUDY_ID_COL_NAME].equals(pred[STUDY_ID_COL_NAME]))

    print("Computing BLEU scores...")
    pred = add_bleu_col(gt, pred)

    print("Computing BERTScore...")
    pred = add_bertscore_col(gt, pred, use_idf)

    # Run CheXbert encoding
    print("Running CheXbert encoding...")
    print(f"Using CheXbert checkpoint: {CHEXBERT_PATH}")
    
    # Check if CheXbert checkpoint exists
    if not os.path.exists(CHEXBERT_PATH):
        raise FileNotFoundError(f"CheXbert checkpoint not found at: {CHEXBERT_PATH}")
    
    # Run encode.py for predictions using subprocess
    encode_pred_cmd = [
        "python",
        os.path.join(os.path.dirname(__file__), "CheXbert/src/encode.py"),
        "-c", CHEXBERT_PATH,
        "-d", cache_pred_csv,
        "-o", pred_embed_path
    ]
    print(f"Running command: {' '.join(encode_pred_cmd)}")
    try:
        result = subprocess.run(
            encode_pred_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Prediction encoding output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)
        raise RuntimeError(f"CheXbert encoding failed for predictions: {e.stderr}")
    
    # Run encode.py for ground truth using subprocess
    encode_gt_cmd = [
        "python",
        os.path.join(os.path.dirname(__file__), "CheXbert/src/encode.py"),
        "-c", CHEXBERT_PATH,
        "-d", cache_gt_csv,
        "-o", gt_embed_path
    ]
    print(f"Running command: {' '.join(encode_gt_cmd)}")
    try:
        result = subprocess.run(
            encode_gt_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Ground truth encoding output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)
        raise RuntimeError(f"CheXbert encoding failed for ground truth: {e.stderr}")
    
    # Verify embedding files were created
    if not os.path.exists(pred_embed_path):
        raise FileNotFoundError(f"Prediction embeddings not found at: {pred_embed_path}")
    if not os.path.exists(gt_embed_path):
        raise FileNotFoundError(f"Ground truth embeddings not found at: {gt_embed_path}")
    
    print("Computing s_emb scores...")
    pred = add_semb_col(pred, pred_embed_path, gt_embed_path)

    print("Computing RadGraph scores...")
    entities_path = os.path.join(cache_path, "entities_cache.json")
    relations_path = os.path.join(cache_path, "relations_cache.json")
    run_radgraph(cache_gt_csv, cache_pred_csv, cache_path, RADGRAPH_PATH,
                 entities_path, relations_path)
    pred = add_radgraph_col(pred, entities_path, relations_path)

    # compute composite metric: RadCliQ-v0
    with open(COMPOSITE_METRIC_V0_PATH, "rb") as f:
        composite_metric_v0_model = pickle.load(f)
    with open(NORMALIZER_PATH, "rb") as f:
        normalizer = pickle.load(f)
    # normalize
    input_data = np.array(pred[COLS])
    norm_input_data = normalizer.transform(input_data)
    # generate new col
    radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
    pred[composite_metric_col_v0] = radcliq_v0_scores

    # compute composite metric: RadCliQ-v1
    with open(COMPOSITE_METRIC_V1_PATH, "rb") as f:
        composite_metric_v1_model = pickle.load(f)
    input_data = np.array(pred[COLS])
    radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
    pred[composite_metric_col_v1] = radcliq_v1_scores

    # save results in the out folder
    pred.to_csv(out_csv)
