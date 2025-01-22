import math
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse
import subprocess
import sys

"""Code adapted from https://physionet.org/content/radgraph/1.0.0: models/inference.py."""

# Add DyGIE to Python path
dygie_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dygie')
sys.path.append(dygie_path)

def preprocess_reports(data_path, start=None, end=None):
    """Preprocesses reports for RadGraph.
    
    Args:
        data_path: Path to CSV file containing reports
        start: Start index (optional)
        end: End index (optional)
    """
    # Read the CSV file
    df = pd.read_csv(data_path)
    
    # If start/end not provided, process all reports
    if start is None:
        start = 0
    if end is None:
        end = len(df)
    
    # Process reports
    processed_data = []
    for idx in range(start, end):
        row = df.iloc[idx]
        doc = {
            "doc_key": str(row["study_id"]),
            "sentences": [row["report"].split()],
            "ner": [[]], 
            "relations": [[]]
        }
        processed_data.append(doc)
    
    # Save to temporary file
    temp_input = os.path.join(os.path.dirname(data_path), 'temp_input.json')
    with open(temp_input, 'w') as f:
        for doc in processed_data:
            f.write(json.dumps(doc) + '\n')
    
    return temp_input

def run_inference(model_path, data_source, output_file):
    """Runs RadGraph inference."""
    # Ensure model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Run allennlp predict with explicit path to dygie
    cmd = [
        "allennlp",
        "predict",
        model_path,
        data_source,
        "--predictor",
        "dygie",
        "--include-package",
        "dygie",
        "--use-dataset-reader",
        "--output-file",
        output_file,
        "--cuda-device",
        "-1",
        "--silent",
        "--batch-size",
        "1"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(
                os.environ,
                PYTHONPATH=f"{dygie_path}:{os.environ.get('PYTHONPATH', '')}"
            )
        )
        if result.returncode != 0:
            print("Error running allennlp:", result.stderr)
            raise RuntimeError("RadGraph inference failed")
        print("AllennNLP output:", result.stdout)
    except Exception as e:
        print(f"Exception running allennlp: {str(e)}")
        raise
    
    # Verify the output file exists and has content
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("RadGraph inference produced no output")
    
    return output_file

def postprocess_reports(data_source, data_split, output_file):
    """Post-processes the RadGraph output."""
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"RadGraph output file not found: {output_file}")
        
    # Read the file line by line since it contains one JSON object per line
    predictions = {}
    with open(output_file, 'r') as f:
        for line in f:
            try:
                # Parse each line as a separate JSON object
                pred = json.loads(line.strip())
                if 'doc_key' in pred:
                    doc_key = pred['doc_key']
                    predictions[doc_key] = {
                        'entities': pred.get('predicted_ner', []),
                        'relations': pred.get('predicted_relations', [])
                    }
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line in output file: {e}")
                continue
    
    if not predictions:
        raise ValueError("No valid predictions found in output file")
        
    return predictions

def postprocess_individual_report(file, final_dict, data_source=None, data_split="inference"):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = data_split

        if file['doc_key'] in final_dict:  # Handle duplicate study IDs.
            final_dict[file['doc_key'] + '+'] = temp_dict
        else:
            final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
def get_entity(n,r,s):
    
    """Gets the entities for individual reports
    
    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence
        
    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json 
    
    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
        for i,val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab,str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx+1)] = temp_dict
    
    return dict_entity

def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    # os.system("rm temp_file_list.json")
    os.system("rm temp_dygie_input.json")
    os.system("rm temp_dygie_output.json")

def _json_to_csv(path, csv_path):
    with open(path, "r") as f:
        match_results = json.load(f)
    reconstructed_reports = []
    for _, (_, train, match) in match_results.items():
        test_report_id = match[0][0][:8]
        reconstructed_reports.append((test_report_id, train))
    pd.DataFrame(reconstructed_reports, columns=["study_id", "report"]).to_csv(csv_path)

def _add_ids_column(
            csv_path, study_id_csv_path, output_path):
    with open(csv_path, "r") as f:
        generated_reports = pd.read_csv(f)
    with open(study_id_csv_path, "r") as f:
        ids_csv = pd.read_csv(f)
        study_ids = ids_csv["study_id"]
        dicom_ids = ids_csv["dicom_id"]
        subject_ids = ids_csv["subject_id"]
    generated_reports["study_id"] = study_ids
    generated_reports["dicom_id"] = dicom_ids
    generated_reports["subject_id"] = subject_ids
    #generated_reports.drop_duplicates(subset=["study_id"], keep="first")
    generated_reports.to_csv(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=False,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')

    
    args = parser.parse_args()
    
    run(args.model_path, args.data_path, args.out_path, args.cuda_device)
