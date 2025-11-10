import json
import re
import os
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict

def compare_single_file(annotated_file, gold_standard_file):
    """
    Compares a single pair of annotation files.
    """
    KEYS_TO_EXCLUDE = ['institution_involved', 'description', 'experiment_name']
    MINOR_ERROR_THRESHOLD = 0.8

    try:
        with open(annotated_file, 'r', encoding='utf-8') as f:
            annotated_data = json.load(f)
        with open(gold_standard_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except Exception as e:
        print(f"Could not process file: {os.path.basename(annotated_file)}. Reason: {e}")
        return None

    token_usage = annotated_data.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
    duration = annotated_data.get('duration_sec', 0)

    try:
        annotated_records = {int(str(item.get('pi_number', '0')).split()[-1]): item for item in annotated_data.get('crops', [])}
        gold_records = {int(str(item.get('pi_number', '0')).split()[-1]): item for item in gold_data.get('crops', [])}
    except (ValueError, KeyError) as e:
        print(f"Error parsing pi_number in {os.path.basename(annotated_file)}. Reason: {e}")
        return None

    if not gold_records:
        return None
        
    sample_record_keys = list(gold_records.values())[0].keys()
    keys_to_compare = [key for key in sample_record_keys if key not in KEYS_TO_EXCLUDE]

    summary_stats = {'exact_matches': 0, 'minor_errors': 0, 'major_errors': 0}
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pi_number, gold_record in gold_records.items():
        if pi_number not in annotated_records:
            continue
        
        annotated_record = annotated_records[pi_number]
        
        for key in keys_to_compare:
            gold_value, annotated_value = str(gold_record.get(key, '')), str(annotated_record.get(key, ''))
            processed_gold, processed_annotated = gold_value, annotated_value
            
            if key == 'name_of_crop':
                processed_gold = ' '.join(gold_value.split()[:2])
                processed_annotated = ' '.join(annotated_value.split()[:2])
                punc_remover = r'[^\w\s]'
                processed_gold = re.sub(punc_remover, '', processed_gold)
                processed_annotated = re.sub(punc_remover, '', processed_annotated)
            
            is_match = False
            gold_lower, annotated_lower = processed_gold.strip().lower(), processed_annotated.strip().lower()

            if gold_lower == annotated_lower or (gold_lower and annotated_lower and (gold_lower in annotated_lower or annotated_lower in gold_lower)):
                is_match = True
            
            field_stats[key]['total'] += 1
            if is_match:
                field_stats[key]['correct'] += 1
                summary_stats['exact_matches'] += 1
            else:
                similarity = SequenceMatcher(None, gold_value.lower(), annotated_value.lower()).ratio()
                error_type = "Minor" if similarity >= MINOR_ERROR_THRESHOLD else "Major"
                summary_stats[f"{error_type.lower()}_errors"] += 1
    
    return {'summary_stats': summary_stats, 'field_stats': field_stats, 'token_usage': token_usage, 'duration': duration}


def process_directory(annotated_dir, gold_dir):
    """
    Aggregates results for a single pair of directories.
    """
    aggregate_summary = defaultdict(int)
    aggregate_fields = defaultdict(lambda: {'correct': 0, 'total': 0})
    aggregate_tokens = defaultdict(int)

    aggregate_duration = 0
    files_processed = 0
    
    annotated_files = [f for f in os.listdir(annotated_dir) if f.endswith('.json')]

    for filename in annotated_files:
        annotated_path = os.path.join(annotated_dir, filename)
        gold_path = os.path.join(gold_dir, filename)

        if os.path.exists(gold_path):
            results = compare_single_file(annotated_path, gold_path)
            if results:
                files_processed += 1
                aggregate_duration += results['duration']
                for key, value in results['summary_stats'].items():
                    aggregate_summary[key] += value
                for key, value in results['field_stats'].items():
                    aggregate_fields[key]['correct'] += value['correct']
                    aggregate_fields[key]['total'] += value['total']
                for key, value in results['token_usage'].items():
                    aggregate_tokens[key] += value
        else:
            print(f"Warning: No matching gold standard for {filename}. Skipping.")
            
    if not aggregate_fields:
        return pd.DataFrame(), {}, 0, {}, 0
        
    results_df = pd.DataFrame.from_dict(aggregate_fields, orient='index')
    results_df.rename(columns={'correct': 'Correct', 'total': 'Total'}, inplace=True)
    results_df['Accuracy (%)'] = (results_df['Correct'] / results_df['Total'] * 100).round(2)
    
    return results_df, dict(aggregate_summary), files_processed, dict(aggregate_tokens), aggregate_duration


def run_batch_analysis(directory_pairs):
    """
    Main function: Processes a list of directory pairs and produces a final, pivoted report.
    """
    all_results_dfs = []
    all_metrics_data = []
    
    print(f"Starting batch analysis for {len(directory_pairs)} runs...")

    for annotated_dir, gold_dir in directory_pairs:
        run_name = os.path.basename(os.path.normpath(annotated_dir))
        print(f"\nProcessing Run: {run_name}...")
        
        field_df, _, files_processed, tokens_data, duration_data = process_directory(annotated_dir, gold_dir)
        
        if not field_df.empty:
            field_df['Run'] = run_name
            field_df['Files Processed'] = files_processed
            all_results_dfs.append(field_df)
        
        if tokens_data or duration_data:
            metrics = {'Run': run_name, 'duration_sec': duration_data, **tokens_data}
            all_metrics_data.append(metrics)
    
    if not all_results_dfs:
        print("\nNo data processed across all directories.")
        return pd.DataFrame()
        
    long_df = pd.concat(all_results_dfs).reset_index().rename(columns={'index': 'Field'})
    
    total_comparisons = long_df.groupby(['Run', 'Files Processed'])['Total'].sum().reset_index()
    
    pivoted_df = long_df.pivot_table(
        index=['Run', 'Files Processed'],
        columns='Field',
        values='Accuracy (%)'
    ).reset_index()
    
    final_wide_df = pd.merge(pivoted_df, total_comparisons, on=['Run', 'Files Processed'])
    
    if all_metrics_data:
        metrics_df = pd.DataFrame(all_metrics_data)
        final_wide_df = pd.merge(final_wide_df, metrics_df, on='Run')

    # Reorder columns for clarity
    field_columns = sorted([col for col in pivoted_df.columns if col not in ['Run', 'Files Processed']])
    metric_columns = sorted([col for col in metrics_df.columns if col not in ['Run']])
    final_wide_df = final_wide_df[['Run', 'Files Processed', 'Total'] + metric_columns + field_columns]
    
    print("\n" + "=" * 80)
    print("Final Pivoted Report Across All Runs")
    print("=" * 80)
    print(final_wide_df.to_string())
    
    return final_wide_df


# --- RUN THE BATCH PROCESS ---
if __name__ == "__main__":
    directory_pairs_to_process = [
        ('llm_output/gemini-1.5-pro-002/', 'evaluation_data/'),
        ('llm_output/gemini-1.5-flash-002/', 'evaluation_data/'),
        ('llm_output/gemini-2.5-flash/', 'evaluation_data/'),
        ('llm_output/gemini-2.5-pro/', 'evaluation_data/'),
        ('llm_output/gpt-3.5-turbo/', 'evaluation_data/'),
        ('llm_output/gpt-4o/', 'evaluation_data/'),
        ('llm_output/gpt-4o-mini/', 'evaluation_data/'),
        ('llm_output/gpt-5-2025-08-07/', 'evaluation_data/'),
        ('llm_output/gpt-5-mini-2025-08-07/', 'evaluation_data/'),
        ('llm_output/gpt-5-nano-2025-08-07/', 'evaluation_data/'),
        ('llm_output/llama3-8b/', 'evaluation_data/'),

    ]

    final_results_df = run_batch_analysis(directory_pairs_to_process)