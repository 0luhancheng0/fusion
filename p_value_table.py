"""
Generate summary table with performance improvements and p-values
"""

import pandas as pd
from scipy import stats


def generate_p_value_table(baseline_performance, fusion_data):
    """
    Generate a summary table comparing fusion methods to baseline with p-values.
    
    Args:
        baseline_performance (dict): Dictionary with baseline performance metrics
        fusion_data (DataFrame): DataFrame containing fusion model results
        
    Returns:
        DataFrame: Summary table with improvements and p-values
    """
    summary_table = []
    
    for model_type in ['TextualEmbeddings (Baseline)', 'EarlyFusion', 'GatedFusion', 'LowRankFusion']:
        row = {'Model': model_type}
        
        if model_type == 'TextualEmbeddings (Baseline)':
            row['Accuracy'] = f"{baseline_performance['acc/test']:.4f}"
            row['AUROC'] = f"{baseline_performance['lp_hard/auc']:.4f}"
            row['Acc_Improve'] = "0.0%"
            row['AUC_Improve'] = "0.0%"
            row['Acc_P_Value'] = "-"
            row['AUC_P_Value'] = "-"
        else:
            model_data = fusion_data[fusion_data['model_type'] == model_type]
            
            # Calculate accuracy metrics
            acc_values = model_data['acc/test'].dropna()
            acc_mean = acc_values.mean()
            acc_improvement = ((acc_mean - baseline_performance['acc/test']) / baseline_performance['acc/test']) * 100
            _, acc_p_value = stats.ttest_1samp(acc_values, baseline_performance['acc/test'])
            
            # Calculate AUROC metrics
            auc_values = model_data['lp_hard/auc'].dropna()
            auc_mean = auc_values.mean()
            auc_improvement = ((auc_mean - baseline_performance['lp_hard/auc']) / baseline_performance['lp_hard/auc']) * 100
            _, auc_p_value = stats.ttest_1samp(auc_values, baseline_performance['lp_hard/auc'])
            
            row['Accuracy'] = f"{acc_mean:.4f}"
            row['AUROC'] = f"{auc_mean:.4f}"
            row['Acc_Improve'] = f"{acc_improvement:+.1f}%"
            row['AUC_Improve'] = f"{auc_improvement:+.1f}%"
            row['Acc_P_Value'] = f"{acc_p_value:.2e}"
            row['AUC_P_Value'] = f"{auc_p_value:.2e}"
        
        summary_table.append(row)
    
    return pd.DataFrame(summary_table)


def print_p_value_table(baseline_performance, fusion_data):
    """
    Generate and print the p-value table.
    
    Args:
        baseline_performance (dict): Dictionary with baseline performance metrics
        fusion_data (DataFrame): DataFrame containing fusion model results
    """
    summary_df = generate_p_value_table(baseline_performance, fusion_data)
    print("SUMMARY TABLE: Performance Improvement vs Baseline")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    return summary_df
