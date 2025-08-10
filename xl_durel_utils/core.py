import pandas as pd
import numpy as np
from typing import Optional, List
from scipy.stats import pearsonr, spearmanr
from tabulate import tabulate
from scipy.optimize import minimize
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt


def tokenize_truncate_decode(sentence, positions, tokenizer, max_seq_len=128):
    def center_sentence(input_ids, positions, max_seq_len):
        left = input_ids[:positions[0]]
        right = input_ids[positions[1]:]

        overflow_left = len(left) - int((max_seq_len - len(input_ids[positions[0]:positions[1]])) / 2)

        overflow_right = len(right) - int((max_seq_len - len(input_ids[positions[0]:positions[1]])) / 2)


        if overflow_left > 0 and overflow_right > 0:
            left = left[overflow_left:]
            right = right[:len(right)-overflow_right]
        elif overflow_left > 0 and overflow_right <= 0:
            left = left[overflow_left:]
        else:
            right = right[:len(right)-overflow_right]

        return left + input_ids[positions[0]:positions[1]] + right

    def tokenize_sentence(sentence, positions):
        left, target, right = sentence[:positions[0]], sentence[positions[0]:positions[1]], sentence[positions[1]:]

        token_positions = [0, 0]
        tokens = []

        if left:
            tokens += tokenizer.tokenize(left)
        token_positions[0] = len(tokens)
        tokens += tokenizer.tokenize('<t>')
        target_subtokens = tokenizer.tokenize(target)
        tokens += target_subtokens
        tokens += tokenizer.tokenize('</t>', max_length=128)
        token_positions[1] = len(tokens)
        if right:
            tokens += tokenizer.tokenize(right)

        return tokens, token_positions

    # Step 1: Tokenize the sentence with target marking
    tokens, token_positions = tokenize_sentence(sentence, positions)
    
    n_extra_tokens = 2  
    len_input = len(tokens) + n_extra_tokens
    
    if len_input > max_seq_len:
        tokens = center_sentence(tokens, token_positions, max_seq_len - n_extra_tokens)
        
    # Step 4: Decode back to text
    decoded_text = tokenizer.convert_tokens_to_string(tokens)
    return decoded_text

def calculate_spearman(df: pd.DataFrame, group_by: Optional[list] = None, return_df:Optional[bool] = True) -> Optional[pd.DataFrame]:
     
        if group_by:
            rows = []
            for group_name, group_df in df.groupby(group_by):
                labels = group_df['mapped_label']
                scores = group_df['similarity']
                pearson, _ = pearsonr(labels, scores)
                spearman, _ = spearmanr(labels, scores)

                group_values = group_name if isinstance(group_name, tuple) else (group_name,)
                rows.append((*group_values, pearson, spearman))

            columns = group_by + ['pearson', 'spearman']
            results_df = pd.DataFrame(rows, columns=columns)
            if return_df:

                return results_df
            else:
                print(tabulate(results_df, headers='keys', tablefmt="grid", floatfmt=".4f"))
        return None
     

def calculate_krippendorff(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    return_df: Optional[bool] = True
) -> Optional[pd.DataFrame]:
   
    if group_by:
        rows = []

        # Create grouped dictionaries for lookup
        dev_groups = dict(tuple(dev_df.groupby(group_by)))
        test_groups = dict(tuple(test_df.groupby(group_by)))


        for group_name, test_group in test_groups.items():
            dev_group = dev_groups.get(group_name)

            if dev_group is None:
                print(f"Skipping group {group_name}: not found in dev set.")
                continue

            # Compute thresholds using dev group
            dev_labels = dev_group['label'].astype(float)
            dev_scores = dev_group['similarity']
            

            optimized_bins = calc_threshold(dev_scores, dev_labels, groupname=group_name)
           
            test_scores = test_group['similarity'].astype(float)
            test_labels = test_group['label'].astype(float)
            unique_levels = sorted(set(test_labels))
            k=len(unique_levels)
            if k < 2:
                    raise ValueError("Need at least two distinct labels to threshold.")
            if k==2:
                labels = [2.0, 4.0]
                measurement = 'nominal'
            else:
                labels = [float(i) for i in range(1, k+1)]
                measurement = 'ordinal'

            predictions = pd.cut(
                test_scores,
                bins=optimized_bins,
                labels=labels
            ).astype(float)

            # Calculate Krippendorff's alpha
            data = [test_labels.tolist(), predictions.tolist()]
            alpha = krippendorff.alpha(reliability_data=data, level_of_measurement=measurement)
            # print("test label ", data[0])
            # print("predictions", data[1])
            # print("test_score", test_scores.tolist())

            group_values = group_name if isinstance(group_name, tuple) else (group_name,)
            rows.append((*group_values, alpha))

        # Create final results DataFrame
        columns = group_by + ['krippendorff_alpha']
        results_df = pd.DataFrame(rows, columns=columns)

        if return_df:
            return results_df
        else:
            print(tabulate(results_df, headers='keys', tablefmt="grid", floatfmt=".4f"))

    else:
        print("Something went wrong, no group_by specified, using full dev + test set")


def calc_threshold(cosine_sim, median_cleaned_label, groupname=None):
        min_sim = float(min(cosine_sim))
        max_sim = float(max(cosine_sim))
        
        # initial bins
        unique_levels = sorted(set(median_cleaned_label))
        k=len(unique_levels)
        if k < 2:
                raise ValueError("Need at least two distinct labels to threshold.")
        
        n = k - 1 
        delta = (max_sim - min_sim) / (n + 1)
        bins = [min_sim + delta*(i+1) for i in range(n)]

        # loss function
        def min_loss(bins, cos_sim, y):
            bins = sorted([-np.inf] + list(bins) + [np.inf])
            if k==2:
                labels = [2.0, 4.0]
                measurement = 'nominal'
            else:
                 labels = [float(i) for i in range(1, k+1)]
                 measurement = 'ordinal'  
            binned_similarities = pd.cut(cos_sim, bins=bins, labels=labels)
            y_pred = binned_similarities.tolist()
            y = [float(i) for i in y]
            data = [y, y_pred]
           
            alpha = krippendorff.alpha(reliability_data=data, level_of_measurement=measurement)
            return 1 - alpha
        
        # optimizing bin edges
        result = minimize(min_loss, bins, args=(cosine_sim, median_cleaned_label), method='nelder-mead')
        optimized_bins = sorted([-np.inf] + result.x.tolist() + [np.inf])
        print(groupname, optimized_bins )

        return optimized_bins

def plot(df: pd.DataFrame, metrics_to_plot: list, group_by=['dataset', 'language']):

    if not all(col in df.columns for col in group_by):
        raise ValueError(f"DataFrame must contain {group_by} columns")

    df = df.copy()
    df['dataset_language'] = df[group_by[0]] + ' / ' + df[group_by[1]]

    grouped = df.groupby('dataset_language')[metrics_to_plot].mean().reset_index()

    melted = grouped.melt(
        id_vars='dataset_language',
        value_vars=metrics_to_plot,
        var_name='metric',
        value_name='value'
    )

    # Pivot for heatmap
    pivot_df = melted.pivot_table(
        index='dataset_language',
        columns='metric',
        values='value',
        aggfunc='mean'
    )
    pivot_df = pivot_df.round(2)

    # Create annotation matrix with no leading zero before decimal point
    annot = pivot_df.applymap(lambda x: f"{x:.2f}".lstrip("0") if pd.notnull(x) else "")

    # Plot heatmap with fixed color scale
    plt.figure(figsize=(6, 0.6 * len(pivot_df)))
    sns.heatmap(
        pivot_df,
        annot=annot,
        cmap='YlGnBu',
        fmt="",
        linewidths=0.5,
        cbar=True,
        vmin=0.0,
        vmax=1.0
    )
    plt.title("Evaluation Across Datasets & Languages")
    plt.xlabel("Metric")
    plt.ylabel("Dataset / Language")
    plt.tight_layout()
    plt.show()
