import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src import data_loader

def analyze_split(name, json_path):
    print(f"--- Analyzing {name} [{json_path}] ---")
    loader = data_loader.CocoLoader(train_json_path=json_path)
    anns = loader.get_annotations()
    
    counts = Counter()
    for ann in anns:
        cat_name = loader.get_category_name(ann["category_id"])
        counts[cat_name] += 1
        
    total = sum(counts.values())
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])
    df['Percentage'] = (df['Count'] / total * 100).map('{:.1f}%'.format)
    df.index.name = 'Class'
    
    print(df.sort_values('Count', ascending=False))
    print(f"Total: {total}\n")
    return df

def main():
    print("Dataset Imbalance Analysis\n")
    
    train_df = analyze_split("TRAIN", config.TRAIN_JSON)
    val_df   = analyze_split("VAL",   config.VAL_JSON)
    
    # Calculate weights suggestion (inverse frequency)
    print("--- Class Weights Suggestion (Inverse Freq) ---")
    train_counts = train_df['Count'].astype(int)
    total_train = train_counts.sum()
    weights = total_train / (len(train_counts) * train_counts)
    # Normalize so max is reasonable or mean is 1
    weights = weights / weights.mean()
    
    w_df = pd.DataFrame(weights)
    w_df.columns = ['Weight']
    print(w_df)

if __name__ == "__main__":
    main()
