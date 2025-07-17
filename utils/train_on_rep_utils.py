import pandas as pd

def select_n_patches(train_df, n_patches=10):
    if 'black_ratio' in train_df.columns:
        grouped_sorted = train_df.groupby('page', group_keys=False).apply(lambda x: x.sort_values('black_ratio', ascending=False))
    else:
        grouped_sorted = train_df.groupby('page', group_keys=False).apply(lambda x: x)
    grouped_sorted = grouped_sorted.groupby('page', group_keys=False).head(n_patches)
    return grouped_sorted