import os
import pandas as pd
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extends_with_info_from_source(df_input,df_pre_patch,columns=['unique name', 'type']):
    # For each row in df, map 'original raw file' to 'experiment' in df_pre_patch and extract 'unique name' and 'type'
    df = df_input.copy()
    df.drop(columns=['experiment'], inplace=True, errors='ignore')
    l=len(df)
    cols= ['experiment']+ columns
    df = df.merge(
        df_pre_patch[cols],
        left_on='original raw file',
        right_on='experiment',
        how='left'
    )
    assert len(df) == l, f"Length changed after merge: before={l}, after={len(df)}"
    return df

def get_crossval_results(df_source, types=['individual','ensembled_weighted'],groups=['arabic,all','english,all']):
    df= df_source.copy()
    new_columns = {}
    for type in types:
        values_list = []
        for idx, row in df.iterrows():
            c_val = row['cross_val_accuracies']
            OOF_values = c_val['OOF']
            accuracies_OOF = []
            for value in OOF_values:
                accuracies_OOF.append(value[type])
            values_list.append(copy.deepcopy(accuracies_OOF))
        new_columns[f"{type}_accuracies"] = values_list
    for group in groups:
        for type in types:
            values_list = []
            for idx, row in df.iterrows():
                c_val = row['cross_val_subgroup_accuracies']
                group_accuracies=[]
                for fold_accuracies in c_val:
                    group_accuracies.append(fold_accuracies[group][type])
                values_list.append(copy.deepcopy(group_accuracies))
            new_columns[f"{group}_{type}_accuracies"] = values_list
    #print(new_columns)
    acc_df = pd.DataFrame(new_columns)
    df=pd.concat([df.reset_index(drop=True), acc_df.reset_index(drop=True)], axis=1)
    return df

def box_and_whiskers(df, type='squares', accuracy_col='ensembled_weighted_accuracies', 
                     title='Distribution of Individual Accuracies per FE Model',y_title='Accuracy',
                     save_to='C:\\Users\\andre\\VsCode\\PD related projects\\gender_detection\\outputs\\plots',save_title='fe_model_accuracy_boxplot'):
    # 1) Long/tidy data (explode the list-of-accuracies into one row per value)
    df_plot = (
        df.loc[df['type'] == type, ['FE model', accuracy_col]]
        .explode(accuracy_col)
        .rename(columns={accuracy_col: 'accuracy'})
        .dropna()
    )
    df_plot['accuracy'] = df_plot['accuracy'].astype(float)

    # 2) Order models by median accuracy (robust)
    order = (df_plot.groupby('FE model')['accuracy']
                    .median()
                    .sort_values(ascending=False)
                    .index)

    # 3) Aesthetics suitable for papers
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(
        data=df_plot, x='FE model', y='accuracy', order=order,
        whis=1.5, showmeans=True,
        meanprops=dict(marker='^', markersize=6,
                    markerfacecolor='white', markeredgecolor='black')
    )
    # Overlay raw points (helps when n is small)
    sns.stripplot(
        data=df_plot, x='FE model', y='accuracy', order=order,
        jitter=0.2, size=2.5, alpha=0.5, color='black'
    )

    ax.set_xlabel('')
    ax.set_ylabel(y_title)
    ax.set_title(title)
    plt.xticks(rotation=40, ha='right')

    # Optional: show sample sizes under each tick
    counts = df_plot.groupby('FE model')['accuracy'].size().reindex(order)
    for i, (label, n) in enumerate(zip(order, counts)):
        ax.text(i, ax.get_ylim()[0], f"n={n}", ha='center', va='bottom',
                fontsize=9, rotation=0, clip_on=False)

    plt.tight_layout()
    plt.savefig(f"{save_to}\\{save_title}.pdf")   # vector for publication
    plt.savefig(f"{save_to}\\{save_title}.png", dpi=300)
    plt.show()

###script generalization###
def script_generalization_compute_diffs(row,metric='individual'): ######### DONE ########
    if row['train_on_language'] == 'arabic':
        diffs=[]
        for acc in zip(row[f'arabic,all_{metric}_accuracies'], row[f'english,all_{metric}_accuracies']):
            diffs.append(acc[0]-acc[1])
    elif row['train_on_language'] == 'english':
        diffs=[]
        for acc in zip(row[f'english,all_{metric}_accuracies'], row[f'arabic,all_{metric}_accuracies']):
            diffs.append(acc[0]-acc[1])
    else:
        diffs=None
    return diffs

def script_generalization_join_acc_lists(df,merge_cols=['FE model','type'],columns=['script_generalization']):
    df_copy=df.copy()
    df_copy = df_copy.groupby(merge_cols, sort=False, as_index=False).agg({col: 'sum' for col in columns})   # list + list => concatenation
    return df_copy