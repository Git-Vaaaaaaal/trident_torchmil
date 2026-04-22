import datasets
import pandas as pd

# Download labels as csv
datasets.load_dataset(
    'MahmoodLab/Patho-Bench', 
    cache_dir='./tutorial-3',
    dataset_to_download='cptac_ccrcc',     
    task_in_dataset='BAP1_mutation',           
    trust_remote_code=True
)

# Visualize my labels and splits
df = pd.read_csv('tutorial-3/cptac_ccrcc/BAP1_mutation/k=all.tsv', sep="\t")
df
# Check the label distribution
df_counts = df['BAP1_mutation'].value_counts().reset_index()
df_counts.columns = ['BAP1_mutation', 'Count']
df_counts