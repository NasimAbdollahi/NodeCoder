import pandas as pd

entry_name = 'MUC2_HUMAN'

features_path = f"/home/ssmackin/work/gcn/2021-12/9606/features/{entry_name}.features.csv"
features_df = pd.read_csv(features_path)
print(features_path)
print(features_df.head())
print(features_df.tail())

tasks_path = f"/home/ssmackin/work/gcn/2021-12/9606/tasks/{entry_name}.tasks.csv"
tasks_df = pd.read_csv(tasks_path)
print(tasks_path)
print(tasks_df.head())
print(tasks_df.tail())

combined_df = tasks_df.append(features_df)
print(combined_df.head())
print(combined_df.tail())
