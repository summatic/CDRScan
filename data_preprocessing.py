import sys
from random import shuffle
from collections import defaultdict

import pandas as pd
import numpy as np

prefix = '/home/hanseok/sandbox/cdrscan/CDRscan_data/'

# Drug Vectors
binary_drug = pd.read_csv(prefix + 'binary_drug_221_3072.csv')
drug_vecs = {}
for _, row in binary_drug.iterrows():
    pubchem_id = row[0]
    vec = row[1:].tolist()
    drug_vecs[int(pubchem_id)] = vec
    if len(vec) != 3072:
        print(pubchem_id)
print(len(drug_vecs))

# Cellline Vectors
binary_cell_line = pd.read_csv(prefix + 'binary_cellline_784_28087.csv', index_col='Mutation position')
binary_cell_line = binary_cell_line.T
cell_line_vecs = {}
for idx, row in binary_cell_line.iterrows():
    vec = row.tolist()
    cell_line_vecs[idx] = vec
    if len(vec) != 28087:
        print(idx)

# Drug name - PubchemID
drug_data = pd.read_csv(prefix + '1_Drugdata_265.csv')
drug_ids = {}
for _, row in drug_data.iterrows():
    try:
        drug_ids[row[0]] = int(row[1])
    except ValueError:
        continue

# Cellline - Cancer Types
cell_line_data = pd.read_csv(prefix + '1_CellLines_787.csv')
cancer_types = {}
for _, row in cell_line_data.iterrows():
    cell_line_name = row['Sample Name']
    cancer_type = row['Cancer Type\n(matching TCGA label)']
    cancer_types[cell_line_name] = cancer_type

# IC50
responses = pd.read_csv(prefix + 'IC50_nonfiltered.csv')
datasets = {}
ln_ic50s = {}
cancer_type_per_vec = defaultdict(lambda: list())
count = 0
for n, (_, row) in enumerate(responses.iterrows()):
    if n % 5000 == 0:
        sys.stdout.write('\r %d / %d | %d %d %d' % (n, len(responses), len(datasets), len(ln_ic50s), count))
    cell_line_name = row['CellNAME']
    pubchem_id = row['PubChemID']
    ln_ic50 = float(row['LN_IC50'])

    try:
        cell_line_vec = cell_line_vecs[cell_line_name]
        drug_vec = drug_vecs[int(pubchem_id)]
    except Exception:
        count += 1
        continue

    k = '%s_%s' % (cell_line_name, pubchem_id)
    datasets[k] = cell_line_vec + drug_vec
    ln_ic50s[k] = ln_ic50
    cancer_type_per_vec[cancer_types[cell_line_name]].append(k)

# Split train / test
train_X = []
train_y = []
test_X = []
test_y = []
for _, cell_lines in cancer_type_per_vec.items():
    n_train = int(len(cell_lines) * 0.95)
    n_test = len(cell_lines) - n_train

    shuffle(cell_lines)

    for k in cell_lines[:n_train]:
        train_X.append(datasets[k])
        train_y.append(ln_ic50s[k])

    for k in cell_lines[n_train:]:
        test_X.append(datasets[k])
        test_y.append(ln_ic50s[k])

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

np.save(prefix + 'train_X', train_X)
np.save(prefix + 'train_y', train_y)
np.save(prefix + 'test_X', test_X)
np.save(prefix + 'test_y', test_y)
