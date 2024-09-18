# Write actual degree sequences to test the performance of the estimate in those cases

degree_sequences_folder = '../data/real_degrees/degreesequences/'

output_folder = '../data/real_degrees/margins'

from os import listdir
import pandas as pd

input_files = listdir(degree_sequences_folder)

for file in input_files:
    sequence_df = pd.read_csv(f"{degree_sequences_folder}/{file}")

    degree_sequence = []
    for index, row in sequence_df.iterrows():
        if row['xvalue'] != 0:
            for i in range(row['counts']):
                degree_sequence.append(row['xvalue'])

    outfile = f"{output_folder}/{file}"
    with open(outfile, 'w') as f:
        for item in degree_sequence:
            f.write("%s " % item)
    