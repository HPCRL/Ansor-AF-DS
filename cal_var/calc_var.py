import os
import pandas as pd
import numpy as np
import argparse
from scipy import stats


# Create an argument parser
parser = argparse.ArgumentParser(description='Calculate LOP for files.')

# Define the expected arguments
parser.add_argument('data', type=str, help='Path for the data folder.')
parser.add_argument('cmpr_point', type=str, help='two or end')
#parser.add_argument('out', type=str, help='path for generating output log and CSVs.')
#parser.add_argument('--arg3', type=float, default=1.0, help='Description of arg3 (optional, default=1.0)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments using args
datapath = args.data
cmpr_point = args.cmpr_point
#output_path = args.out

# functions
def calc_var(df, cmpr_point):
    if cmpr_point == "end":
        df['variability'] = (df['best_max'] - df['best_min']) / df['best_max']
    elif cmpr_point == "two":
        df['variability'] = (df['2_max'] - df['2_min']) / df['2_max']
    else:
        print("The 'cmpr_point' argument is neigher 'two' nor 'end'")
        exit(-1)


# the main
csv_files = [file for file in os.listdir(datapath) if file.endswith('.csv')]

#w_var_folder = output_path
w_var_folder = os.path.join(datapath, "w_var_folder")
os.makedirs(w_var_folder, exist_ok=False)

for file in csv_files:
    df = pd.read_csv(os.path.join(datapath, file))
    calc_var(df, cmpr_point)
    new_file_path = os.path.join(w_var_folder, file.replace('.csv', '_with_var.csv'))
    df.to_csv(new_file_path)
    #overall_avg_variability = df['variability'].mean()
    overall_var_geomean = stats.gmean(df['variability'])
    print(f"Overall Variability Geomean for {file}: {overall_var_geomean}")
    output_txt_path = os.path.join(w_var_folder, 'overall_variability_geomean.txt')
    with open(output_txt_path, 'a') as txt_file:
        txt_file.write(f"Overall Variability Geomean for {file}: {overall_var_geomean}\n")


