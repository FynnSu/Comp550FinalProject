import pandas as pd
import os.path as osp
import os

def main(args):
    # Read in all csvs from input_dir into a dict of dataframes
    dataframes = {}
    for csv_file in sorted(os.listdir(args.input_dir)):
        dataframes[csv_file] = pd.read_csv(osp.join(args.input_dir, csv_file), lineterminator='\n')
    
    # For each dataframe, print the number of rows, keep track of total, max, min, and average
    total_rows = 0
    max_rows = 0
    min_rows = 1000000000
    for csv_file in sorted(dataframes.keys()):
        rows = dataframes[csv_file].shape[0]
        total_rows += rows
        if rows > max_rows:
            max_rows = rows
        if rows < min_rows:
            min_rows = rows
        print("{} has {} rows".format(csv_file, rows))
    
    # Print total, max, min, and average
    print("Total rows: {}".format(total_rows))
    print("Max rows: {}".format(max_rows))
    print("Min rows: {}".format(min_rows))
    print("Average rows: {}".format(total_rows / len(dataframes)))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)