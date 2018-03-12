import argparse
import sys
import pandas as pd
import random

FEATURES_LIST = ['Hotel Name', 'Weekday', 'Snapshot Date', 'Checkin Date', 'DayDiff']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotels CSV Prediction Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to use for shuffle predict", type=str)
    parser.add_argument('--generated_count', default=100, help='How much rows to generate')
    parser.add_argument('--output_csv', required=True, help='Output CSV path')

    args = parser.parse_args(sys.argv[1:])

    hotels_csv = pd.read_csv(args.input_csv, encoding="ISO-8859-1")[FEATURES_LIST]

    # Randomly Generate new rows from selecting random features
    # and discount code from the possibilites in the existing DF
    # This is only used to test the model predictor
    # Go K times and over every column and randomly select a row and extract the value from it
    df_rows = []
    for g in range(args.generated_count):
        df_row = []
        for col in hotels_csv.columns:
            # Generate a random index
            idx = random.randint(0, len(hotels_csv))
            df_row.append(hotels_csv.loc[idx, col])
        df_rows.append(df_row)
    df = pd.DataFrame(df_rows, columns=hotels_csv.columns)
    df.to_csv(args.output_csv)
