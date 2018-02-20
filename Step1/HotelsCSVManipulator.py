import sys
import pandas as pd
import argparse


def convert_df_datetime(hotels_df, col_name):
    hotels_df[col_name] = pd.to_datetime(hotels_df[col_name], format="%m/%d/%Y %H:%M")


def add_day_diff(hotels_df):
    hotels_df['DayDiff'] = (hotels_df['Checkin Date'] - hotels_df['Snapshot Date']).astype('timedelta64[D]').astype('int')


def add_week_day(hotels_df):
    # Can be also
    # df['Weekday'] = df['Checkin Date'].dt.weekday_name
    # But easier to work with numbers for later on
    hotels_df['Weekday'] = hotels_df['Checkin Date'].dt.dayofweek


def add_discount_diff(hotels_df):
    hotels_df['DiscountDiff'] = hotels_df['Original Price'] - hotels_df['Discount Price']


def add_discount_percentage(hotels_df):
    hotels_df['DiscountPerc'] = hotels_df['DiscountDiff'] / hotels_df['Original Price']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotels CSV Manipulator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to manipulate", type=str)
    parser.add_argument('--output_csv', required=True, help="Output CSV path", type=str)

    args = parser.parse_args(sys.argv[1:])

    try:
        hotels_raw_df = pd.read_csv(args.input_csv)
        if not hotels_raw_df.empty:
            print("Read CSV, adding features")
            print("Time will be converted to DF timestamp for convenience later on")
            convert_df_datetime(hotels_raw_df, 'Snapshot Date')
            convert_df_datetime(hotels_raw_df, 'Checkin Date')

            add_day_diff(hotels_raw_df)
            add_week_day(hotels_raw_df)
            add_discount_diff(hotels_raw_df)
            add_discount_percentage(hotels_raw_df)

            print("Saving CSV...")
            hotels_raw_df.to_csv(args.output_csv)

            print("Done")
    except Exception as e:
        print(str(e))
