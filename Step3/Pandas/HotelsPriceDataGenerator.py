import sys
import argparse
import pandas as pd
import numpy as np


def min_max_discount_normalization(x):
    max_x = x.max()
    min_x = x[x != -1].min()
    x = x.apply(lambda v: -1 if np.isnan(v) else
                (100-0)/(max_x-min_x)*(v-max_x)+100)

    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hotels Price Data Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to work with", type=str)
    parser.add_argument('--output_path', required=True, help="Where to export the csv", type=str)

    args = parser.parse_args(sys.argv[1:])

    # Read CSV
    hotels_df = pd.read_csv(args.input_csv, encoding="ISO-8859-1")

    # First col index
    hotels_df.set_index('Unnamed: 0', inplace=True)

    # Collect the top 150 hotels
    hotel_names_df = hotels_df.groupby('Hotel Name')['Hotel Name'].count().reset_index(name='count'). \
        sort_values(['count'], ascending=False).head(150)
    hotel_names_df = pd.merge(hotel_names_df[['Hotel Name']], hotels_df, on='Hotel Name', how='left')

    # Collect the top 40 checkin dates
    hotel_checkin_date_df = hotel_names_df.groupby('Checkin Date')['Checkin Date'].count().reset_index(
        name='count_checkin'). \
        sort_values(['count_checkin'], ascending=False).head(40)
    hotel_checkin_date_df = pd.merge(hotel_checkin_date_df[['Checkin Date']], hotel_names_df, on='Checkin Date',
                                     how='left')

    # Group by and create the column to use for the pivot
    hotels_groups = hotel_checkin_date_df.groupby(['Hotel Name', 'Checkin Date', 'Discount Code'])[
        'Discount Price'].min()
    hotels_groups = pd.DataFrame(hotels_groups)
    hotels_groups.reset_index(inplace=True)

    hotels_groups['Checkin Date@Code'] = hotels_groups['Checkin Date'].astype(str) + '@' + hotels_groups[
        'Discount Code'].astype(str)

    hotels_pivoted = hotels_groups.pivot(index='Hotel Name', columns='Checkin Date@Code', values='Discount Price')

    # Apply min max norm on the rows
    hotels_pivoted.columns.name = None
    hotels_pivoted = hotels_pivoted.apply(min_max_discount_normalization, axis=1).fillna(-1)
    hotels_pivoted.reset_index(inplace=True)

    # Save the CSV
    hotels_pivoted.to_csv(args.output_path, index=False)
