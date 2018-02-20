import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, concat, lit, struct, udf, collect_list, first, explode

HOTELS_CSV_SCHEMA = StructType([
    StructField("_c0", IntegerType(), True),
    StructField("Snapshot ID", IntegerType(), True),
    StructField("Snapshot Date", DateType(), True),
    StructField("Checkin Date", DateType(), True),
    StructField("Days", IntegerType(), True),
    StructField("Original Price", IntegerType(), True),
    StructField("Discount Price", IntegerType(), True),
    StructField("Discount Code", IntegerType(), True),
    StructField("Available Rooms", IntegerType(), True),
    StructField("Hotel Name", StringType(), True),
    StructField("Hotel Stars", IntegerType(), True),
    StructField("DayDiff", IntegerType(), True),
    StructField("Weekday", IntegerType(), True),
    StructField("DiscountDiff", IntegerType(), True),
    StructField("DiscountPerc", DoubleType(), True),
])

NORMALIZATION_UDF_SCHEMA = ArrayType(
    StructType([StructField("Normalized Discount Price", DoubleType(), False),
                StructField("Checkin Date@Code", StringType(), False)])
)


def min_max_discount_normalization_udf(*x):
    row = x[0]
    min_x = min(row, key=lambda x: x['min(Discount Price)'])['min(Discount Price)']
    max_x = max(row, key=lambda x: x['min(Discount Price)'])['min(Discount Price)']

    # For division by 0
    if min_x == max_x:
        return [[100, value['Checkin Date@Code']] for value in row]
    return [[100 / (max_x - min_x) * (value['min(Discount Price)'] - max_x) + 100,
             value['Checkin Date@Code']] for value in row]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hotels Price Data Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to work with", type=str)
    parser.add_argument('--output_path', required=True, help="Where to export the data", type=str)

    args = parser.parse_args(sys.argv[1:])

    spark = SparkSession.builder.appName('HotelsPriceDataGeneratorSession').getOrCreate()
    # Read CSV
    # Note that the schema is already defined, a fully null df will result if the csv does not fit the schema
    print('Reading CSV from ' + args.input_csv)
    hotels_df = spark.read.csv(args.input_csv, header=True, inferSchema=True, schema=HOTELS_CSV_SCHEMA)

    # Collect the top 150 hotels
    print('Adding lazy op - Collecting top 150 hotels')
    top_hotels_df = hotels_df.join(hotels_df.
                                   groupBy('Hotel Name').
                                   count().
                                   sort("count", ascending=False).
                                   limit(150), ['Hotel Name'], 'leftsemi')

    # Collect the top 40 checkin dates
    print('Adding lazy op - Collecting top 40 checkin dates')
    top_hotels_checkin_df = top_hotels_df.join(top_hotels_df.
                                               groupBy('Checkin Date').
                                               count().
                                               sort("count", ascending=False).
                                               limit(40), ['Checkin Date'], 'leftsemi')

    # Create the normalized data frame and the final pivoted dataframe
    # Couple of notes:
    # - The names were not renamed at all, less overhead but longer names
    # - The entire operation is long, due to the length of the data
    # - UDF is perhaps not the best way to perform the min max normalization (consider foreach?)
    print('Adding collection op - Performing minimum discount price normalized as a row DF for checkin date@code')
    price_df = top_hotels_checkin_df \
        .groupBy(['Hotel Name', 'Checkin Date', 'Discount Code']) \
        .min('Discount Price') \
        .withColumn('Checkin Date@Code', concat(col('Checkin Date'), lit('@'), col('Discount Code'))) \
        .groupBy('Hotel Name') \
        .agg(collect_list(struct('min(Discount Price)', "Checkin Date@Code"))) \
        .withColumn("Normalized With Checkin",
                    udf(min_max_discount_normalization_udf, NORMALIZATION_UDF_SCHEMA)(
                        "collect_list(named_struct(NamePlaceholder(), "
                        "min(Discount Price), NamePlaceholder(), Checkin Date@Code))")) \
        .select('Hotel Name', explode('Normalized With Checkin')) \
        .withColumn('Normalized Discount Price', col('col.Normalized Discount Price')) \
        .withColumn('Checkin Date@Code', col('col.Checkin Date@Code')) \
        .groupBy(['Hotel Name']) \
        .pivot('Checkin Date@Code') \
        .agg(first('Normalized Discount Price')) \
        .na.fill(-1)

    # Write the partitioned CSV to the filesystem (No HDFS, so partitioned on the local computer)
    print('Saving partitioned CSV to ' + args.output_path)
    price_df.write.csv(args.output_path)
