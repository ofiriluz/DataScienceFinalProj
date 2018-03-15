Step3
=====

Step 3 consists of clustering the data from step 1

With addition to different data manipulation to create a new feature set.

This part is implemented using both pyspark and pandas
Both sides support the same args and the same executables

The step consists of two executables
- HotelsPriceDataGenerator
    - This executable is used to create the new feature set according to the instructions
    - It uses pyspark dataframes manipulations to achieve that.
    - Usage:
      ```
      python HotelsPriceDataGenerator.py < ARGS >
      ```
    - Args:
        - --input_csv < PATH >
        - --output_csv < PATH >
- HotelsPriceClustering
    - This executable is used to perform the actual clustering on the new data
    - It uses two methods to show the final dendrogram
        - PDist on the rows
        - Bisecting KMeans to perform double hierarchy clustering
    - The final output is the plotted dendrogram of the clustering
    - Note that the input csv is the csv from the data generator
    - Usage:
      ```
      python HotelsPriceClustering.py < ARGS > < MODE >
      ```
    - Args:
        - --input_csv < PATH >
        - mode [KMeans, PDist]
        - KMeans
            - --num_clusters <SIZE>
    - Example:
      ```
      python HotelsPriceClustering.py --input_csv x.csv PDist
      ```