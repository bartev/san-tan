# import necessary libraries

# since using python, import SparkSession and related functions from the PySpark module

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: mnmcount <file>', file=sys.stderr)
        sys.exit(-1)

    # build a SparkSession using the SparkSessioin APIs
    # if one does not exist, then create an instance
    # There can only be one SparkSession per JVM

    spark = (SparkSession
                .builder
                .appName('PythonMnMCount')
                .getOrCreate())

    # get the M&M data set file name from the command line args
    mnm_file = sys.argv[1]

    # read the file into a Spark DataFrame using the CSV
    # format by inferring the schema and informing that the file
    # contains a header, which are column names for comma separated fields

    mnm_df = (spark.read.format('csv')
                .option('header', 'true')
                .option('inferSchema', 'true')
                .load(mnm_file))

    # We use the Structured DataFrame high-level APIs. Note
    # that we don't use RDDs at all. Because some Spark functions
    # return the same object, we can chain function calls
    # 1. Select from the DF fields 'State', 'Color', and 'Count'
    # 2. Since we want them to group each state and its M&M color
    # count, we use groupBy()
    # 3. Aggregate count of all colors and groupBy state and color
    # 4. orderBy descending order

    count_mnm_df = (mnm_df.select('State', 'Color', 'Count')
                    .groupBy('State', 'Color')
                    .agg(count('Count')
                        .alias('Total'))
                    .orderBy('Total', ascending=False))
    
    # show all the resulting aggregation for all the states, colors
    # a total count of each color per state
    # Note show() is an action, which will trigger the above
    # query to be executed.

    count_mnm_df.show(n=60, truncate=False)
    print(f'Total Rows = {count_mnm_df.count()}')

    # While the above code aggregated and counted for all 
    # the states, what if we just
    # a single state, e.g., CA. 
    # 1. Select from all rows in the DataFrame
    # 2. Filter only CA state
    # 3. GroupBy() State and Color as we did above
    # 4. Aggregate its count for each color
    # 5. OrderBy() by Total in a descending order 
    # find the aggregate count for California by filtering

    ca_count_mnm_df = (mnm_df.select('State', 'Color', 'Count')
                        .where(mnm_df.State == 'CA')
                        .groupBy('State', 'Color')
                        .agg(count('Count').alias('Total'))
                        .orderBy('Total', ascending=False))


    # show the resulting aggregation for California
    # As above, show() is an action that will trigger the execution
    # of the entire computation.

    ca_count_mnm_df.show(n=10, truncate=False)
    # stop the SparkSession
    spark.stop()