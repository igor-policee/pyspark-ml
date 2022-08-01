import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType


def process(spark, input_file, target_path):
    df = spark.read.parquet(input_file)

    output_df = \
        df.groupBy(F.col("ad_id").cast(IntegerType()).alias("ad_id")) \
            .agg(
            F.first(F.col("target_audience_count")) \
                .cast(DecimalType()).alias("target_audience_count"),
            F.first(F.col("has_video")) \
                .cast(IntegerType()).alias("has_video"),
            F.first(when(df.ad_cost_type == "CPM", 1).otherwise(0)) \
                .cast(IntegerType()).alias("is_cpm"),
            F.first(when(df.ad_cost_type == "CPC", 1).otherwise(0)) \
                .cast(IntegerType()).alias("is_cpc"),
            F.first(F.col("ad_cost")) \
                .cast(DoubleType()).alias("ad_cost"),
            F.countDistinct(F.col("date")) \
                .cast(IntegerType()).alias("day_count"),
            (F.sum(when(df.event == "click", 1).otherwise(0)) / F.sum(when(df.event == "view", 1).otherwise(0))) \
                .cast(DoubleType()).alias("CTR")
        )

    output_train_split, output_test_split = output_df.randomSplit([0.75, 0.25], 2019)
    output_train_split.write.mode("overwrite").format("parquet").save(os.path.join(target_path, "train"))
    output_test_split.write.mode("overwrite").format("parquet").save(os.path.join(target_path, "test"))


def main(argv):
    input_path = argv[0]
    print(f"Input path to file: {input_path}")
    target_path = argv[1]
    print(f"Target path: {target_path}")
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
