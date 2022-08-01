import sys
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Путь для загрузки модели
MODEL_PATH = "./spark_ml_prod_model"


def process(spark, input_file, output_file):
    # input_file - путь к файлу с данными для которых нужно предсказать ctr
    # output_file - путь к файлу с результатами [ads_id, prediction]

    # Загрузка модели
    production_model = PipelineModel.load(MODEL_PATH)

    # Загрузка данных
    file_type = "parquet"
    columns_list = [F.col("ad_id").cast("integer"), F.col("target_audience_count").cast("double"),
                    F.col("has_video").cast("double"), F.col("is_cpm").cast("double"),
                    F.col("is_cpc").cast("double"), F.col("ad_cost").cast("double"),
                    F.col("day_count").cast("double"), F.col("ctr").cast("double")]

    test_df = spark.read.format(file_type).load(input_file).select(columns_list)

    # Предсказания модели
    predictions = production_model.transform(test_df)
    result = predictions.select(F.col("ad_id").alias("ad_id"),
                                F.col("prediction").alias("ctr_prediction"))

    # Сохранение результатов [ads_id, prediction]
    result.write.mode("overwrite").format(file_type).save(output_file)


def main(argv):
    input_path = argv[0]
    print(f"Input path to file: {input_path}")
    output_file = argv[1]
    print(f"Output path to file: {output_file}")
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
