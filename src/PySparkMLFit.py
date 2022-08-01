import sys
from numpy import inf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Path to save the best model
MODEL_PATH = "./spark_ml_prod_model"


def process(spark, train_data, test_data):
    # train_data - the path to the data file for training the model
    # test_data - the path to the data file for evaluating the quality of the model

    # Upload data to dataframe
    file_type = "parquet"
    columns_list = [F.col("ad_id").cast("integer"), F.col("target_audience_count").cast("double"),
                    F.col("has_video").cast("double"), F.col("is_cpm").cast("double"),
                    F.col("is_cpc").cast("double"), F.col("ad_cost").cast("double"),
                    F.col("day_count").cast("double"), F.col("ctr").cast("double")]

    train_df = spark.read.format(file_type).load(train_data).select(columns_list)
    test_df = spark.read.format(file_type).load(test_data).select(columns_list)

    # Feature engineering
    categorical_columns = ["has_video", "is_cpm", "is_cpc"]
    num_columns = ["target_audience_count", "ad_cost", "day_count"]

    stages = []
    assembler_input_features = list(categorical_columns) + num_columns
    assembler_features = VectorAssembler(inputCols=assembler_input_features, outputCol="features")
    stages += [assembler_features]

    pre_pipeline = Pipeline(stages=[assembler_features])
    pre_pipeline_model = pre_pipeline.fit(train_df)

    # Initializing models
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ctr", metricName="rmse")
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="ctr")
    rf = RandomForestRegressor(featuresCol="features", labelCol="ctr")
    gbt = GBTRegressor(featuresCol="features", labelCol="ctr")

    models = {
        "model_dt": {"raw_model": dt, "best_model": None, "rmse": None},
        "model_rf": {"raw_model": rf, "best_model": None, "rmse": None},
        "model_gbt": {"raw_model": gbt, "best_model": None, "rmse": None}
    }

    best_rmse = inf
    production_model = None

    # Hyperparameter selection, model training, saving the best model
    for model in models:
        train_pipeline = Pipeline(stages=[pre_pipeline_model, models[model]["raw_model"]])
        param_grid = ParamGridBuilder().addGrid(models[model]["raw_model"].maxDepth, list(range(2, 10+1))).build()
        cv = CrossValidator(estimator=train_pipeline, estimatorParamMaps=param_grid,
                            evaluator=evaluator, parallelism=2, numFolds=2)
        best_model = cv.fit(train_df).bestModel
        models[model]["best_model"] = best_model
        predictions = best_model.transform(test_df)
        rmse = evaluator.evaluate(predictions)
        models[model]["rmse"] = rmse
        print(f"Model = {model}\tRMSE = {rmse}")
        if rmse < best_rmse:
            best_rmse = rmse
            production_model = best_model

    # Saving the best model
    production_model.write().overwrite().save(MODEL_PATH)


def main(argv):
    train_data = argv[0]
    print(f"Input path to train data: {train_data}")
    test_data = argv[1]
    print(f"Input path to test data: {test_data}")
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
