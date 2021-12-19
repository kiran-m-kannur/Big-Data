import json
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel,LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def Convert_Json(x):
    l=json.loads(x).values()
    for d in l:
        for key in d:
            d[key]=str(d[key])
    return(l)

sc = SparkContext("local[2]", "BD2_187_228_299_438")
spark=SQLContext(sc)
ssc = StreamingContext(sc, 5)
lines=ssc.socketTextStream("localhost", 6100)

def Convert_Df(time, rdd):
    try:
        if(rdd==[] or rdd==None):
            return
        rdd=rdd.flatMap(lambda x:Convert_Json(x))
        dataframe=spark.createDataFrame(rdd,["Sentiment","Tweet"])
        (train_set,val_set)=dataframe.randomSplit([0.8,0.2],seed=100)
        #pipeline= PipelineModel.load("saved_model")
        pipelineFit = pipeline.fit(train_set)
        #incremental learning
        train_df = pipelineFit.transform(train_set)
        predictions = pipelineFit.transform(val_set)
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        evaluator.evaluate(predictions)
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
        print("{0:.2f}".format(accuracy))
        pipelineFit.write().overwrite().save("saved_model")
        #load
        sameModel=PipelineModel.load("saved_model")
    except Exception as e:
        pass

tokenizer = Tokenizer(inputCol="Tweet", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
svm = LinearSVC(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, svm])
lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
