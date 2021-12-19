import json
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

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
    print("......................")
    try:
        if(rdd==[] or rdd==None):
            return
        rdd=rdd.flatMap(lambda x:Convert_Json(x))
        dataset=spark.createDataFrame(rdd,["Sentiment","Tweet"])
        model = pipeline.fit(dataset)
# Make predictions
        predictions = model.transform(dataset)
# Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator()
# Shows the result.
        silhouette = evaluator.evaluate(predictions)
        print(str(silhouette))

    except Exception as e:
        print("No Data",e)

bkm = BisectingKMeans().setK(2).setSeed(1)
tokenizer = Tokenizer(inputCol="Tweet", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, bkm])
lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
