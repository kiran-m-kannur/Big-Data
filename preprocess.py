import json
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

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
        dataframe=spark.createDataFrame(rdd,["Sentiment","Tweet"])
        #dataframe.show(10)
        pipelineFit = pipeline.fit(dataframe)
        train_df = pipelineFit.transform(dataframe)
        val_df = pipelineFit.transform(dataframe)
        train_df.show(5)

    except:
        print("No Data")

tokenizer = Tokenizer(inputCol="Tweet", outputCol="tweet")
hashtf = HashingTF(numFeatures=2**16, inputCol="tweet", outputCol='tw')
idf = IDF(inputCol='tw', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
