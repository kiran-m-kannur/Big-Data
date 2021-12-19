import json
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score

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
        (training_data,test_data)=dataframe.randomSplit([0.7,0.3],seed=100)
        pipelineFit = pipeline.fit(training_data)      
        tp=pipelineFit.transform(test_data)
        tp=tp.select('label','prediction')     
        tnp=np.array((tp.collect()))
        print(accuracy_score(tnp[:,0],tnp[:,1]) * 100)
    except Exception as e:
        print("No Data",e)

tokenizer = Tokenizer(inputCol="Tweet", outputCol="tweet")
hashtf = HashingTF(numFeatures=2**16, inputCol="tweet", outputCol='tw')
idf = IDF(inputCol='tw', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
lr = LogisticRegression(featuresCol="features",labelCol="label",maxIter=100)
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx, lr])
lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
