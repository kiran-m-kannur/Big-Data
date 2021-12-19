import json
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

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
        dataframe.show(10)

    except:
        print("No Data")

lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
