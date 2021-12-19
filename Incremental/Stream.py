import json
import csv
import numpy as np
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.sql.functions import length
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from pyspark.sql.functions import length

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

fields = ['NaiveBayesM','NaiveBayesB']

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

def convertToDf(rdd):
	if not rdd.isEmpty():
		obj = rdd.collect()[0]
		load = json.loads(obj)
		df=sql_context.createDataFrame(rdd,["Sentiment","Tweet"])
		processed = preProcess(df)
		acc1 = naiveBayes(processed)
		acc2 = naiveBayesB(processed)
		acc = [acc1,acc2]
		csv_write(acc)

if lines:
	lines.foreachRDD(lambda x: convertToDf(x))

streaming_context.start()
streaming_context.awaitTermination()
streaming_context.stop()

