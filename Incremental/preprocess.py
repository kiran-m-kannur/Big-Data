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

def preProcess(df):
	df=df.withColumn('len', length(df['Tweet']))
	sentiment_indexer = StringIndexer(inputCol='Sentiment', outputCol='sentiment')
	token = Tokenizer(inputCol='Tweet', outputCol='tweet')
	remove_stop = StopWordsRemover(inputCol='textToken', outputCol='token_stop')
	vector_count = CountVectorizer(inputCol='token_stop', outputCol='vector')
	idf = IDF(inputCol='vector', outputCol='tf')
	clean = VectorAssembler(inputCols=['tf', 'len'], outputCol='tw')
	pipeLine = Pipeline(stages=[spam_indexer, token, remove_stop, vector_count, idf, clean])
	data_new = (pipeLine.fit(df)).transform(df)
	return data_new