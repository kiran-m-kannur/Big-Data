import json
import numpy as np
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.linalg import Vectors 
from sklearn.feature_extraction.text import HashingVectorizer

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
run=0;


def Convert_Df(time, rdd):
    print("......................")
    try:
        if(rdd==[] or rdd==None):
            return
        rdd=rdd.flatMap(lambda x:Convert_Json(x))
        dataframe=spark.createDataFrame(rdd,["Sentiment","Tweet"])
        global run
        global sgd
        (train_set,val_set)=dataframe.randomSplit([0.8,0.2],seed=100)
        #pipeline= PipelineModel.load("saved_model")
        pipelineFit = pipeline.fit(train_set)
        #incremental learning
        train_df = pipelineFit.transform(train_set)
        feature_rdd=train_df.select('features','label').rdd
        labels=feature_rdd.map(lambda row: row.label)
        dense_vector_rdd=feature_rdd.map(lambda row: Vectors.dense(row.features))
        x_train=[]
        y_train=[]
        for x in dense_vector_rdd.collect():
        	x_train.append(x.toArray().tolist())
        x_train=np.array(x_train) 
        x_train=vectorizer.transform(x_train)
        for y in labels.collect():
        	y_train.append(y)
        y_train=np.array(y_train) 
        '''for i in train_df.select("features").collect():
        	x_train.append(list(i))
        for j in train_df.select("label").collect():
        	y_train.append(j)
        print(x_train)
        x_train=[]
        y_train=[]
        x_array = np.array(train_df.select("features").collect())
        for x in x_array:
        	x_train.append(x[0])
        #print(x_train)
        y_array = np.array(train_df.select("label").collect())
        for y in y_array:
        	y_train.append(y[0])
        #print(x_train)
        xx,xy,xz = x_train.shape
        yx,yy = y_train.shape
        print(x_train.shape)
        print(y_train.shape)
        x_train = x_train.reshape(xx, xy*xz)
        y_train = y_train.reshape(yx*yy)'''
        #print(y_train)
        print("TRAINED")        
        if(run==0):
            print("=====SGD Model Created=====")
            sgd = linear_model.SGDClassifier()
        else:
            predictions = sgd.predict(x_train)
            accuracy = accuracy_score(y_train,predictions)
            print("Accuracy: ",accuracy)
        sgd.partial_fit(x_train,y_train,classes=np.unique([0,1]))
        print("===FITTED===")
        run=run+1
        '''predictions = pipelineFit.transform(val_set)
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        evaluator.evaluate(predictions)
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
        roc_auc = evaluator.evaluate(predictions)
        print("Accuracy Score: {0:.4f}".format(accuracy))
        print("ROC-AUC: {0:.4f}".format(roc_auc))'''
    except Exception as e:
        print(e)

tokenizer = Tokenizer(inputCol="Tweet", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
#sgd = linear_model.SGDClassifier(loss='log',max_iter=100)
vectorizer=HashingVectorizer(decode_error='ignore',n_features=2**18)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx])
lines.foreachRDD(Convert_Df)
ssc.start()
ssc.awaitTermination()
