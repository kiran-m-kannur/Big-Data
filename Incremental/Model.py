import json
import warnings
warnings.filterwarnings("ignore")
import csv
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score, recall_score



def naiveBayes(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape((n, n_x*n_y))
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=45)
	model = MultinomialNB()
	x=np.unique(y_train)
	model.partial_fit(x_train, y_train, x)
	y_pred=model.predict(x_test)
	recall = precision_score(y_test,y_pred,average='binary')
	return(recall)

