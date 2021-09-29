In the following codealong, we will combine our new NLP knowledge with our knowledge of pipelines. We will apply this combination of skills to a common task: effectively separate `spam` from `ham` in a set of messages. 


```python
# Import not necessary for students
import sys
sys.path.append('../..')

from new_caller.random_student_engager.student_caller import CohortCaller
from new_caller.random_student_engager.student_list import avocoder_toasters

caller = CohortCaller(avocoder_toasters)
```

    hello


The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). 


```python
# Run cell with no changes to import Ham vs. Spam SMS dataset
import pandas as pd

with open('data/SMSSpamCollection') as read_file:
    texts = read_file.readlines()
    
text = [text.split('\t')[1] for text in texts]
label = [text.split('\t')[0] for text in texts]
df = pd.DataFrame(text, columns=['text'])
df['label'] = label
df['label'] = df['label']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...\n</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>



As the head method shows, our data is labeled either ham or spam.

Check the distribution of the target in the cell below.


```python
# Use pandas to find the distribution of Spam to Ham in the dataset

```


```python
#__SOLUTION__
df['label'].value_counts()
```




    ham     4827
    spam     747
    Name: label, dtype: int64




```python
caller.call_n_students(1)
```




    array(['Seth'], dtype='<U7')



Certain metrics require that our target be in the form of 0's and 1's. Use the LabelEncoder method to transform the target.  


```python
# f1 metric requires 0,1 labels
# Which should be 0 and which should be 1
from sklearn.preprocessing import LabelEncoder

```


```python
#__SOLUTION__
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
```


```python
caller.call_n_students(1)
```




    array(['Rashid'], dtype='<U7')



# Target Distribution and Train-Test Split

The model building workflow is similar to what we have performed in Phase 3.  

To begin, train-test split the data set.  Preserve the class balance in the test set.


```python
# train-test split the dataset while preserving the class balance show above
# Pass random_state=42 as an argument as well
from sklearn.model_selection import train_test_split


```


```python
#__SOLUTION__
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['text']], df['label'], 
                                                    random_state=42, 
                                                   stratify=df['label'])
```


```python
caller.call_n_students(1)
```




    array(['Meaghan'], dtype='<U7')



# Count Vectorizor and TFIDF Vectorizer

In a small group, take 10 minutes to move through one model building iteration.  What can that look like? Through some steps you decide on as a group, fit a vectorizer and a model on a training set(s), transform the "test" set, and score on it. 

Two points to take into careful consideration:
    
    1. What metric is appropriate in this case? Or, to put it another way, is one error more costly when creating a spam detector?
    2. Will you use cross-validation/pipelines?
    3. What vectorizer and model will you use? 

Whatever you decide, start with a simple document-term matrix. Start with a max_features of 50.  Go ahead and feed arguments to the vectorizer to take out stopwords. Use default params for the rest.

After you are finished, generate a confusion matrix of your "test" predictions. If you are using cross_validate, use cross_validate_predict along with sklearn's confusion_matrix to create it.


```python
# your code here
```


```python
#__SOLUTION__
# pass the pipeline into sklearn's cross validate function.  
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords


f1_precision = make_scorer(fbeta_score, beta=.5)

pipe = make_pipeline(TfidfVectorizer(max_features=25, stop_words=stopwords.words('english')), MultinomialNB())

# your code here: return the train score so we can look at the bias variance tradeoff
cv = cross_validate(pipe, X_train['text'], y_train, return_train_score=True,
                    scoring=f1_precision)
cv
```




    {'fit_time': array([0.04860902, 0.04546094, 0.04029584, 0.0393002 , 0.03818822]),
     'score_time': array([0.00984597, 0.008919  , 0.00850511, 0.00795603, 0.00825381]),
     'test_score': array([0.64516129, 0.61688312, 0.60483871, 0.66901408, 0.64236111]),
     'train_score': array([0.6294964 , 0.65498155, 0.65159574, 0.63858696, 0.64545455])}




```python
#__SOLUTION__

# Pass the pipeline, as well as X_train['text'] and y_train to cross_val_predict
from sklearn.model_selection import cross_val_predict

y_hat_train = cross_val_predict(pipe, X_train['text'], y_train)
```


```python
#__SOLUTION__
# Create a confusion matrix with the results of cross_val_predict
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_hat_train)
```




    array([[3591,   29],
           [ 385,  175]])



# Iterate

For the next 15 minutes, improve you model.  

Discuss with your group steps you can take to improve your "test" score.

What you should consider: 
    
    1. What hyperparameters can you tune on your vectorizer?
    2. How should you tune those hyperparameters? 
    3. What other preprocessing steps, transformers, and estimators should you try?
    4. Once you achieve a satisfying score, can you simplify the term matrix and achieve similar performance?


```python
# Your code here
```


```python
#__SOLUTION__
from sklearn.model_selection import GridSearchCV
# Define new pipeline with default parameters for tfidf and multinomialNB
parameter_dict = {'tfidfvectorizer__max_features':[25,50,100,1000]}

new_pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())

gs = GridSearchCV(new_pipe, parameter_dict, scoring=f1_precision)
gs.fit(X_train['text'], y_train)
```




    GridSearchCV(estimator=Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer()),
                                           ('multinomialnb', MultinomialNB())]),
                 param_grid={'tfidfvectorizer__max_features': [25, 50, 100, 1000]},
                 scoring=make_scorer(fbeta_score, beta=0.5))




```python
#__SOLUTION__
parameter_dict = {'tfidfvectorizer__max_features': [100,500, 1000, 2000, 3000, 4000], 
                 'tfidfvectorizer__stop_words': [None, stopwords.words('english')]}

pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())

def gs_tfidf(parameter_dict, pipe, verbose=True):

    gs = GridSearchCV(pipe, parameter_dict, scoring=f1_precision, verbose=verbose)
    gs.fit(X_train['text'], y_train)
    print(gs.best_score_)

    
    y_hat_train = cross_val_predict(gs.best_estimator_, X_train['text'], y_train)
    
    print(confusion_matrix(y_train, y_hat_train))
    
    print(gs.best_params_)
    
```


```python
#__SOLUTION__
gs_tfidf(parameter_dict, pipe)
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:    3.2s finished


    0.9632032250411017
    [[3619    1]
     [  86  474]]
    {'tfidfvectorizer__max_features': 2000, 'tfidfvectorizer__stop_words': ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]}



```python
#__SOLUTION__
parameter_dict = {'tfidfvectorizer__max_features': [None, 500,1000, 1500, 2000], 
                 'tfidfvectorizer__stop_words': [None, stopwords.words('english')],
                  'tfidfvectorizer__max_df': [1.0, .9, .8], 
                 'tfidfvectorizer__min_df': [1, 5]}


gs_tfidf(parameter_dict, pipe)
```

    Fitting 5 folds for each of 60 candidates, totalling 300 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 300 out of 300 | elapsed:   15.4s finished


    0.9632032250411017
    [[3619    1]
     [  86  474]]
    {'tfidfvectorizer__max_df': 1.0, 'tfidfvectorizer__max_features': 2000, 'tfidfvectorizer__min_df': 1, 'tfidfvectorizer__stop_words': ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]}



```python
#__SOLUTION__
from sklearn.ensemble import RandomForestClassifier
parameter_dict = {'tfidfvectorizer__max_features': [1500, 2000],
                  'tfidfvectorizer__max_df': [1.0,.9, .8], 
                 'tfidfvectorizer__min_df': [1, 5, 10], }

pipe = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('english')), RandomForestClassifier())
gs_tfidf(parameter_dict, pipe)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  90 out of  90 | elapsed:   40.4s finished


    0.964416779387471
    [[3614    6]
     [  69  491]]
    {'tfidfvectorizer__max_df': 0.9, 'tfidfvectorizer__max_features': 1500, 'tfidfvectorizer__min_df': 5}



```python

```
