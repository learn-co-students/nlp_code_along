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

```
