#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import opendatasets as od
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Downloading the data

# In[2]:


dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'


# In[3]:


od.download(dataset_url)


# In[4]:


import os


# In[5]:


data_dir = './weather-dataset-rattle-package'
train_csv = data_dir + './weatherAUS.csv'


# In[6]:


raw_df = pd.read_csv(train_csv)


# In[7]:


raw_df


# In[8]:


raw_df.info()


# In[9]:


raw_df.dropna(subset=['RainToday','RainTomorrow'],inplace = True)


# # Exploratory Data Analysis

# In[10]:


px.histogram(raw_df, x = 'Location', title='Location vs. Rainy Days', color='RainToday')


# In[11]:


px.histogram(raw_df, x='Temp3pm', title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow')


# In[12]:


px.histogram(raw_df, x='RainTomorrow', color='RainToday', title='Rain Tomorrow vs. Rain Today')


# In[13]:


px.scatter(raw_df.sample(2000), title='Min Temp. vs Max Temp.',x='MinTemp',y='MaxTemp', color='RainToday')


# In[14]:


px.scatter(raw_df.sample(2000), title='Temp (3 pm) vs. Humidity (3 pm)',x='Temp3pm',y='Humidity3pm',color='RainTomorrow')


# In[15]:


top_locations = raw_df['Location'].value_counts().head(10).index
filtered_data = raw_df[raw_df['Location'].isin(top_locations)].sample(5000)
plt.figure(figsize=(12,8))
sns.boxplot(data=filtered_data, x='Location', y='MaxTemp')
plt.title('Max Temperature Distribution by Top 10 Locations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Training, Validation and Test Sets

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[18]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[19]:


plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)


# In[20]:


# Create training, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


# In[21]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[22]:


train_df


# # Identifying Input and Target Columns
# 

# In[23]:


# Create inputs and targets
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'


# In[24]:


print(input_cols)
print(target_col)


# In[25]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[26]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()


# In[27]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# In[28]:


train_inputs


# In[29]:


train_targets


# In[30]:


# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[31]:


numeric_cols


# In[32]:


categorical_cols


# In[33]:


train_inputs[numeric_cols].describe()


# In[34]:


train_inputs[categorical_cols].nunique()


# # Imputing Missing Numeric Data

# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')


# In[36]:


raw_df[numeric_cols].isna().sum()


# In[37]:


train_inputs[numeric_cols].isna().sum()


# In[38]:


# Impute missing numerical values
imputer.fit(raw_df[numeric_cols])


# In[39]:


list(imputer.statistics_)


# In[40]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[41]:


train_inputs[numeric_cols].isna().sum()


# In[42]:


raw_df[numeric_cols].describe()


# # Scaling Numeric Features

# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])


# In[44]:


list(scaler.data_min_)


# In[45]:


list(scaler.data_max_)


# In[46]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# # Encoding Categorical Data

# In[47]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, handle_unknown= 'ignore')
encoder.fit(raw_df[categorical_cols])


# In[48]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# # Training a Logistic Regression Model

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix


# In[50]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[51]:


model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)


# In[52]:


train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)


# # Results and Evaluation

# In[53]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted')  
    recall = recall_score(targets, preds, average='weighted')        
    f1 = f1_score(targets, preds, average='weighted')               
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True, fmt=".2f", cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name} Confusion Matrix')
    
    return preds


# In[54]:


train_preds = predict_and_plot(X_train, train_targets, 'Training' )


# In[55]:


val_preds = predict_and_plot(X_val, val_targets, 'Validation' )


# In[56]:


test_preds = predict_and_plot(X_test, test_targets, 'Test')


# In[57]:


def plot_scores(accuracy, precision, recall, f1, name=''):
    scores = [accuracy, precision, recall, f1]
    score_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(score_names, scores, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f"{score:.2f}", ha='center', va='bottom', fontsize=12)
    
    plt.title(f'{name} Set Performance Metrics', fontsize=16)
    plt.ylabel('Scores', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.show()

val_accuracy = accuracy_score(val_targets, val_preds)
val_precision = precision_score(val_targets, val_preds, average='weighted')
val_recall = recall_score(val_targets, val_preds, average='weighted')
val_f1 = f1_score(val_targets, val_preds, average='weighted')


plot_scores(val_accuracy, val_precision, val_recall, val_f1, name='Validation')


# In[58]:


test_accuracy = accuracy_score(test_targets, test_preds)
test_precision = precision_score(test_targets, test_preds, average='weighted')
test_recall = recall_score(test_targets, test_preds, average='weighted')
test_f1 = f1_score(test_targets, test_preds, average='weighted')

plot_scores(test_accuracy, test_precision, test_recall, test_f1, name='Test')


# # Prediction on Single Input

# In[59]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

predict_input(new_input)


# # Model comparsion

# In[68]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

scaler_results = {}

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average='weighted')
    recall = recall_score(y_val, val_preds, average='weighted')
    f1 = f1_score(y_val, val_preds, average='weighted')
    return accuracy, precision, recall, f1

models = {
    'LogisticRegression': LogisticRegression(solver='liblinear'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=40),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=40),
    'SVM': SVC(kernel='linear', max_iter=5000),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=40),  
}

for scaler_name, scaler in [('StandardScaler', StandardScaler()), ('MinMaxScaler', MinMaxScaler())]:
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    
    results = {}
    
    
    for name, model in models.items():
        accuracy, precision, recall, f1 = evaluate_model(model, X_train_scaled, train_targets, X_val_scaled, val_targets)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
    scaler_results[scaler_name] = pd.DataFrame(results).T

plt.figure(figsize=(20, 10))

for i, (scaler_name, results_df) in enumerate(scaler_results.items(), 1):
    plt.subplot(1, 2, i)
    results_df.plot(kind='bar', figsize=(20, 10), rot=0, ax=plt.gca())
    plt.title(f'Model Comparison ({scaler_name}) and Mean Imputer')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()

plt.show()

for scaler_name, results_df in scaler_results.items():
    print(f"Results for {scaler_name}:\n")
    print(results_df, "\n")

