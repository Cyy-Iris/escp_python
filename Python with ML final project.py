#!/usr/bin/env python
# coding: utf-8

# In[185]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.encoding import WoEEncoder, RareLabelEncoder
from sklearn.feature_selection import f_regression
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# # 1. General data preparation

# In[186]:


#import data
retail_train = pd.read_csv('/Users/yifeichen/Downloads/train.csv')
retail_test = pd.read_csv('/Users/yifeichen/Downloads/test.csv')


# # Check missing values, create new label, drop features

# In[187]:


#Explore data
retail_train.head()
retail_test.head()


# In[188]:


#check if the IDs are unique: yes they are, so there is no need to group data
len(retail_train)
retail_train['id'].value_counts()


# In[189]:


#check missing values: no missing values for both training and testing sets
retail_train.isnull().any()


# In[190]:


retail_test.isnull().any()


# In[191]:


#check duplicates: no duplicates in training and testing data
retail_train.duplicated().any()
retail_test.duplicated().any()


# In[192]:


#check unique labels in training dataset：
#1) the negative samples were labelled as -1 
# 2） the training dataset is extremely imbalanced
retail_train['label'].value_counts() 


# In[193]:


# 1) For negative samples, replace label -1 with 0 in both datasets
retail_train['newlabel'] = retail_train.label.map(lambda x: 0 if x == -1 else 1)
retail_test['newlabel'] = retail_test.label.map(lambda x: 0 if x == -1 else 1)


# In[194]:


retail_train['newlabel'].value_counts() 


# In[195]:


#drop the original label
train_new = retail_train.drop(['label'], axis = 1)
test_new = retail_test.drop(['label'], axis = 1)


# In[196]:


#final check on training and testing datasets
train_new.head()


# In[197]:


test_new.head()


# In[198]:


#drop visitTime and purchaseTime 
train_new = train_new.drop(columns=['visitTime','purchaseTime'], axis = 1)
test_new = test_new.drop(columns=['visitTime','purchaseTime'], axis = 1)


# # 2. Training data preprocessing 

# # SMOTE for highly imbalanced label

# In[199]:


# 2) for imbalanced data, chose oversampling for label 1 using SMOTE
sm = SMOTE(random_state = 42)
X_sm, y_sm = sm.fit_resample(train_new, train_new.newlabel)


# In[200]:


print(f'''Shape of X before SMOTE: {train_new.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize = True) * 100


# In[201]:


#drop id
X_sm = X_sm.drop(['id'], axis = 1)


# # Feature handling: numerical

# In[202]:


#isolate the numerical features (column names denoted with N) and apply scaling 


# In[203]:


X_sm_num = X_sm[['hour','N1','N2','N3','N4','N5', 'N6','N7', 'N8', 'N9', 'N10']]


# In[204]:


# inspect the distribution of the numerical variables before scaling
fig, subplot_arr = plt.subplots(3,4,figsize=(18,12))
plt.subplot(3, 4, 1)
plt.hist(X_sm_num.loc[:,"hour"])
plt.title("hour")
for i in range(0,10):
  plt.subplot(3, 4, i+2)
  col="N"+str(i+1)
  plt.hist(X_sm_num.loc[:,col])
  plt.title(col)


# In[205]:


#apply scaling to numerical features based on distribution:
#apply standardscaler to hour
Scaler = StandardScaler()
X_sm_num.iloc[:,0:1] = Scaler.fit_transform(X_sm_num.iloc[:,0:1])


# In[206]:


#apply min-max scaler to the rest 
Scaler1 = MinMaxScaler()
X_sm_num.iloc[:,1:11] = Scaler1.fit_transform(X_sm_num.iloc[:,1:11])


# In[207]:


X_sm_num


# # Feature handling: categorical

# In[208]:


#isolate the categorical features(column names denoted with C)
X_smc = X_sm[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']]


# In[209]:


#the hased categorical features (column names denoted with C) as well as the class label
#are recognised as integer by the machine, convert to string then apply label encoding 

#1. label encoding: 
#class label
X_sm['newlabel'] = pd.factorize(X_sm['newlabel'].astype(str), sort = False)[0]


# In[210]:


X_sm_c = X_smc.astype(str)
X_sm_c1 = X_sm_c.apply(lambda x: pd.factorize(x, sort = False)[0])
X_sm_c1


# In[211]:


#2. rare label + weight of evidence (WOE) encoding:
#as the categorical features are hashed, we do not know if they are ordinal data 
#so to avoid ranking these features, we apply rare label + weight of evidence encoding for the categorical features
#this is for the Logistic Regression model, as ordinality isn't a problem for tree-based models


# In[212]:


# rare label encoding: 
# we set the threshold to 0.1 
# categories with proportion lower than 0.1 may not have any class label 1 due to the label imbalance
# and this will impede the application of WOE encoding (log 0 is undefined)

encoder = RareLabelEncoder(tol=0.1, n_categories=2, variables=['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'],
                           replace_with='Rare')
train_enc = encoder.fit_transform(X_sm_c)


# In[213]:


#WOE encoding:
woe_encoder = WoEEncoder(variables=['C1','C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'])
train_enc1 = woe_encoder.fit_transform(train_enc, X_sm['newlabel'])


# In[214]:


train_enc1


# # 3. Model Building 

# # Logistic Regression

# In[215]:


#reassemble training dataset
#for categorical features, use the datasets after applying rare label + WOE encoding


# In[216]:


#training dataset
train = X_sm_num.join(train_enc1).join(X_sm['newlabel'])
train


# In[217]:


# the most highly correlated variables (equal to or higher than 0.5): N4, N8, N9, N10, C7, C12. Select these features for our model
plt.figure(figsize=(12,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[218]:


# split the training dataset into train and test set 
X = train[['N4', 'N8', 'N9','N10', 'C7','C12']]
y = train['newlabel'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[219]:


#Model 1


# In[220]:


prediction = dict() 
model = LogisticRegression()
model.fit(X_train, y_train)


# In[221]:


#get coefficients and P value
model_coeffs = model.coef_
print("Coefficients:")
print(model_coeffs)
f_val, p_val = f_regression(X_train, y_train)
print("\n")
print("P value:") # all features are statistically significant
p_val


# In[222]:


#build the final train, test set with features that have coefficients higher than 0.5 
X_train_new = X_train[['N4','N8','N9','N10']]
X_test_new = X_test[['N4','N8','N9', 'N10']]


# In[223]:


#Model 2: Final


# In[224]:


prediction = dict() 
LR_final = LogisticRegression()
LR_final.fit(X_train_new, y_train)


# In[225]:


prediction['Logistic Regression'] = LR_final.predict(X_test_new)


# In[226]:


#accuracy, precision, recall, confusion matrix
print("Acurracy:")
print(accuracy_score(y_test, prediction['Logistic Regression']))
print("\n")
print("Classfication report:")
print(classification_report(y_test, prediction['Logistic Regression']))
print("\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction['Logistic Regression']))


# In[227]:


#scoring with train set
print('train score:', LR_final.score(X_train_new, y_train))


# In[228]:


# scoring with test set
print('test score:', LR_final.score(X_test_new, y_test))


# # Naive Bayes

# In[229]:


#use the same train test set as logistic regression
prediction = dict() 
NB = CategoricalNB()
NB.fit(X_train_new, y_train)


# In[230]:


prediction['Naive Bayes'] = NB.predict(X_test_new)


# In[231]:


#accuracy, precision, recall, confusion matrix
print("Acurracy:")
print(accuracy_score(y_test, prediction['Naive Bayes']))
print("\n")
print("Classfication report:")
print(classification_report(y_test, prediction['Naive Bayes']))
print("\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction['Naive Bayes']))


# In[232]:


#scoring with train set
print('train score:', NB.score(X_train_new, y_train))


# In[233]:


# scoring with test set
print('test score:', NB.score(X_test_new, y_test))


# # Random Forest

# In[234]:


#reassemble training dataset
#for numerical features, use the ones selected in Logistic Regression 
#for categorical features, use the datasets after applying label encoding 


# In[243]:


x = X_sm_num.drop(columns=['hour','N1','N2','N3','N5','N6','N7'], axis = 1)
train1 = x.join(X_sm_c1).join(X_sm['newlabel'])
train1


# In[244]:


#split the training dataset into train and test set 
X1 = train1.drop(['newlabel'] , axis =1)
y1 = train1['newlabel'] 
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2)


# In[245]:


prediction = dict() 
RF = RandomForestClassifier(random_state = 42)
RF.fit(X_train, y_train)


# In[246]:


prediction['Random Forest'] = RF.predict(X_test)


# In[247]:


#accuracy, precision, recall, confusion matrix
print("Acurracy:")
print(accuracy_score(y_test, prediction['Random Forest']))
print("\n")
print("Classfication report:")
print(classification_report(y_test, prediction['Random Forest']))
print("\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction['Random Forest']))


# In[248]:


#scoring with train set
print('train score:', RF.score(X_train, y_train))


# In[249]:


#scoring with test set
print('test score:', RF.score(X_test, y_test))


# # 4. Conclusion

# In[250]:


#Random Forest has the best train & test score
#However Logistic Regression achieved a good result with much fewer features
#We'll pick logistic regression to create the probablities table 


# # 5. Testing data preprocessing

# In[256]:


test_new_num = test_new[['N4','N8','N9','N10']]
test_new_num.iloc[:,0:4] = Scaler1.fit_transform(test_new_num.iloc[:,0:4])


# # 6. Probabilities Table

# In[257]:


#create probabilites table using logistic regression
prob = LR_final.predict_proba(test_new_num)
prob_df = pd.DataFrame(prob, index = test_new_num.index, columns = ['0','1'])
prob1 = prob_df.drop(['0'], axis =1) #keep the probablities for label 1
prob1


# In[258]:


prob_new = prob1.rename(columns={'1': 'Probabilites'})
final1 = test_new.join(prob_new)
finalcsv1 = final1.drop(final1.columns[1:25], axis = 1)
finalcsv1


# In[259]:


finalcsv1.to_csv(r'/Users/yifeichen/Downloads/Final_update.csv')

