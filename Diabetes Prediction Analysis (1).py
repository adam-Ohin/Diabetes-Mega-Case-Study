#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries

# In[405]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[406]:


# using pandas to read the csv file
df = pd.read_csv('heart_2020_cleaned.csv.zip')


# In[407]:


# view the first 20 rows of the dataset
df.head(10)


# In[408]:


#view the general information of the dataset using the function
df.info()


# In[409]:


#df.drop(df.index[1500:319794], inplace=True)


# In[410]:


df.info()


# In[411]:


# Check for missing values
df.isnull().sum()


# In[412]:


#df.Race.value_counts()


# In[413]:


# check for zeros in the dataset, even though it is not a NAN
(df==0).sum()


# In[414]:


# replace zeros with NaN
df['PhysicalHealth'].replace(0,np.nan, inplace=True)
df['MentalHealth'].replace(0,np.nan, inplace=True)


# In[415]:


# view the first twenty rows to see the NAN
df.head(10)


# In[416]:


#Recheck for missing values
df.isnull().sum()


# In[417]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['HeartDisease'] = le.fit_transform(df['HeartDisease'])
df['Smoking'] = le.fit_transform(df['Smoking'])
df['AlcoholDrinking'] = le.fit_transform(df['AlcoholDrinking'])
df['Stroke'] = le.fit_transform(df['Stroke'])
df['DiffWalking'] = le.fit_transform(df['DiffWalking'])
df['Sex'] = le.fit_transform(df['Sex'])
df['Diabetic'] = le.fit_transform(df['Diabetic'])
df['PhysicalActivity'] = le.fit_transform(df['PhysicalActivity'])
df['GenHealth'] = le.fit_transform(df['GenHealth'])
df['Asthma'] = le.fit_transform(df['Asthma'])
df['KidneyDisease'] = le.fit_transform(df['KidneyDisease'])
df['SkinCancer'] = le.fit_transform(df['SkinCancer'])
df['Race'] = le.fit_transform(df['Race'])




# Print the resulting dataframe
df.head(10)


# In[418]:


df.head(10)


# In[419]:


df.drop(columns=['PhysicalHealth', 'MentalHealth', 'DiffWalking','GenHealth','SleepTime'], inplace=True)


# In[420]:


# create column and replace values below 10 by zero, equal and above 10 by one
#df['Diabetic'] = np.where(df['Diabetic'] <= 1.0, '0', '1')


# In[421]:


df.head(10)


# In[422]:


#df.drop(columns=['Diabetic'], inplace=True)


# In[423]:


df.head(10)


# In[424]:


# Give a glimpse of the measure of central tendency and a bulk of measure of dispersion
df.describe()


# In[425]:


#Recheck for missing values
df.isnull().sum()


# In[426]:


# total count of the outcome variable based on it's labels
df.HeartDisease.value_counts()


# In[429]:


#handling class imbalance
from sklearn.utils import resample

# Oversample the minority class
df_over = resample(df[df['HeartDisease'] == 1], replace=True, random_state=0)
# Combine the oversampled data with the original data
df = pd.concat([df, df_over], ignore_index=True)
# Check the new class distribution
print(df['HeartDisease'].value_counts())


# In[430]:



#from sklearn.utils import resample

# Undersample the majority class
#df_under = resample(df[df['HeartDisease'] == 0], replace=False, n_samples=27373)
# Combine the undersampled data with the original data
#df = pd.concat([df_under, df], ignore_index=True)
# Check the new class distribution
#print(df['HeartDisease'].value_counts())


# In[431]:



#from sklearn.utils import resample
#from imblearn.over_sampling import SMOTE

# Apply SMOTE
#df = SMOTE().fit_resample(df, df['HeartDisease'])
# Check the new class distribution
#print(df['HeartDisease'].value_counts())


# In[432]:


# plot outcome frequency
fig = plt.figure()   # produce the figure

df.HeartDisease.value_counts().plot (kind= 'bar', color='blue') # method for plotting
plt.xlabel ('Heart Disease')
plt.ylabel ('Number of patients')
plt.title ('A Bar Chart showing the frequencies of patients based on heart disease')


# In[433]:


df.Sex.value_counts()


# In[434]:


# plot outcome frequency
fig = plt.figure()   # produce the figure

df.Sex.value_counts().plot (kind= 'bar', color='Yellow') # method for plotting
plt.xlabel ('Sex')
plt.ylabel ('Number of patients')
plt.title ('A Bar Chart showing the frequencies of patients Sex')


# In[435]:


df.groupby(['Sex', 'HeartDisease']).size().unstack().plot(kind='bar')
# Show the plot
plt.legend()
plt.show()


# In[529]:


fig, ax = plt.subplots(figsize=(10, 5))
corr_mat = df.corr().round(2)  #creating a correlation matrix
mask = np.triu(np.ones_like(corr_mat, dtype=bool))  #masking the matrix to remove upper triangle

sns.heatmap(corr_mat, vmin= -1, vmax= 1, annot= True, center= 0, cmap= 'YlGnBu', mask=mask)


# In[437]:


#give hist plot of all variables based on the outcome labels
df.groupby('HeartDisease').hist(figsize=(10, 10))


# In[438]:


# a plot of boxplot to view outliers
sns.boxplot(data=df, orient="h", palette="Set1")
plt.title('Boxplot for each variable')


# In[439]:


#calculate for inter quantile range to see actual outliers
INR = df.quantile(q=0.75) - df.quantile(q=0.25) 

higher_outliers = df.quantile(q=0.75) + 0.15 * INR # higher band

lower_outliers = df.quantile(q=0.25) - 0.15 * INR  # lower band


# In[440]:


lower_outliers


# In[441]:


higher_outliers


# In[442]:


# removing outliers from the columns
df[(df['Smoking'] <= higher_outliers['Smoking']) & (df['Smoking'] >= lower_outliers['Smoking'])]
df[(df['Sex'] <= higher_outliers['Sex']) & (df['Sex'] >= lower_outliers['Sex'])]
df[(df['PhysicalActivity'] <= higher_outliers['PhysicalActivity']) & (df['PhysicalActivity'] >= lower_outliers['PhysicalActivity'])]


# # Create dependent and independent variable

# In[443]:


# create indepedent variable named x
x = df.drop(columns=['HeartDisease', 'AgeCategory']).values
#create dependent variable named y
y = df.iloc[:,:1].values


# In[444]:


print(x)


# In[445]:


y = np.ravel(y)


# 
# # Feature Selection

# In[446]:


from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
# Create the chi-square selector
selector = SelectKBest(score_func=chi2, k=11)  # Select the top 11 features
selector.fit_transform(x, y)
# Get the selected features
selected_features = selector.get_support(indices=True)
# Print the selected features
print(selected_features)


# In[447]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Create the ANOVA F1 selector
selector = SelectKBest(f_classif, k='all')
# Fit the selector to the data
selector.fit(x, y)
# Get the selected features
selected_feature = selector.get_support(indices=True)
# Print the selected features
print(selected_features)


# # Split data to test and train

# In[448]:


# split x and y to train and test and specify test smaple size.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split (x, y, random_state=0, test_size=0.3)


# In[449]:


x_test.size


# In[450]:


x_train.size


# # Feature scaling

# In[451]:


# from sklearn imoort  a feature scaler for the dependent variable
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


# In[452]:


# show the values of x
print(x_train)


# In[453]:


print(x_test)


# # Train model

# In[454]:


from sklearn.linear_model import LogisticRegression
logistic_regres = LogisticRegression()
logistic_regres.fit(x_train, y_train)


# In[455]:


from sklearn.naive_bayes import GaussianNB
baiyes_class = GaussianNB()
baiyes_class.fit(x_train, y_train)


# In[456]:


from sklearn.tree import DecisionTreeClassifier
tree_class = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree_class.fit(x_train, y_train)


# In[457]:


from sklearn.ensemble import RandomForestClassifier
forest_class = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True )
forest_class.fit(x_train, y_train)


# In[458]:


from sklearn.ensemble import GradientBoostingClassifier
xgboost = GradientBoostingClassifier()
xgboost.fit(x_train, y_train)


# # Predict model

# In[459]:


y_pred1 = logistic_regres.predict(x_test)
y_pred1


# In[460]:


y_pred2 = baiyes_class.predict(x_test)
y_pred2


# In[461]:


y_pred3 = tree_class.predict(x_test)
y_pred3


# In[462]:


y_pred4 = forest_class.predict(x_test)
y_pred4


# In[463]:


y_pred5 = xgboost.predict(x_test)
y_pred5


# In[116]:


# using SHAP to understand features that has more impact to our predictions
explainer = shap.LinearExplainer(logistic_regres, masker=x_test)
shap_values = explainer.shap_values(x_test)

features_names = ['BMI', 'Smoking','AlcholDrinking','PhysicalActivity', 'Stroke', 'Sex', 'Race','Diabetic', 'Asthma', 'KidneyDisease',
       'SkinCancer']


# In[117]:



# generate your summary plot
summary_plot = shap.summary_plot(shap_values, x_train, feature_names= features_names, plot_type="bar", plot_size=(6,4))


# In[ ]:


# using SHAP to understand features that has more impact to our predictions
explainer3 = shap.Explainer(tree_class)
shap_values3 = explainer3.shap_values(x_test)


# In[ ]:


# generate your summary plot
summary_plot = shap.summary_plot(shap_values3, x_train, feature_names= features_names, plot_type="bar",plot_size=(6,4))


# In[68]:


# using SHAP to understand features that has more impact to our predictions
explainer4 = shap.Explainer(forest_class)
shap_values4 = explainer4.shap_values(x_test)


# In[70]:


# generate your summary plot
summary_plot = shap.summary_plot(shap_values4, x_train, feature_names= features_names, plot_type='bar', plot_size=(6,4))


# In[71]:


# using SHAP to understand features that has more impact to our predictions
explainer5 = shap.Explainer(xgboost)
shap_values5 = explainer5.shap_values(x_test)


# In[73]:


# generate your summary plot
summary_plot = shap.summary_plot(shap_values5, x_train, feature_names= features_names, plot_type='bar', plot_size=(6,4))


# In[80]:


shap.decision_plot(explainer3.expected_value[0], shap_values3[0], feature_names= features_names, auto_size_plot=(4,4))


# In[82]:


shap.decision_plot(explainer4.expected_value[0], shap_values4[0], feature_names= features_names, auto_size_plot=(2,4))


# In[76]:


shap.decision_plot(explainer5.expected_value[0], shap_values4[0], feature_names= features_names)


# In[464]:


# Display confusion matrix from metrics class
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test, y_pred1)
print (cm_log)


# In[465]:


#Display confusion matrix from metrics class
from sklearn.metrics import confusion_matrix
cm_baiyes = confusion_matrix(y_test, y_pred2)
print (cm_baiyes)


# In[466]:


#Display confusion matrix from metrics class
from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test, y_pred3)
print (cm_tree)


# In[467]:


#Display confusion matrix from metrics class
from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(y_test, y_pred4)
print (cm_forest)


# In[468]:


#Display confusion matrix from metrics class
from sklearn.metrics import confusion_matrix
cm_xgboost = confusion_matrix(y_test, y_pred5)
print (cm_xgboost)


# In[469]:


# Extracting the confusion matrix features and evaluating the sensitivity and specificity of logistic regression model

cm_log_tp, cm_log_tn, cm_log_fp, cm_log_fn = cm_log.flatten()
log_sensitivity = cm_log_tp / (cm_log_tp + cm_log_tn)
log_specificity = cm_log_tn / (cm_log_tn + cm_log_fp)
print ('The model sensitivity is:', log_sensitivity*100)


# In[470]:


print ('The model specificity is:', log_specificity*100)


# In[471]:


# Extracting the confusion matrix features and evaluating the sensitivity and specificity of svm model

cm_baiyes_tp, cm_baiyes_tn, cm_baiyes_fp, accu_baiyes_fn = cm_baiyes.flatten() #unwind features in confusion matrix
baiyes_sensitivity = cm_baiyes_tp / (cm_baiyes_tp + cm_baiyes_tn)
baiyes_specificity = cm_baiyes_tn / (cm_baiyes_tn + cm_baiyes_fp)
print ('The model sensitivity is:', baiyes_sensitivity*100)


# In[472]:


print ('The model specificity is:', baiyes_specificity*100)


# In[473]:


# Extracting the confusion matrix features and evaluating the sensitivity and specificity of decision tree model
cm_tree_tp, cm_tree_tn, cm_tree_fp, accu_tree_fn = cm_tree.flatten()   #unwind features in confusion matrix
tree_sensitivity = cm_tree_tp / (cm_tree_tp + cm_tree_tn)
tree_specificity = cm_tree_tn / (cm_tree_tn + cm_tree_fp)
print ('The model sensitivity is:', tree_sensitivity*100)


# In[474]:


print ('The model specificity is:', tree_specificity*100)


# In[475]:


# Extracting the confusion matrix features and evaluating the sensitivity and specificity of random forest model
cm_forest_tp, cm_forest_tn, cm_forest_fp, accu_forest_fn = cm_forest.flatten()    #unwind features in confusion matrix
forest_sensitivity = cm_forest_tp / (cm_forest_tp + cm_forest_tn)
forest_specificity = cm_forest_tn / (cm_forest_tn + cm_forest_fp)
print ('The model sensitivity is:', forest_sensitivity*100)


# In[476]:


print ('The model specificity is:', forest_specificity*100)


# In[477]:


# Extracting the confusion matrix features and evaluating the sensitivity and specificity of xgboost model
cm_xgboost_tp, cm_xgboost_tn, cm_xgboost_fp, accu_xgboost_fn = cm_xgboost.flatten()    #unwind features in confusion matrix
xgboost_sensitivity = cm_xgboost_tp / (cm_xgboost_tp + cm_xgboost_tn)
xgboost_specificity = cm_xgboost_tn / (cm_xgboost_tn + cm_xgboost_fp)
print ('The model sensitivity is:', xgboost_sensitivity*100)


# In[478]:


print ('The model specificity is:', xgboost_specificity*100)


# In[479]:


cm_sensitivity = {'cm_log': log_sensitivity, 'cm_tree':tree_sensitivity,'cm_baiyes':baiyes_sensitivity, 'cm_forest':forest_sensitivity, 'cm_xgboost':xgboost_sensitivity }
cm_specificity = {'cm_log': log_specificity, 'cm_tree':tree_specificity, 'cm_baiyes':baiyes_specificity, 'cm_forest':forest_specificity, 'cm_xgboost':xgboost_specificity }
cm_df = pd.DataFrame(data=[cm_sensitivity, cm_specificity], index=['model_sensitivity', 'model_specificity'])
cm_df = cm_df*100
cm_df


# In[480]:


# get the accuracy score
from sklearn.metrics import accuracy_score
accu_log = accuracy_score(y_test, y_pred1)
accu_log


# In[481]:


# get the accuracy score
from sklearn.metrics import accuracy_score
accu_baiyes = accuracy_score(y_test, y_pred2)
accu_baiyes


# In[482]:


# get the accuracy score
from sklearn.metrics import accuracy_score
accu_tree = accuracy_score(y_test, y_pred3)
accu_tree


# In[483]:


# get the accuracy score
from sklearn.metrics import accuracy_score
accu_forest = accuracy_score(y_test, y_pred4)
accu_forest


# In[484]:


# get the accuracy score
from sklearn.metrics import accuracy_score
accu_xgboost = accuracy_score(y_test, y_pred5)
accu_xgboost


# In[485]:


accuracy = ['accu_log', 'accu_baiyes','accu_tree', 'accu_forest', 'accu_xgboost'] # creating variable names
accu_values = [accu_log, accu_baiyes, accu_tree, accu_forest, accu_xgboost]

accu_metric = np.arange(len(accuracy))
plt.figure (figsize=(6,6))    # create figure 
plt.xticks(accu_metric, accuracy)
barlist = plt.bar(accu_metric,accu_values) # plot values
barlist[0].set_color('g')
barlist[1].set_color('r')
barlist[2].set_color('y')
barlist[3].set_color('b')
barlist[4].set_color('c')
  

#display plot
plt.show()


# In[486]:


# get recall score
from sklearn.metrics import recall_score
recall_log = recall_score(y_test, y_pred1)
recall_log


# In[487]:


# get recall score
from sklearn.metrics import recall_score
recall_baiyes = recall_score(y_test, y_pred2)
recall_baiyes


# In[488]:


# get recall score
from sklearn.metrics import recall_score
recall_tree = recall_score(y_test, y_pred3)
recall_tree


# In[489]:


# get recall score
from sklearn.metrics import recall_score
recall_forest = recall_score(y_test, y_pred4)
recall_forest


# In[490]:


# get recall score
from sklearn.metrics import recall_score
recall_xgboost = recall_score(y_test, y_pred5)
recall_xgboost


# In[491]:


recall = ['recall_log','recall_baiyes', 'recall_tree', 'recall_forest','recall_xgboost' ] # create variable names
recall_values = [recall_log, recall_baiyes, recall_tree, recall_forest, recall_xgboost]

recall_metric = np.arange(len(recall))

plt.figure (figsize=(8,6))   # plot figure size
plt.xticks(recall_metric, recall)
barlist = plt.bar(recall_metric, recall_values) # plot figure
barlist[0].set_color('g')
barlist[1].set_color('r')
barlist[2].set_color('y')
barlist[3].set_color('b')
barlist[4].set_color('c')
 


# In[492]:


#get precision score
from sklearn.metrics import precision_score
precision_log = precision_score(y_test, y_pred1)
precision_log


# In[493]:


#get precison score
from sklearn.metrics import precision_score
precision_baiyes = precision_score(y_test, y_pred2)
precision_baiyes


# In[494]:


#get precison score
from sklearn.metrics import precision_score
precision_tree = precision_score(y_test, y_pred3)
precision_tree


# In[495]:


#get precison score
from sklearn.metrics import precision_score
precision_forest = precision_score(y_test, y_pred4)
precision_forest


# In[496]:


#get precison score
from sklearn.metrics import precision_score
precision_xgboost = precision_score(y_test, y_pred5)
precision_xgboost


# In[497]:


precision = ['precision_log','precision_baiyes','precision_tree', 'precision_forest', 'precision_xgboost'] # create variable names
precision_values = [precision_log, precision_baiyes, precision_tree, precision_forest, precision_xgboost]

precision_metric = np.arange(len(precision))

plt.figure (figsize=(12,8))   # create figure size
plt.xticks(precision_metric, precision)
barlist = plt.bar(precision_metric, precision_values) # plot figure
barlist[0].set_color('g')
barlist[1].set_color('r')
barlist[2].set_color('y')
barlist[3].set_color('b')
barlist[4].set_color('c') 


# In[498]:


#get f1-score
from sklearn.metrics import f1_score
f1_log = f1_score(y_test, y_pred1)
f1_log


# In[499]:


#get f1-score
from sklearn.metrics import f1_score
f1_baiyes = f1_score(y_test, y_pred2)
f1_baiyes


# In[500]:


#get f1-score
from sklearn.metrics import f1_score
f1_tree = f1_score(y_test, y_pred3)
f1_tree


# In[501]:


#get f1-score
from sklearn.metrics import f1_score
f1_forest = f1_score(y_test, y_pred4)
f1_forest


# In[502]:


#get f1-score
from sklearn.metrics import f1_score
f1_xgboost = f1_score(y_test, y_pred5)
f1_xgboost


# In[503]:


f1_scores = ['f1_log', 'f1_baiyes','f1_tree', 'f1_forest', 'f1_xgboost'] #create variable names
f1_values = [f1_log, f1_baiyes, f1_tree, f1_forest, f1_xgboost]

f1_metric = np.arange(len(f1_scores))
plt.figure (figsize=(12,8))            # plot figure size
plt.xticks(f1_metric, f1_scores)
barlist = plt.bar(f1_metric, f1_values) #plot figure
barlist[0].set_color('g')
barlist[1].set_color('r')
barlist[2].set_color('y')
barlist[3].set_color('b')
barlist[4].set_color('c') 

#display plot
plt.show()


# In[504]:


# get auc score
from sklearn.metrics import roc_auc_score
auc_log = roc_auc_score(y_test, y_pred1)
auc_log


# In[505]:


# get auc score
from sklearn.metrics import roc_auc_score
auc_baiyes = roc_auc_score(y_test, y_pred2)
auc_baiyes


# In[506]:


# get auc score
from sklearn.metrics import roc_auc_score
auc_tree = roc_auc_score(y_test, y_pred3)
auc_tree


# In[507]:


# get auc score
from sklearn.metrics import roc_auc_score
auc_forest = roc_auc_score(y_test, y_pred4)
auc_forest


# In[508]:


# get auc score
from sklearn.metrics import roc_auc_score
auc_xgboost = roc_auc_score(y_test, y_pred5)
auc_xgboost


# In[509]:


auc_scores = ['auc_log', 'auc_baiyes','auc_tree', 'auc_forest', 'auc_xgboost' ] # create variable names
auc_values = [auc_log, auc_baiyes, auc_tree, auc_forest, auc_xgboost]

auc_metric = np.arange(len(auc_scores))

plt.figure (figsize=(12,8))   # plot figure sze
plt.xticks(auc_metric, auc_scores)
barlist = plt.bar(auc_metric, auc_values) # plot figure
barlist[0].set_color('g')
barlist[1].set_color('r')
barlist[2].set_color('y')
barlist[3].set_color('b')
barlist[4].set_color('c')

#display plot
plt.show()


# In[510]:


# craete a dataframe of various evaluation metrics outcome for a better visualization
data = {'accuracy_score': accu_values, 'recall':recall_values, 'precision':precision_values, 'f1_score': f1_values, 'roc_auc': auc_values}
metrics = pd.DataFrame(data=data, index=['log_regres','baiyes','tree','forest', 'xgboost'])
metrics = metrics*100
metrics  #call variable


# In[511]:


model = ['log_regre', 'baiyes','tree', 'forest', 'xgboost'] # create variable names

# asssign metric values into a new variable
accu_values = [accu_log, accu_baiyes, accu_tree, accu_forest, accu_xgboost]
recall_values = [recall_log, recall_baiyes, recall_tree, recall_forest, recall_xgboost]
precision_values = [precision_log, precision_baiyes, precision_tree, precision_forest,  precision_xgboost]
f1_values = [f1_log, f1_baiyes, f1_tree, f1_forest, f1_xgboost]
auc_values = [auc_log, auc_baiyes, auc_tree, auc_forest, auc_xgboost]


model_metric = np.arange(len(model))

plt.figure(figsize=(6,4))  # plot figure size
plt.xticks(model_metric, model)

# plot figure
barlist = plt.bar(model_metric+0.2, accu_values, width=0.2, label= 'Accu_values')
barlist = plt.bar(model_metric, recall_values, width=0.2, label= 'Recall_values')
barlist = plt.bar(model_metric+0.4, precision_values, width=0.2, label= 'Precision_values')
barlist = plt.bar(model_metric-0.2, f1_values, width=0.2, label= 'F1_values')
barlist = plt.bar(model_metric+0.5, auc_values, width=0.2, label= 'Auc_values')

plt.title('bar charts showing the performance of different algorithms')
plt.ylabel('metrics')




plt.legend()
plt.savefig('metrics bar charts.jpeg')

