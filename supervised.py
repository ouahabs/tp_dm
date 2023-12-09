import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


df = pd.read_csv("./Telco-Customer-Churn.csv", index_col = 'customerID')

def preprocess(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df_columns = df.columns.tolist()
    for column in df_columns:
        print(f"{column} unique values : {df[column].unique()}")

    features_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]

    for feature in features_na:
        print(f"{feature}, {round(df[feature].isnull().mean(), 4)} % Missing values")

    df[df[features_na[0]].isnull()]

    df.dropna(inplace=True)

    df.drop_duplicates(inplace=True)

    feature_le = ["Partner","Dependents","PhoneService", "Churn","PaperlessBilling"]
    def label_encoding(df,features):
        for i in features:
            df[i] = df[i].map({"Yes":1, "No":0})
        return df

    df = label_encoding(df,feature_le)
    df["gender"] = df["gender"].map({"Female":1, "Male":0})

    features_ohe = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
    df_ohe = pd.get_dummies(df, columns=features_ohe)

    features_mms = ["tenure","MonthlyCharges","TotalCharges"]

    df_mms = pd.DataFrame(df_ohe, columns=features_mms)
    df_remaining = df_ohe.drop(columns=features_mms)

    mms = MinMaxScaler(feature_range=(0,1))
    rescaled_feature = mms.fit_transform(df_mms)

    rescaled_feature_df = pd.DataFrame(rescaled_feature, columns=features_mms, index=df_remaining.index)
    df = pd.concat([rescaled_feature_df,df_remaining],axis=1)
    return df


class Model:
    def __init__(self):
        self.model = KNeighborsClassifier()
            
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
        




# #### Check observation of missing values

# In[10]:





# #### Drop missing values

# In[11]:





# ### 1.4. Checking for duplicate rows

# In[12]:





# #### Drop duplicate rows

# In[13]:




# ### 1.5. Checking for outliers

# In[14]:


# def outlier_check_boxplot(df,numerical_values):
#     number_of_columns = 2
#     number_of_rows = math.ceil(len(numerical_values)/2)
    
#     fig = plt.figure(figsize=(12,5*number_of_rows))
#     for index, column in enumerate(numerical_values, 1):
#         ax = fig.add_subplot(number_of_rows, number_of_columns, index)
#         ax = sns.boxplot(x = column, data = df, palette = "Blues")
#         ax.set_title(column)
#     # plt.savefig("Outliers_check.png", dpi=300)
#     return plt.show()


# In[15]:


# numerical_values = ["tenure","MonthlyCharges","TotalCharges"]
# outlier_check_boxplot(df,numerical_values)


# Each numerical variable doesn’t have an outlier

# ### 1.6. Checking the payment method

# In[16]:


# df["PaymentMethod"].unique()


# # #### Delete "automatic" from PaymentMethod

# # In[17]:


# df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", "", regex=False)


# ### 1.7. Target Variable Visualization

# In[18]:


# plt.style.use("ggplot")

# plt.figure(figsize=(5,5))
# ax = sns.countplot(x = df["Churn"],palette="Blues")
# ax.bar_label(ax.containers[0])
# # plt.savefig("Target_variable.png", dpi=300)
# plt.show()


# the dataset is not balanced 

# ### 1.8.  Plotting the data by feature

# #### 1.8.1. Custumer services

# In[19]:


# def plot_categorical_to_target(df,categorical_values, target):
#     number_of_columns = 2
#     number_of_rows = math.ceil(len(categorical_values)/2)
    
#     fig = plt.figure(figsize = (12, 5*number_of_rows))
    
#     for index, column in enumerate(categorical_values, 1):
#         ax = fig.add_subplot(number_of_rows,number_of_columns,index)
#         ax = sns.countplot(x = column, data = df, hue = target, palette="Blues")
#         ax.set_title(column)
#     return plt.show()

# customer_services = ["PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
#                     "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
# plot_categorical_to_target(df,customer_services, "Churn")


# #### 1.8.2. Customer Account Information [Categorical]

# In[20]:


# customer_account_cat = ["Contract","PaperlessBilling","PaymentMethod"]
# plot_categorical_to_target(df,customer_account_cat,"Churn")


# #### 1.8.3. Customer Account Information [Numerical]

# In[21]:


# def histogram_plots(df, numerical_values, target):
#     number_of_columns = 2
#     number_of_rows = math.ceil(len(numerical_values)/2)
    
#     fig = plt.figure(figsize=(12,5*number_of_rows))
    
#     for index, column in enumerate(numerical_values,1):
#         ax = fig.add_subplot(number_of_rows, number_of_columns, index)
#         ax = sns.kdeplot(df[column][df[target]=="Yes"] ,fill = True)
#         ax = sns.kdeplot(df[column][df[target]=="No"], fill = True)
#         ax.set_title(column)
#         ax.legend(["Churn","No Churn"], loc='upper right')
#     # plt.savefig("numerical_variables.png", dpi=300)
#     return plt.show()

# customer_account_num = ["tenure", "MonthlyCharges","TotalCharges"]
# histogram_plots(df,customer_account_num, "Churn")


# ## <span style="color:blue;"> 2. Feature Engineering </span> <a class="anchor" id="feature-engineering"></a>

# ### 1. Monthly Charges and Total Charges

# In[22]:


# df['monthly_to_total_ratio'] = df['MonthlyCharges'] / df['TotalCharges']


# In[23]:


# print(df.head())


# ## <span style="color:blue;"> 3. Data Preprocessing </span> <a class="anchor" id="data-preprocessing"></a>

# ### Converting categorical features to numerical features

# #### 3.2.1. Label encoding

# In[24]:





# #### 3.2.2. One Hot Encoding

# In[25]:





# In[26]:


# df.dtypes


# ## <span style="color:blue;"> 4. Feature Scaling </span> <a class="anchor" id="feature-scaling"></a>

# In[27]:




# ## <span style="color:blue;"> 5. Correlation analysis </span> <a class="anchor" id="feature-scaling"></a>

# In[28]:


# plt.figure(figsize=(10,6))
# df.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")
# # plt.savefig("correlation.png", dpi=300)
# plt.show()


# ## <span style="color:blue;"> 6. Splitting the dataset </span> <a class="anchor" id="feature-scaling"></a>

# In[29]:


# X = df.drop(columns = "Churn")
# y = df.Churn

# X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## <span style="color:blue;"> 7. Balancing the dataset </span> <a class="anchor" id="feature-scaling"></a>

# In[30]:


# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# #### Create a DataFrame from the resampled data

# In[31]:


# resampled_df = pd.DataFrame(X_train_resampled)
# resampled_df["Churn"] = y_train_resampled


# #### View the class distribution

# In[32]:


# resampled_df.describe()


# #### Plot the class distribution

# In[47]:


# plt.figure(figsize=(5, 5))
# ax = sns.countplot(x="Churn", data=resampled_df, palette="Blues")
# ax.bar_label(ax.containers[0])
# plt.title("Resampled Class Distribution")
# # plt.savefig("Target_variable_balanced.png", dpi=300)
# plt.show()


# ## <span style="color:blue;"> 7. Training and evaluating the model </span> <a class="anchor" id="feature-scaling"></a>

# ### 1. Preparing the functions

# In[34]:


# # For logistic Regression
# def feature_weights(X_df, classifier, classifier_name):
#     weights = pd.Series(classifier.coef_[0], index = X_df.columns.values).sort_values(ascending=False)
    
#     top_10_weights = weights[:10]
#     plt.figure(figsize=(7,6))
#     plt.title(f"{classifier_name} - Top 10 Features")
#     top_10_weights.plot(kind="bar")
    
#     bottom_10_weights = weights[len(weights)-10:]
#     plt.figure(figsize=(7,6))
#     plt.title(f"{classifier_name} - Bottom 10 Features")
#     bottom_10_weights.plot(kind="bar")
#     print("")


# # In[35]:


# def confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred, classifier, classifier_name):
#     cm = confusion_matrix(y_pred,y_test)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
#     disp.plot()
#     plt.title(f"Confusion Matrix - {classifier_name}")
#     # plt.savefig("Confusion_Matrix.png", dpi=300)
#     plt.show()
    
#     print(f"Accuracy Score Test = {accuracy_score(y_pred,y_test)}")
#     print(f"Accuracy Score Train = {classifier.score(X_train,y_train)}")
#     return print("\n")


# # In[36]:


# def roc_curve_auc_score(X_test, y_test, y_pred_probabilities,classifier_name):
#     y_pred_prob = y_pred_probabilities[:,1]
#     fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    
#     plt.plot([0,1],[0,1],"k--")
#     plt.plot(fpr,tpr,label=f"{classifier_name}")
#     plt.title(f"{classifier_name} - ROC Curve")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     # plt.savefig("ROC_Curve.png", dpi=300)
#     plt.show()
#     return print(f"AUC Score (ROC):{roc_auc_score(y_test,y_pred_prob)}")


# # In[37]:


# def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
#     y_pred_prob = y_pred_probabilities[:,1]
#     precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
#     plt.plot(recall,precision, label=f"{classifier_name}")
#     plt.title(f"{classifier_name}-ROC Curve")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.show()
#     # plt.savefig("Precision_recall_curve.png", dpi=300)
#     f1_score_result, auc_score = f1_score(y_test,y_pred), auc(recall,precision)
#     return print(f"f1 Score : {f1_score_result} \n AUC Score (PR) : {auc_score}")


# ### 1. KNN Model

# #### 1.1. Setting up the model

# In[38]:




# #### 1.2. Confusion matrix

# In[39]:


# confusion_matrix_plot(X_train,y_train,X_test, y_test, y_pred_knn, knn, "K-Nearest Neighbors")


# #### 1.3. ROC curve

# In[40]:


