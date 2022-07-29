#importing libraries
import streamlit as st
import pandas as pd
import pycaret
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
import operator 

from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pycaret.utils import check_metric
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

#Setting up the front page
st.set_page_config(layout="wide", page_title='Explainable AI ML Model')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<u><h1 style='text-align: center; color: black;'>Explainable AI</h1></u>", unsafe_allow_html=True)
st.markdown("<u><h1 style='text-align: center; color: black;'>Diving into Artificial Intelligence</h1></u>", unsafe_allow_html=True)
st.write("")

st.markdown("An open source software that provides a user interface for implementing several Machine Learning algorithms, as well as visualisation tools, allowing users to develop machine learning techniques, apply them to real-world data mining problems, and gain insights from the data")
st.markdown("**This tool provides:**")
st_col1,st_col2,st_col3=st.columns(3)
with st_col1:
    st.markdown("Data Insights ")
    st.markdown("Pre-Processing Techniques used")
with st_col2:
    st.markdown("Confusion Matrix")
    st.markdown("Feature Importance Plots")
with st_col3:
    st.markdown("Correlation and Heat Map")
    st.markdown("Visualizations of important features")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#Radio button for the user
Option = st.sidebar.radio(
"Do you want to give a dataset?",
('Yes','No'))

if(Option=='Yes'):
    dataset_name = st.sidebar.file_uploader("Upload the file", type={"csv"})
    if dataset_name is None:
        st.stop()
    else:
        df = pd.read_csv(dataset_name)
        text_input = st.sidebar.text_input('Please enter the column name', )
        if(text_input in df.columns):
            target=text_input
        else:
            st.sidebar.markdown("Please enter the correct Column name")

        
else:
    dataset_name = st.sidebar.selectbox("Select DataSet",
                                    ("Diabetes Prediction", "Readmission after Diabetes Treatment", "Liver DataSet"))

    def get_dataset(dataset_name):
        if(dataset_name == "Diabetes Prediction"):
            df = pd.read_csv(r"C:/Users/AI04821/Ujwal_Legato/dataset_diabetes/diabetes.csv")
            st.markdown("""
                    
            This interactive application explains the prediction of re admisison of Diabetic Patinets [Diabetes Data](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dataset using **Explainable AI** technique.
        """)
            st.markdown("""
            **Data Set Information:** 
            The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
        
        """)
        #Setting up the dataframe.
        elif(dataset_name =="Readmission after Diabetes Treatment"):
            df = pd.read_csv(r"C:/Users/AI04821/Ujwal_Legato/dataset_diabetes/processed file.csv")
            df = df.drop(['Unnamed: 0'], axis=1)
            st.markdown("""
                    
            This interactive application explains the prediction of re admisison of Diabetic Patinets [Diabetes Readmission Data](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) dataset using **Explainable AI** technique.
        """)

            st.markdown("""
            **Data Set Information:**

            The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.
            (1) It is an inpatient encounter (a hospital admission).
            (2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
            (3) The length of stay was at least 1 day and at most 14 days.
            (4) Laboratory tests were performed during the encounter.
            (5) Medications were administered during the encounter.
            The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.
        
        """)

        else:
            df = pd.read_csv(r"C:/Users/AI04821/Ujwal_Legato/dataset_diabetes/indian_liver_patient.csv")
            st.markdown("""
                    
            This interactive application explains the prediction of re admisison of Liver Patinets [Liver Data](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records) dataset using **Explainable AI** technique.
        """)

            st.markdown(
                """
                **Data Set Information:**
                This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.
                """
            )
            #df=df.drop(['Unnamed: 0'],axis=1)
        target = df.columns[-1]
        return df, target
    df, target = get_dataset(dataset_name)

#Model Selection
classifier_name = st.sidebar.selectbox("Select Classifier", ("Please select a Model",
    "Logistic Regression", "Decision Tree", "Random Forest","XG Boost", "LGBM", "Support Vector Machine"))

def get_classifier(classifier_name):
    if(classifier_name=="Please select a Model"):
        st.stop()
    elif (classifier_name == "Logistic Regression"):
        val = "lr"
    elif (classifier_name == "Decision Tree"):
        val = "dt"
    elif (classifier_name == "Random Forest"):
        val = "rf"
    elif (classifier_name == "Support Vector Machine"):
        val = "svm"
    elif (classifier_name == "XG Boost"):
        val = "xgboost"
    elif (classifier_name == "LGBM"):
        val = "lightgbm"
    return val

#Data Preprocessing information
st.markdown("<u><h2 style='text-align: center; color: black;'>Data Preprocessing Information</h2></u>", unsafe_allow_html=True)
st.write("")
ex_col1, ex_col2, ex_col3= st.columns(3)
with ex_col1:
    st.write("**Size of the data is:**", df.shape)
    st.markdown("**Test - Train Split:** 70% and 30%")

with ex_col2:
    st.markdown("**Outlier Treatment:** True")
    st.markdown('**Normalize Data:** True')

with ex_col3:
    st.markdown('**Fix Imbalance Dataset:** True')
    st.markdown('**Technique to use imbalance Dataset:** SMOTE(Oversampling)')


classifier = get_classifier(classifier_name)

#MOdel Training
s = setup(data=df, target=target, session_id=123,fix_imbalance=True, normalize=True, remove_outliers=True, silent=True)
model_temp = create_model(classifier)
tuned_model=tune_model(model_temp)
new_prediction = predict_model(tuned_model)

#Performance metrics
accuracy_score=check_metric(new_prediction[target], new_prediction['Label'], metric = 'Accuracy')
precision_rate=check_metric(new_prediction[target], new_prediction['Label'], metric = 'Precision')
recall_rate=check_metric(new_prediction[target], new_prediction['Label'], metric = 'Recall')
f1_score=check_metric(new_prediction[target], new_prediction['Label'], metric = 'F1')

#Displaying Performance Metrics
st.sidebar.write("")
st.sidebar.subheader("**Model Accuracy**")
st.sidebar.markdown(round(accuracy_score*100,2))

st.sidebar.subheader("**Model Precision**")
st.sidebar.markdown(round(precision_rate*100,2))

st.sidebar.subheader("**Model Recall**")
st.sidebar.markdown(round(recall_rate*100,2))

st.sidebar.subheader("**Model F1 Score**")
st.sidebar.markdown(round(f1_score*100,2))


st.markdown("<u><h1 style='text-align: center; color: black;'>RESULTS</h1></u>", unsafe_allow_html=True)

#Confusion Matrix PLot
st.markdown("<u><h2><ins style='text-align: center; color: black;'>Confusion Matrix</ins></h2></u>", unsafe_allow_html=True)
st.write("")
st.markdown("""A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model. This gives us a holistic view of how well our classification model is performing and what kinds of errors it is making [link](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20Confusion%20matrix%20is%20an,by%20the%20machine%20learning%20model)""")


plot_model(tuned_model,plot='confusion_matrix', scale=0.00001,display_format='streamlit')
st.markdown("**Need for Confusion Matrix in Machine learning**")
st.markdown("It evaluates the performance of the classification models, when they make predictions on test data, and tells how good our classification model is.")
st.markdown("It not only tells the error made by the classifiers but also the type of errors such as it is either type-I or type-II error.")
st.markdown("With the help of the confusion matrix, we can calculate the different parameters for the model, such as accuracy, precision, etc.")
ex_col1, ex_col2= st.columns(2)

#Feature Importance Plots
st.markdown("<u><h2 style='text-align: center; color: black;'>Feature Importance Graphs</h2></u>", unsafe_allow_html=True)
st.write("")
st.markdown("Feature Importance assigns the score of input features based on their importance to predict the output. More the features will be responsible to predict the output more will be their score. We can use it in both classification and regression problem.")
st.markdown("<u><h2><ins style='text-align: center; color: black;'>Encoded Feature Importance Graph</ins></h2></u>", unsafe_allow_html=True)
st.markdown("This plot shows the importance for the encoded features. The graph is displaying the importance of top 10 features.")
plot_model(tuned_model,plot='feature', scale=1,display_format='streamlit')

#Getting Feature scores for Original Features 
if(classifier=="dt" or classifier=="rf" or classifier=="xgboost" or classifier=="lightgbm"):
    col=tuned_model.feature_importances_
    df1=pd.DataFrame({'Feature': get_config('X_train').columns})
    df1['Value'] = col.tolist()
    df1['Value']=df1['Value'].abs()
    df1.sort_values(by="Value", ascending=False)
    arr_original=df.columns
    coded_list = arr_original.tolist()
    dict={}
    i=0
    for ele in df1['Feature']:
        if ele in coded_list:
            dict[ele]=df1['Value'][i]
            coded_list.remove(ele)
        else:
            add=0
            for items in coded_list:
                if items in dict:
                    dict[items]=dict[items]+df1['Value'][i]
                else:
                    string = ele
                    substring = items
                    if substring in string:
                        add=add+df1['Value'][i]
                        dict[items]=add
                    else:
                        continue
            del add
        i+=1
    del i
    sorted_dict = sorted(dict.items(),key = operator.itemgetter(1),reverse=True)

elif(classifier=="lr" or classifier=="svm"):
    col=tuned_model.coef_
    df1=pd.DataFrame({'Feature': get_config('X_train').columns})
    col=col.tolist()
    col=" ".join(str(x) for x in col)
    col=col.split(',')
    for i in range(len(col)):
        col[i] = col[i].replace('[', '')
        col[i] = col[i].replace(']', '')
    df1['Value'] = col
    df1['Value']=pd.to_numeric(df1['Value'])
    df1['Value']=df1['Value'].abs()
    df1.sort_values(by="Value", ascending=False)
    arr_original=df.columns
    coded_list = arr_original.tolist()
    dict={}
    i=0
    for ele in df1['Feature']:
        if ele in coded_list:
            dict[ele]=df1['Value'][i]
            coded_list.remove(ele)
        else:
            add=0
            for items in coded_list:
                if items in dict:
                    dict[items]=dict[items]+df1['Value'][i]
                else:
                    string = ele
                    substring = items
                    if substring in string:
                        add=add+df1['Value'][i]
                        dict[items]=add
                    else:
                        continue
            del add
        i+=1
    sorted_dict = sorted(dict.items(),key = operator.itemgetter(1),reverse=True)

top_five_features=list()
for i in range(len(sorted_dict[:5])):
    top_five_features.append(sorted_dict[:5][i][0])


st.markdown("<u><h2><ins style='text-align: center; color: black;'>Feature Plot</ins></h2></u>", unsafe_allow_html=True)
st.write("")
st.markdown("This plot shows the importance for the Original features. The graph is displaying the importance of top 10 features.")
#st.header("*Feature Plot*")
x, y = zip(*reversed(sorted_dict[:10]))
fig = plt.figure(figsize = (10, 5))
plt.ylabel("Features")
plt.xlabel("Value")
plt.title("Feature Importance Graph")
plt.barh(x, y, height=0.05, align="center")
plt.plot(y, x, 'o')
st.pyplot(fig)
if(classifier=="lr" or classifier=="svm"):
    st.markdown("The **Linear Model** will fit on the classification dataset and retrieve the coeff_ property that contains the coefficients found for each input variable.These coefficients can provide the basis for a crude feature importance score. This assumes that the input variables have the same scale or have been scaled prior to fitting a model.[Link](https://machinelearningmastery.com/calculate-feature-importance-with-python/)")
else:
    st.markdown(" The **Tree based Model** will  will fit on the classification dataset, then the model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.[Link](https://machinelearningmastery.com/calculate-feature-importance-with-python/)")

#Displaying top five features
st.markdown("<u><h2><ins style='text-align: center; color: black;'>Top Five Features</ins></h2></u>", unsafe_allow_html=True)
st.markdown("From the above Feature Plot,we got the below top five features.")
ax_col1,ax_col2=st.columns(2)
with ax_col1:
    st.write("**Feature One:**",top_five_features[0])
    st.write("**Feature Two:**",top_five_features[1])
    st.write("**Feature Three:**",top_five_features[2])
with ax_col2:
    st.write("**Feature Four:**",top_five_features[3])
    st.write("**Feature Five:**",top_five_features[4])

#HeatMap 
st.markdown("<u><h2><ins style='text-align: center; color: black;'>Important Feature - Heatmap</ins></h2></u>", unsafe_allow_html=True)
st.write("")
st.markdown("A heatmap is a graphical representation of data that uses a system of color-coding to represent different values. Heatmaps are used in various forms of analytics but are most commonly used to show user behavior on specific webpages or webpage templates.")
st.markdown("Heatmaps are also a lot more visual than standard analytics reports, which can make them easier to analyze at a glance. This makes them more accessible, particularly to people who are not accustomed to analyzing large amounts of data.")

if(len(sorted_dict)<=10):
    cor=df.corr()
    fig=plt.figure(figsize=(20,20))
    sns.heatmap(cor, annot=True)
    st.pyplot(fig)
else:
    top_ten_features=list()
    for i in range(len(sorted_dict[:10])):
        top_ten_features.append(sorted_dict[:10][i][0])
    df_temp=df[top_ten_features]
    df_temp1=df_temp
    df_temp1[target]=df[target]
    cor=df_temp1.corr()
    fig=plt.figure(figsize=(20,20))
    sns.heatmap(cor, annot=True)
    st.pyplot(fig)

st.markdown("Heatmaps are used to show relationships between two variables, one plotted on each axis. By observing how cell colors change across each axis, you can observe if there are any patterns in value for one or both variables.")
st.markdown("The variables plotted on each axis can be of any type, whether they take on categorical labels or numeric values. In the latter case, the numeric value must be binned like in a histogram in order to form the grid cells where colors associated with the main variable of interest will be plotted.")
st.markdown("Cell colorings can correspond to all manner of metrics, like a frequency count of points in each bin, or summary statistics like mean or median for a third variable. One way of thinking of the construction of a heatmap is as a table or matrix, with color encoding on top of the cells. In certain applications, it is also possible for cells to be colored based on non-numeric values (e.g. general qualitative levels of low, medium, high")

#Important Feature Visualization
cat_feature=[]
numeric_feature=[]

if(len(sorted_dict)<=10):
    df_temp_small=df[top_five_features]
    for ele in df_temp_small.columns:
        if df[ele].dtype==object:
            cat_feature.append(ele)
        else:
            numeric_feature.append(ele)

else:
    for ele in df_temp.columns:
        if df[ele].dtype==object:
            cat_feature.append(ele)
        else:
            numeric_feature.append(ele)

st.markdown("<u><h2><ins style='text-align: center; color: black;'>Important Feature Visulatization</ins></h2></u>", unsafe_allow_html=True)
#st.header("*Important Feature Visulatization*")
if(len(cat_feature)>0):
    for ele in cat_feature:
        fig = plt.figure(figsize=(9, 6))
        ax = sns.countplot(x=ele, data=df, orient="H")
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
        st.pyplot(fig)

if(len(numeric_feature)>0):
    for ele in numeric_feature:
        fig = plt.figure(figsize=(15,9))
        sns.histplot(df[ele])
        plt.xlabel(ele)
        st.pyplot(fig)












