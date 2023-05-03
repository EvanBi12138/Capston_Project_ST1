'''
1. According to this dataset concerning various domains, which of them should implement into model's sample.
2. With regard to this dataset is consist of 60 features, how many should model keep to increase the efficiency of the prediction.
3. Is it possible to reduce the number of variables before starting the modeling process?
4. How the keywords impact the popularity of news?
5. Since the share count is a continuous variable, how should the threshold for the classifier be chosen?
'''



import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
import streamlit as st
from PIL import Image


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from joblib import dump, load

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# description of original dataset
def describeDataset():
    st.title('Section 1 Describe the Original dataset')
    st.write('This dataset has demonstrate the states of numerous online news, including multiple field from life to word,'
            'And Measuring the popularity of news articles based on their share count.')
    # read the csv file
    dfO = pd.read_csv(r"OnlineNewsPopularity.csv")
    # Check the description of original dataset
    st.subheader('1.1 Check the description of original dataset')
    headFiveO = dfO.head()
    tailFiveO = dfO.tail()
    st.write(headFiveO,tailFiveO)
    # rows and columns
    dataShapeO = dfO.shape
    # name of the attributes of original data set
    attributeO = dfO.columns
    st.subheader('1.2 The shape of original dataset(Attribute)')
    st.write(dataShapeO,attributeO)
    # unique values for each attribute
    nuniqueO = dfO.nunique()
    nuniqueO = nuniqueO.sort_values(ascending=False)
    st.subheader('1.3 unique values for each attribute (ordered)')
    st.write(nuniqueO)
    return dfO

def contentInterpretation(dfO):
    st.title('Section 2 Content Interpretation')

    st.subheader('2.1 The number of different field news')
    count1 = (dfO['data_channel_is_lifestyle'] == 1).sum()
    count2 = (dfO['data_channel_is_entertainment'] == 1).sum()
    count3 = (dfO['data_channel_is_bus'] == 1).sum()
    count4 = (dfO['data_channel_is_socmed'] == 1).sum()
    count5 = (dfO['data_channel_is_tech'] == 1).sum()
    count6 = (dfO['data_channel_is_world'] == 1).sum()

    st.write('lifestyle: ', count1)
    st.write('entertainment: ', count2)
    st.write('business: ', count3)
    st.write('social media: ', count4)
    st.write('technology: ', count5)
    st.write('world: ', count6)

    st.subheader('2.2 Given that there is a moderate sample size for business-related content, making it conducive to modeling,'
             ' this modeling task will focus on selecting business news articles.')
    st.write('Keep the target sample -- "Business"')

    count = (dfO['data_channel_is_bus'] == 1).sum()
    print('the number of entertainment sample: ', count)
    # withdraw this samples
    dfE = dfO.loc[dfO['data_channel_is_bus'] == 1]
    dfE.head()
    dfEShape = dfE.shape
    st.write('the shape of the new dataset: ', dfEShape)
    return dfE

def visualData(dfE):
    st.title('Section 3 Visualising data  distribution in detail')
    st.subheader('3.1 For each attribute')
    fig = plt.figure(figsize=(18, 18))
    ax = fig.gca()
    dfE.hist(ax=ax, bins=30)
    plt.show()
    st.pyplot(plt)

    st.subheader('3.2 Goal Attribute --- "SHARES"')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    goalAttribute = dfE['shares']
    goalAttribute.hist(ax=ax, bins=2500)
    plt.xlim([0, 30000])
    st.pyplot(plt)

def findThreshold(dfE):
    goalAttribute = dfE['shares']
    st.title('Section 4 Confirm goal feature')
    st.subheader('4.1 Threshold')
    q2 = goalAttribute.quantile(0.5)
    st.write('The threshold(0.5 quantile):', q2)

    st.subheader ('4.2 Renew the goal features')

    def get_popularity(shares):
        if shares <= q2:
            return 0
        else:
            return 1

    dfE['popularity'] = dfE['shares'].apply(get_popularity)
    st.write(dfE.head())

    return dfE


def reduceAttribute(dfE):
    st.title('Section 5 Reduce the number of variables')
    # 1 remove share cuz we get the new goal attribute, the exisitance of shares will absoultly influence the classifition
    st.subheader('  5.1 remove share cuz we get the new goal attribute, the exisitance of shares will absoultly influence the classifition.')
    dfN = dfE.drop(columns=['shares'])

    # 2 remove the url , txt varible is useless
    st.subheader('  5.2 remove the url , txt variable is useless.')
    dfN = dfN.drop(columns=['url'])

    # 3 remove timedelta, from the image of its disturibution ,its meaningless
    st.subheader('  5.3 remove timedelta, from the image of its distribution ,its meaningless.')
    dfN = dfN.drop(columns=['timedelta'])

    # 4 remove data_channel_is_*
    st.subheader('  5.4 remove data_channel_is_*.')
    dfN = dfN.drop(columns=['data_channel_is_lifestyle'])
    dfN = dfN.drop(columns=['data_channel_is_bus'])
    dfN = dfN.drop(columns=['data_channel_is_socmed'])
    dfN = dfN.drop(columns=['data_channel_is_tech'])
    dfN = dfN.drop(columns=['data_channel_is_world'])
    dfN = dfN.drop(columns=['data_channel_is_entertainment'])

    st.write("New dataframe's shape", dfN.shape)
    return dfN

def outlier(dfN):
    st.title('Section 6 Detecting outliers')
    dfN.plot(kind='box', subplots=True,
             layout=(11, 6), sharex=False, sharey=False, figsize=(25, 25), color='deeppink')
    st.pyplot(plt)

    continous_features = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
                          'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs',
                          'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length',
                          'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max',
                          'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
                          'self_reference_min_shares', 'self_reference_max_shares',
                          'self_reference_avg_sharess', 'LDA_00',
                          'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
                          'global_sentiment_polarity', 'global_rate_positive_words',
                          'global_rate_negative_words', 'rate_positive_words',
                          'rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity',
                          'max_positive_polarity', 'avg_negative_polarity',
                          'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
                          'title_sentiment_polarity', 'abs_title_subjectivity',
                          'abs_title_sentiment_polarity', 'popularity']

    def outliers(df, df_out, drop=False):
        for each_feature in df_out.columns:
            feature_data = df_out[each_feature]
            Q1 = np.percentile(feature_data, 25.)
            Q3 = np.percentile(feature_data, 75.)
            IQR = Q3 - Q1  # Interquartile Range
            outlier_step = IQR * 3  # adjust the IQR to reduce the removing of smaple
            outliers = feature_data[
                ~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
            if not drop:
                st.write('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
            if drop:
                df.drop(outliers, inplace=True, errors='ignore')
                st.write('Outliers from {} feature removed'.format(each_feature))
        return df

    dfNN = outliers(dfN, dfN[continous_features], drop=True)
    st.write("New dataframe's shape", dfNN.shape)

    return dfNN

def checkTarget(dfNN):
    st.title('Section 7 Descirbe the dataset after update')
    fig, ax = plt.subplots(figsize=(10, 8))
    name = ["Very popular", "Very unpopular"]
    ax = dfNN.popularity.value_counts().plot(kind='bar')
    ax.set_title("The popularity of the entertainment online news", fontsize=13, weight='bold')
    ax.set_xticklabels(name, rotation=0)

    # To calculate the percentage
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_x() + .09, i.get_height() - 50,
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
                color='white', weight='bold')

    plt.tight_layout()
    st.pyplot(plt)

def checkRelation(dfNN):
    # check correlation between variables
    st.title('Section 8 check correlation between variables')

    sns.set(style="white")
    plt.rcParams['figure.figsize'] = (50, 25)
    sns.heatmap(dfNN.corr(), annot=True, linewidths=.5, cmap="Blues")
    plt.title('Corelation Between Variables', fontsize=30)

    image = Image.open("relation.png")
    st.image(image, use_column_width=True)

    st.subheader('8.1 remove high correlation variables')
    st.write('self_reference_min_shares, self_reference_max_shares, LDA_04, LDA_03, rate_negative_words')
    # self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess
    dfNN = dfNN.drop(columns=['self_reference_min_shares'])
    dfNN = dfNN.drop(columns=['self_reference_max_shares'])
    # LDA_04, num_keywords
    dfNN = dfNN.drop(columns=['LDA_04'])
    # LDA_01, LDA_03
    dfNN = dfNN.drop(columns=['LDA_03'])
    # rate_positive_words, rate_negative_words
    dfNN = dfNN.drop(columns=['rate_negative_words'])
    st.write("New dataframe's shape", dfNN.shape)


def prepareModel(dfNN):
    st.title('Section 9 Original Model establishment')
    st.subheader('9.1 DataFrame of Train & Test')
    df = dfNN.copy()
    df = df.reset_index(drop=True)
    dfA = df.iloc[:, :-1]
    dfG = df.iloc[:, -1]
    seed = 1
    folds = 5
    scoring = 'accuracy'
    X = dfA
    y = dfG
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    st.write('shape of train:', np.shape(X_train))
    st.write('shape of test:', np.shape(X_test))

    st.subheader('9.2 Selection of Classifier')
    models = []
    models.append(('LOG', LogisticRegression()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))

    results = []
    names = []
    print("Performance on Training set")

    for name, model in models:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        msg += '\n'
        st.write(msg)

    st.subheader('9.3 Select RandomForest as Classifier')
    best_model = RandomForestClassifier()
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    st.write("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

    # store the classifier
    dump(best_model, 'rfc.joblib')

    return (dfA, dfG)

def modelOptimization(dfA, dfG):
    # load the classifier
    rf = load('rfc.joblib')
    st.title('Section 10 Model Optimization')
    st.subheader('10.1 Importance analysis')
    important = rf.feature_importances_
    name = dfA.columns
    importantDic = {}
    for i in range(len(name)):
        key = name[i]
        val = important[i]
        importantDic.update({key: val})
    importantDic1 = dict(sorted(importantDic.items(), key=lambda x: x[1]))
    st.write(importantDic1)

    st.subheader('10.2 Selection of important')
    st.write('Inputting variables into the model one by one in order of decreasing importance')
    importantList = list(importantDic1.keys())

    for i in range(3, 45):
        dfAN = createAttrubute(i, dfA, importantList)
        st.write("This is set attribute from 1 to {}: ".format(i), testAttribute(dfAN, dfG))

    st.subheader('10.3 Create new train dataset')
    st.write('Select Top 11 important attribute')
    attributeSet = 11
    dfANF = createAttrubute(11, dfA, importantList)
    st.write("New dataframe's shape", dfANF.shape)

    st.subheader('10.4 Confirm final model***')
    seed = 1
    folds = 5
    scoring = 'accuracy'

    X = dfANF
    y = dfG

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    best_model = load('rfcFinal.joblib')

    # best_model = RandomForestClassifier()

    # best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    st.write("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

    '''
    dump(best_model, 'rfcFinal3.joblib')
    keyFeature = X_train.columns
    with open('keyFeatures3.txt','w') as f:
        f.write(str(keyFeature))
    '''



    st.subheader('10.5 Classification report')
    report = classification_report(y_test, y_pred)
    st.text(report)

    st.subheader('10.6 ROC Curve')
    best_model = load('rfcFinal.joblib')
    best_model.fit(X_train, y_train)
    rf_roc_auc = roc_auc_score(y_test, best_model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest(area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('LOC_ROC')
    st.pyplot(plt)

    st.subheader('10.7 Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    st.pyplot(plt)

    st.subheader('10.8 Key Attribute')
    with open('keyFeatures.txt', 'r') as f:
        keyAttribute = f.read()
    st.write(keyAttribute)






def testAttribute(dfA, dfG):
    seed = 1
    folds = 5
    scoring = 'accuracy'

    X = dfA
    y = dfG

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf = RandomForestClassifier()
    best_model = rf
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    # print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


def createAttrubute(num,dfA,importantList):
    num = num + 1
    dfAN = dfA[importantList[-num:-1]]

    return dfAN
