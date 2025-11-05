##IMPORTING PACKAGES

from django.shortcuts import render, redirect

# Create your views here.
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from django.contrib.auth.models import User
from django.contrib import messages
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Add, Input, Dense, Dropout
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier



Home = 'index.html'
About = 'about.html'
Login = 'login.html'
Registration = 'registration.html'
Userhome = 'userhome.html'
Load = 'load.html'
View = 'view.html'
Preprocessing = 'preprocessing.html'
Model = 'model.html'
Prediction = 'prediction.html'
Graph = 'graph.html'



#Home page
def index(request):

    return render(request, Home)

# About page
def about(request):

    return render(request,About)

# Login Page 
def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = User.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        return redirect(userhome)
    else:
        return render(request, Login)

#registration page user can registration here
def registration(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        print(Name, email, password, conpassword, age, contact)
        if password == conpassword:
            user = User(email=email, password=password)
            # user.save()
            return render(request, Login)
        else:
            msg = 'Register failed!!'
            return render(request, Registration)

    return render(request, Registration)

# user interface 
def userhome(request):

    return render(request, Userhome)

#Load Data
def load(request):
   if request.method == "POST":
        file = request.FILES['file']
        global df
        df = pd.read_csv(file)
        messages.info(request, "Data Uploaded Successfully")

   return render(request,Load)

def view(request):
    dataset = pd.read_csv('lending_club_loan_two.csv')
    dataset_sample = dataset.sample(frac=0.1)  # Take a random sample of 10% of the data
    
    # If you want to display the first 50 rows of the sampled data:
    dataset_sample = dataset_sample.head(50)
   
    return render(request, 'view.html', {'columns': dataset_sample.columns.values, 'rows': dataset_sample.values.tolist()})



#preprocessing data
def preprocessing(request):
    global x_train,x_test,y_train,y_test
    if request.method == "POST": 
        df=pd.read_csv(r'lending_club_loan_two.csv')
        size = int(request.POST['split'])
        size = size / 100
        df['emp_title'].fillna(df['emp_title'].mode()[0], inplace = True)
        df['emp_length'].fillna(df['emp_length'].mode()[0], inplace = True)
        df['title'].fillna(df['title'].mode()[0], inplace = True)
        df['revol_util'] = df['revol_util'].fillna(df['revol_util'].mean())
        df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].mean())
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].mean())
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['term'] = le.fit_transform(df['term'])
        df['grade'] = le.fit_transform(df['grade'])
        df['sub_grade'] = le.fit_transform(df['sub_grade'])
        df['emp_title'] = le.fit_transform(df['emp_title'])
        df['emp_length'] = le.fit_transform(df['emp_length'])
        df['home_ownership'] = le.fit_transform(df['home_ownership'])
        df['verification_status'] = le.fit_transform(df['verification_status'])
        df['issue_d'] = le.fit_transform(df['issue_d'])
        df['loan_status'] = le.fit_transform(df['loan_status'])
        df['purpose'] = le.fit_transform(df['purpose'])
        df['title'] = le.fit_transform(df['title'])
        df['earliest_cr_line'] = le.fit_transform(df['earliest_cr_line'])
        df['initial_list_status'] = le.fit_transform(df['initial_list_status'])
        df['application_type'] = le.fit_transform(df['application_type'])
        df['address'] = le.fit_transform(df['address'])
        x = df.drop(['loan_status'], axis=1)
        y = df['loan_status']
        x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=size,random_state=42)
        messages.info(request,"Data Preprocessed and It Splits Succesfully")

    return render(request, Preprocessing)


#Model Training
def model(request):
    global x_train,x_test,y_train,y_test

    df=pd.read_csv(r'lending_club_loan_two.csv')
    df['emp_title'].fillna(df['emp_title'].mode()[0], inplace = True)
    df['emp_length'].fillna(df['emp_length'].mode()[0], inplace = True)
    df['title'].fillna(df['title'].mode()[0], inplace = True)
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].mean())
    df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].mean())
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].mean())
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['term'] = le.fit_transform(df['term'])
    df['grade'] = le.fit_transform(df['grade'])
    df['sub_grade'] = le.fit_transform(df['sub_grade'])
    df['emp_title'] = le.fit_transform(df['emp_title'])
    df['emp_length'] = le.fit_transform(df['emp_length'])
    df['home_ownership'] = le.fit_transform(df['home_ownership'])
    df['verification_status'] = le.fit_transform(df['verification_status'])
    df['issue_d'] = le.fit_transform(df['issue_d'])
    df['loan_status'] = le.fit_transform(df['loan_status'])
    df['purpose'] = le.fit_transform(df['purpose'])
    df['title'] = le.fit_transform(df['title'])
    df['earliest_cr_line'] = le.fit_transform(df['earliest_cr_line'])
    df['initial_list_status'] = le.fit_transform(df['initial_list_status'])
    df['application_type'] = le.fit_transform(df['application_type'])
    df['address'] = le.fit_transform(df['address'])
    x = df.drop(['loan_status'], axis=1)
    y = df['loan_status']
    x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=42)
    if request.method == "POST":

        model = request.POST['algo']

        if model == "0":
            
            x_train = tf.convert_to_tensor(x_train,  dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            model = Sequential()
            model.add(Dense(30, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            History= model.fit(x_train[:100], y_train[:100], batch_size=500, epochs=10,validation_data=(x_test, y_test))
            abc=model.predict(x_test[:100])
            accc_ann =accuracy_score(abc,y_test[:100])
            msg = 'Accuracy of ANN : ' + str(accc_ann)
            return render(request,Model,{'msg':msg})
        
        
        elif model == "1":
            xgb_classifier = XGBClassifier(n_estimators=100)
            xgb_classifier.fit(x_train[:100], y_train[:100])
            y_pred = xgb_classifier.predict(x_test[:100])
            accc_xgb = accuracy_score(y_test[:100], y_pred)*100
           
                        
            msg = 'Accuracy of XGBClassifier :  ' + str(accc_xgb)
           
            return render(request,Model,{'msg':msg})
        
        elif model == "2":
           
            adb_classifier = AdaBoostClassifier(n_estimators=50,learning_rate=1)
            adb_classifier.fit(x_train[:100], y_train[:100])
            y_pred = adb_classifier.predict(x_test[:100])
            accc_adb = accuracy_score(y_test[:100], y_pred)*100           
            msg = 'Accuracy of AdaBoostClassifier :  ' + str(accc_adb)
            return render(request,Model,{'msg':msg})
        
        elif model == "3":
            
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(x_train[:100], y_train[:100])
            y_pred = rf_classifier.predict(x_test[:100])
            accc_rf = accuracy_score(y_test[:100], y_pred)*100  
            msg = 'Accuracy of RandomForestClassifier :  ' + str(accc_rf)
            return render(request,Model,{'msg':msg})
        
        elif model == "4":
            from sklearn.tree import DecisionTreeClassifier
            dt_classifier = DecisionTreeClassifier()
            dt_classifier.fit(x_train[:100], y_train[:100])
            y_pred = dt_classifier.predict(x_test[:100])
            accc_dt = accuracy_score(y_test[:100], y_pred)*100
            msg = 'Accuracy of DecisionTreeClassifier :  ' + str(accc_dt)
            return render(request,Model,{'msg':msg})
        
        elif model == "5":
            from sklearn.cluster import KMeans
            k_classifier = KMeans()
            k_classifier.fit(x_train)
            y_pred = k_classifier.predict(x_test)
            accc_k = accuracy_score(y_test, y_pred)*100
            msg = 'Accuracy of KMeans :  ' + str(accc_k)
            return render(request,Model,{'msg':msg})
        
        elif model == "6":
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier()
            knn.fit(x_train)
            y_pred = knn.predict(x_test)
            acc_knn = accuracy_score(y_test, y_pred)*100
            msg = 'Accuracy of KNN :  ' + str(acc_knn)
            return render(request,Model,{'msg':msg})

    return render(request, Model)




#Prediction here we can find the result based on user input values.
def prediction(request):

    global x_train,x_test,y_train,y_test,x,y

    df=pd.read_csv(r'lending_club_loan_two.csv')
    df['emp_title'].fillna(df['emp_title'].mode()[0], inplace = True)
    df['emp_length'].fillna(df['emp_length'].mode()[0], inplace = True)
    df['title'].fillna(df['title'].mode()[0], inplace = True)
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].mean())
    df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].mean())
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].mean())
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['term'] = le.fit_transform(df['term'])
    df['grade'] = le.fit_transform(df['grade'])
    df['sub_grade'] = le.fit_transform(df['sub_grade'])
    df['emp_title'] = le.fit_transform(df['emp_title'])
    df['emp_length'] = le.fit_transform(df['emp_length'])
    df['home_ownership'] = le.fit_transform(df['home_ownership'])
    df['verification_status'] = le.fit_transform(df['verification_status'])
    df['issue_d'] = le.fit_transform(df['issue_d'])
    df['loan_status'] = le.fit_transform(df['loan_status'])
    df['purpose'] = le.fit_transform(df['purpose'])
    df['title'] = le.fit_transform(df['title'])
    df['earliest_cr_line'] = le.fit_transform(df['earliest_cr_line'])
    df['initial_list_status'] = le.fit_transform(df['initial_list_status'])
    df['application_type'] = le.fit_transform(df['application_type'])
    df['address'] = le.fit_transform(df['address'])
    x = df.drop(['loan_status'], axis=1)
    y = df['loan_status']
    x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=42)
    

    if request.method == 'POST':

        term = float(request.POST['term'])
        int_rate = float(request.POST['int_rate'])
        installment = float(request.POST['installment'])
        grade = float(request.POST['grade'])
        sub_grade = float(request.POST['sub_grade'])
        annual_inc = float(request.POST['annual_inc'])
        verification_status = float(request.POST['verification_status'])
        title = float(request.POST['title'])
        revol_bal = float(request.POST['revol_bal'])
        total_acc = float(request.POST['total_acc'])
        mort_acc = float(request.POST['mort_acc'])
        pub_rec_bankruptcies = float(request.POST['pub_rec_bankruptcies'])
       


        PRED = [[term,int_rate,installment,grade,sub_grade,annual_inc,verification_status,title,revol_bal,total_acc,mort_acc,pub_rec_bankruptcies]]
        
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train[:1000], y_train[:1000])
        result = rf_classifier.predict(x_test)
        result=result[0]
        if result==0:
            msg="The prediction result is the Fully Paid"
        elif result==1:
            msg="The prediction result is the Charged Off"
   
        return render(request,Prediction,{'msg':msg})
    return render(request,Prediction)


#graph page
# def graph(request):

#     return render(request, Graph)