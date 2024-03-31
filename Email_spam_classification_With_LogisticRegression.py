
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')

print(df)

data = df.where((pd.notnull(df)),'')


data.head(10)


data.tail(10)


df.info()


data.shape


data.loc[data['Category'] == 'spam','Category'] = 0
data.loc[data['Category'] == 'ham','Category'] = 1


X = data['Message']
Y = data['Category']


print(X)

print(Y)


X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size= 0.2,random_state= 3)


print(X.shape)
print(X_train.shape)
print(X_test.shape)


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True) 

X_train_features = feature_extraction.fit_transform(X_train) # Fit and transform
X_test_features = feature_extraction.transform(X_test) # Use the fitted vectorizer

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


print(X_train_features)


model = LogisticRegression()
model.fit(X_train_features,Y_train)


prediction_on_training_data = model.predict(X_train_features)
acuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


print("Accuracy on training data : ",acuracy_on_training_data)


prediction_on_test_data = model.predict(X_test_features)
acuracy_on_test = accuracy_score(Y_test, prediction_on_test_data)


print("Accuracy on Testing data : ",acuracy_on_training_data)

input_message_mail = ["sometimes it’s easy to forget that good old SMS still exists. It may not be the spiffiest messaging technology out there, but the one great thing about SMS is that it's universal; you may now know whether someone is on Facebook Messenger or WhatsApp, but if you know their phone number, it's nearly certain they'll be able to receive an SMS message. What's even better is that the message technology is pretty universal, meaning you can even send a text from email."]
input_data_features = feature_extraction.transform(input_message_mail) 

prediction = model.predict(input_data_features)
print("Result score of prediction =", prediction)

if (prediction[0] == 1): 
    print("Ham mail")
else:
    print("spam mail")


