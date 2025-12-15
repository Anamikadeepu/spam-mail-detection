import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("C:\CSE\HelloWorld\Email\spam.csv")


data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not spam','spam'])

mess = data['Message']
cat = data['Category']
(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)

cv=CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#creating model

model= MultinomialNB()
model.fit(features,cat_train)

#Test the model
features_test = cv.transform(mess_test)
#print(model.score(features_test,cat_test))

#predct the data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header('Spam Detection')


output=predict('Congratulations , you won a lottery')
input_mess = st.text_input("Enter a message")

if st.button("Validate"):
    if input_mess.strip() == "":
        st.warning("Please enter a message")
    else:
        result = predict(input_mess)
        if result == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT spam")



