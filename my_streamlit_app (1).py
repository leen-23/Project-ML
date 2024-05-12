import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
vectorizer = joblib.load("/Users/leenharbi/Downloads/tfidf_vectorizer.pkl")
import os
print(os.path.exists("/Users/leenharbi/Downloads/tfidf_vectorizer.pkl"))

# تحميل النموذج المدرب والمتجه
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# تحديد العناصر الواجهة
st.title("Classifying sentiments ")
text_input = st.text_input("Please enter text:")

# التنبؤ بالتصنيف
if text_input:
    # تحويل النص إلى متجه TF-IDF
    text_vectorized = vectorizer.transform([text_input])
    # التنبؤ باستخدام النموذج
    prediction = model.predict(text_vectorized)
    # عرض نتيجة التنبؤ
    if prediction == 1:
        st.write("Positive: إيجابي")
    elif prediction == -1:
        st.write("Negative: سلبي")
    else:
        st.write("Neutral: طبيعي")
