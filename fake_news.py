import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
"text": ["Breaking news election win",
         "Aliens landed in city",
         "Government announces new policy",
         "Fake miracle cure discovered"],
"label": [1,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X,y)

test = vectorizer.transform(["Aliens attack earth"])
prediction = model.predict(test)

print("Prediction:", prediction)
