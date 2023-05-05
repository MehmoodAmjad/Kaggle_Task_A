import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("rotten_tomatoes_movies_reviews.csv")
df.dropna(inplace=True)
# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
# Define the features and target variables
features = ['Review']
target = 'Freshness'
# Train the logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(train_data[features], train_data[target])
# Evaluate the model on the testing set
y_pred = lr.predict(test_data[features])
accuracy = accuracy_score(test_data[target], y_pred)
f1 = f1_score(test_data[target], y_pred, pos_label='fresh')
# Save the metrics to a JSON file using DVC
metrics = {'accuracy': accuracy, 'f1_score': f1}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)