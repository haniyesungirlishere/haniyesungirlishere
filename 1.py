import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv ("D:\lessons\datamining\proj2\hepatit.csv")
print(data.head())
print(data.columns)

data.columns = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 
                'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 
                'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 
                'Sgot', 'Albumin', 'Protime', 'Histology']
data.replace('?', np.nan, inplace=True)

numeric_columns = ["Age", "Bilirubin", "Alk Phosphate", "Sgot", "Albumin", "Protime"]

data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

binary_categorical_columns = ["Steroid", "Antivirals", "Fatigue", "Malaise", 
                              "Anorexia",
                              "Liver Big", "Liver Firm", "Spleen Palpable", 
                              "Spiders", "Ascites",
                              "Varices", "Histology"]
for col in binary_categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

X = data.drop("Class", axis=1)
y = data["Class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)


model = RandomForestClassifier(random_state=42)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
