import pandas as pd
import numpy as np
data=pd.read_csv("Crop_recommendation.csv")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
import matplotlib.pyplot as plt
data.head()
data.info()
data.describe()
data.isnull().sum()
# EDA
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 2)
sns.pairplot(data, hue = 'label')
sns.jointplot(x="rainfall",y="humidity",data=data[(data['temperature']<30) & (data['rainfall']>120)],hue="label")
sns.jointplot(x="K",y="N",data=data[(data['N']>40)&(data['K']>40)],hue="label")
sns.jointplot(x="K",y="humidity",data=data,hue='label',size=8,s=30,alpha=0.7)
sns.boxplot(y='label',x='ph',data=data)
sns.boxplot(y='label',x='P',data=data[data['rainfall']>150])
sns.lineplot(data = data[(data['humidity']<65)], x = "K", y = "rainfall",hue="label")
# plt.show()
# Data preprocessing
import pandas as pd
import numpy as np
data=pd.read_csv("Crop_recommendation.csv")
c=data.label.astype('category')
targets = dict(enumerate(c.cat.categories))
data['target']=c.cat.codes

y=data.target
X=data[['N','P','K','temperature','humidity','ph','rainfall']]
sns.heatmap(X.corr())
# Feature scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Model selection

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
sns.set(font_scale=1.0) 
plt.figure(figsize = (12,8))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},cmap="terrain")
k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.vlines(k_range,0, scores, linestyle="dashed")
plt.ylim(0.96,0.99)
plt.xticks([i for i in range(1,11)])
from sklearn.svm import SVC

svc_linear = SVC(kernel = 'linear').fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ",svc_linear.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}

model = GridSearchCV(estimator = SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)
print(model.best_score_ )
print(model.best_params_ )
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
clf.score(X_test,y_test) 
plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), clf.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
# plt.show()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from yellowbrick.classifier import ClassificationReport
classes=list(targets.values())
visualizer = ClassificationReport(clf, classes=classes, support=True,cmap="Blues")

visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
visualizer.show()
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier().fit(X_train, y_train)
print('Gradient Boosting accuracy : {}'.format(grad.score(X_test,y_test)))

import joblib

joblib.dump(clf, "models/crop_model.pkl")   
joblib.dump(scaler, "models/scaler.pkl")    
model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]

print(predict_crop(90, 40, 40, 25, 80, 6.5, 200))  

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]


st.title("🌱 Crop Recommendation System")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temp = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=70.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

if st.button("Predict Crop"):
    result = predict_crop(N, P, K, temp, humidity, ph, rainfall)
    st.success(f"✅ Recommended Crop: {result}")



