import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("student_data.csv")
print(df["age"])
print(df["address"])
print(df["school"].tail(50))
print(df["famsize"])
print(df["Pstatus"])
print(df["Medu"])
print(df["Fedu"])
print(df["Mjob"].head(50))
print(df["Fjob"].head(50))
print(df["reason"].head(50))
print(df["guardian"])
print(df["traveltime"])
print(df["studytime"])
print(df["failures"])
print(df["schoolsup"])
print(df["famsup"])
print(df["paid"])
print(df["activities"])
print(df["nursery"])
print(df["higher"])
print(df["internet"])
print(df["romantic"])
print(df["famrel"])
print(df["freetime"])
print(df["goout"])
print(df["Dalc"])
print(df["Walc"])
print(df["health"])
print(df["absences"])
print(df[["G1","G2","G3"]])
print(df.info())
from sklearn.preprocessing import LabelEncoder
encoders = {}
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
print("Label Encoding Done Successfully! ")
print(df)


df.to_csv("cleaned student data.csv")
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("cleaned student data.csv")

#Male v/s Female student
count = df["sex"].value_counts()
plt.figure()
plt.bar(count.index,count.values)
plt.xticks=([0,1],["Female","Male"])
plt.xlabel("Gender")
plt.ylabel("Gender distribution of student")
plt.show()
#Study time vs performance
df.groupby("studytime")["G3"].mean()
plt.figure()
plt.plot(df.groupby("studytime")["G3"].mean())
plt.xlabel("Study time level ")
plt.ylabel("Performace")
plt.title("Study time vs Performace")
plt.show()
#Absences vs grades
plt.figure()
plt.scatter(df["absences"],df["G3"])
plt.xlabel("Absence")
plt.ylabel("Final Gread")
plt.title("Absence V/S Performace ")
plt.show()
#Alcohol vs grades
plt.figure()
plt.scatter(df["Dalc"],df["G3"])
plt.xlabel("Daily alcohol consumption")
plt.ylabel("Final Gread")
plt.title("Daily alcohol V/S Final Performace")
plt.show()
#Family relation v/s gread
plt.figure()
plt.scatter(df["famrel"],df["G3"])
plt.xlabel("Family relation")
plt.ylabel("Final Grade ")
plt.title("Family relation v/s gread ")
plt.show()
#Going out vs study time
plt.figure()
plt.scatter(df["goout"],df["studytime"])
plt.xlabel("Going out")
plt.ylabel("Study time")
plt.title("Going out vs study time")
plt.show()


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report

df=pd.read_csv("cleaned student data.csv")
rf=RandomForestClassifier(n_estimators=400,random_state=92,class_weight="balanced",max_depth=None)
X=df[["age","school","Pstatus","Medu","Fedu","reason","traveltime","studytime",
      "failures","schoolsup","famsup","paid","activities","higher",
      "internet","romantic","famrel","freetime","goout","health","absences","famsize",
      "Medu","Fedu","Mjob","Fjob","reason",]]
y=df["G3"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=92)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (weighted):", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


"""from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report

df=pd.read_csv("cleaned student data.csv")
rf=RandomForestRegressor(n_estimators=400,random_state=92,max_depth=None)
X=df[["age","school","Pstatus","Medu","Fedu","reason","traveltime","studytime",
      "failures","schoolsup","famsup","paid","activities","higher",
      "internet","romantic","famrel","freetime","goout","health","absences","famsize",
      "Medu","Fedu","Mjob","Fjob","reason",]]
y=df["G3"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=92)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (weighted):", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

df = pd.read_csv("cleaned student data.csv")

X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=92
)

rf = RandomForestRegressor(
    n_estimators=400,
    random_state=92
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
