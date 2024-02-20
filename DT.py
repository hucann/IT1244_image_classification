# Model 1: Decision Tree

# read data
import pandas as pd
df=pd.read_csv("data.csv")
df.head()

# get independent and dependent variables
inputs=df.drop("label",axis="columns")# this is the independent variable, axis=columns means that it is dropping columns intead of dropping rows
target=df["label"]# this is the dependent variable
target = pd.DataFrame(target)
##print(inputs)
##print(target)


# have to convert chacters(labels) into number
from sklearn.preprocessing import LabelEncoder
le_label=LabelEncoder()
target["label_n"] = le_label.fit_transform(target["label"])
target_n=target.drop(["label"],axis="columns")


# training and split the data into training and test data
from sklearn.model_selection import train_test_split
X=inputs.copy()
y=target["label_n"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn import tree
from sklearn.metrics import accuracy_score# for accuracy calculation
from sklearn.metrics import cohen_kappa_score

model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)#0.7895
kappa = cohen_kappa_score(y_test, y_pred)#0.7650795722728985
print(f"Accuracy: {accuracy}, Cohen Kappa: {kappa}")
