import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("jashore.csv")

print(data)
#Data Preprocessing
dataFill=data.fillna(value=300)
pd.set_option('display.max_column',None)
print(dataFill)

#Slicing the dataset
x=dataFill.iloc[:,7:8].values
y=dataFill.iloc[:,9].values
print(x)

#splitting the dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print(x_train)
print(x_test)

from sklearn.preprocessing import MinMaxScaler
scale1=MinMaxScaler()
x_train=scale1.fit_transform(x_train)
print(x_train)
print(x_test)


from sklearn.svm import SVC
cli=SVC(kernel="linear",random_state=0)
cli.fit(x_train,y_train)
y_predict=cli.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict)
print(cm)
acc=accuracy_score(y_test,y_predict)
print(acc)



#Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(data["Per Capita Income"],data["Family"], color='r', alpha=0.6, edgecolors='k')
plt.title("Per Capita Income vs Family")
plt.grid(ls='--', alpha=0.6)
plt.show()

#Histogram
plt.figure(figsize=(8, 6))
data["Per Capita Income"].plot(kind='hist', bins=10, color='Green', edgecolor='black', alpha=0.6)
plt.title("Distribution of Per Capita Income")
plt.grid(ls='--', alpha=0.6)
plt.show()

#Box Plot
plt.figure(figsize=(8, 6))
data.boxplot(column=["Per Capita Income"], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("Per Capita Income Box Plot")
plt.grid(ls='--', alpha=0.6)
plt.show()

#Bar Plot
plt.figure(figsize=(8, 6))
data.plot(kind="bar", x="Per Capita Income", y="Family", color='Red', alpha=0.6)
plt.title("Per Capita Income vs Family")
plt.grid(ls='--', alpha=0.6)
plt.show()

#Pie Chart
plt.figure(figsize=(8, 6))
data.groupby("Per Capita Income")["Family"].sum().plot(kind='pie', autopct='%1.1f%%', colormap='viridis')
plt.ylabel("")
plt.title("Family Distribution by Per Capita Income")
plt.show()
