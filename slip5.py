#Slip 5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
students={'sepal_width':[1,2,3,4,5,6],'sepal_length':[3,4,5,6,7,6],'petal_width':[4,5,6,7,4,8],'petal_length':[3,4,7,5,6,7]}
df=pd.DataFrame(students,columns=['sepal_width','sepal_length','petal_width','petal_length'])
print(df)
print(df.describe())
x=df[['sepal_width','sepal_length','petal_length']]
y=df['petal_width']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred=logistic_regression.predict(x_test)
confusion_matrix=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['predicted'])
sn.heatmap(confusion_matrix,annot=True)
print('Accuracy is:',metrics.accuracy_score(y_test,y_pred))
plt.show()
print("prediction for flower")
new_students={'sepal_width':[3,2,5,1,6,4],'sepal_length':[5,4,7,3,6,6],'petal_width':[6,8,7,5,4,4]}
df2=pd.DataFrame(new_students,columns=['sepal_width','sepal_length','petal_width'])
y_pred=logistic_regression.prediction(df2)
print(df2)
print(y_pred)

