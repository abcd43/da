#Slip 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
students={'ID':[1,2,3,4,5,6],'Gender':[0,1,1,0,1,0],'Age':[30,25,18,21,35,53]}
df=pd.DataFrame(students,columns=['ID','Gender','Age'])
print(df)
x=df[['ID','Gender']]
y=df['Age']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred=logistic_regression.predict(x_test)
confusion_matrix=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['predicted'])
sn.heatmap(confusion_matrix,annot=True)
print('Accuracy is:',metrics.accuracy_score(y_test,y_pred))
plt.show()
print("prediction for new students")
new_students={'ID':[1,3,4,2,5,6],'Gender':['female','female','male','male','female','male']}
df2=pd.DataFrame(new_students,columns=['ID','Gender'])
y_pred=logistic_regression.prediction(df2)
print(df2)
print(y_pred)

