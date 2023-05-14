#slip 29
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
students={'Rollno':[1,2,3,4,5,6],'java':[30,40,50,60,70,66],'DA':[45,55,62,71,44,87],'WT':[36,48,72,59,64,71]}
df=pd.DataFrame(students,columns=['Rollno','java','DA','WT'])
print(df)
x=df[['Rollno','java','DA']]
y=df['WT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred=logistic_regression.predict(x_test)
confusion_matrix=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['predicted'])
sn.heatmap(confusion_matrix,annot=True)
print('Accuracy is:',metrics.accuracy_score(y_test,y_pred))
plt.show()
print("prediction for new students")
new_students={'Rollno':[3,2,5,1,6,4],'java':[50,40,70,30,66,60],'DA':[62,87,71,55,45,44]}
df2=pd.DataFrame(new_students,columns=['Rollno','java','DA'])
y_pred=logistic_regression.prediction(df2)
print(df2)
print(y_pred)

