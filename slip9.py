#slip 6
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
dataset=pd.read_csv('/home/dell/DA Vrushali/Practicles/slip7.csv')
te=TransactionEncoder()
te_array=te.fit(dataset).transform(dataset)
df=pd.DataFrame(te_array,columns=te.columns_)
print("Result after preprocessing:")
print(df)
frequent_itemsets_ap=apriori(df,min_support=0.02,use_colnames=True)
print("\n Result after apriori algorithm")
print(frequent_itemsets_ap)
rules_ap=association_rules(frequent_itemsets_ap,metric="confidence",min_threshold=0.8)
frequent_itemsets_ap['length']=frequent_itemsets_ap['itemsets'].apply(lambda x:len(x))
print("\n Frequent 2 item sets")
print(frequent_itemsets_ap[frequent_itemsets_ap['length']>=2])
print("\n Frequent 3 item sets")
print(frequent_itemsets_ap[frequent_itemsets_ap['length']>=3])
