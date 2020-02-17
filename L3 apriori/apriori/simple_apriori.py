from efficient_apriori import apriori
import pandas as pd
# 设置数据集


fi=open('./Market_Basket_Optimisation.csv')
transactions=list()
lineNum=0

for line in fi:      
   temp=[(item) for item in line.strip('\n').split(',')]
   
   transactions.append(tuple(temp))   
   lineNum+=1
print("样本集大小为%d."%(lineNum))
#print(transactions)


# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.01,  min_confidence=0.5)
print("频繁项集：", itemsets)
print("关联规则：", rules)

