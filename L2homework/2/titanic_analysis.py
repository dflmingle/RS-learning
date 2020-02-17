import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tpot import TPOTClassifier

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

# 得到决策树准确率
#这里用训练集计算准确率不甚合理，要考虑到过拟合的情形，不过只是过一下流程，倒也不必太在意
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print("ID3 score 准确率为 %.4lf%%" %( acc_decision_tree*100))


cls = DecisionTreeClassifier()
cls.fit(train_features, train_labels)
pred_labels = cls.predict(test_features)
acc_cart_tree = round(cls.score(train_features, train_labels), 6)
print("CART score 准确率为 %.4lf%%" %( acc_cart_tree*100))


xg=XGBClassifier()
xg.fit(train_features, train_labels)
pred_labels = xg.predict(test_features)
acc_xgboost = round(xg.score(train_features, train_labels), 6)
print("XGboost score 准确率为 %.4lf%%" %(acc_xgboost*100))

tpotcls=TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpotcls.fit(train_features, train_labels)
pred_labels = tpotcls.predict(test_features)
acc_tpot = round(tpotcls.score(train_features, train_labels), 6)
print("TPOT score 准确率为 %.4lf%%" %(acc_tpot*100))
tpotcls.export('tpot_titanic_pipeline.py')

