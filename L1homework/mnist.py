
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.tree import DecisionTreeClassifier


mnist = input_data.read_data_sets('data/',one_hot=True)
train_img = mnist.train.images 
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

tree=DecisionTreeClassifier(random_state=0)
tree.fit(train_img,train_label)

print("Accuracy on training set:{:.3f}".format(tree.score(train_img,train_label)))
print("Accuracy on test set:{:.3f}".format(tree.score(test_img,test_label)))



