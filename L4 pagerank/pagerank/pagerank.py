import networkx as nx

import numpy as np


# 创建有向图
G = nx.DiGraph() 
# 有向图之间边的关系
edges = [("A", "B"),("A","D"), ("A", "E"), ("A", "F"), ("B", "C"), ("C", "E"),("D","A"),("D","C"),("D","E"), ("E", "B"), ("E", "C"), ("F", "D")]

def createTransMatrix(edges):
	node=dict()
	innode=dict()
	outnode=dict()
	for edge in edges:
    		node[edge[0]]=1
    		node[edge[1]]=1
    		if edge[0] not in outnode:
        		outnode[edge[0]]=1
    		else:
        		outnode[edge[0]]+=1
    		if edge[1] not in innode:
        		innode[edge[1]]=1
    		else:
        		innode[edge[1]]+=1
    
#	print(node)
#	print(innode)
#	print(outnode)
#	print(len(node))
	a=np.zeros((len(node),len(node)))
#	print(a)

	map={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5}
#	print(map)

	for edge in edges:
    		a[map[edge[1]]][map[edge[0]]]=1/outnode[edge[0]] 
    
	b=np.zeros(len(node))
	for i in range(len(node)):
		b[i]=1/len(node)
	return a,b

def simplePageRank(a,b,num):
	w=b
	for i in range(num):
		w=np.dot(a,w)
	return w

def randomPageRank(a,b,num,d,n):
	w=b	
	for i in range(num):
		w=(1-d)/n+d*np.dot(a,w)
	return w

a,b=createTransMatrix(edges)
wi=simplePageRank(a,b,100)
wr=randomPageRank(a,b,100,0.85,6)



#print(a)
#print(b)
print("PageRank简化模型：",wi)
print("PageRank随机模型：",wr)



#for edge in edges:
#    G.add_edge(edge[0], edge[1])

#pagerank_list = nx.pagerank(G, alpha=1)
#print("pagerank值是：", pagerank_list)
#pagerank_list_random=nx.pagerank(G,alpha=0.85)
#print("pagerank值是：", pagerank_list_random)



