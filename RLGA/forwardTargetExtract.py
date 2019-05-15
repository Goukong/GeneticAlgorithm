#创建状态向量，利用状态向量进行前景目标的粗提取
import numpy as np 
import cv2

def getStateVector(find_old,find_new):
	#存储状态向量
	diagram = {}
	for i,(new,old) in enumerate(zip(find_new,find_old)):
		x1,y1 = old.ravel()
		x2,y2 = new.ravel()
		#对每个点计算L^2和tan
		L = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
		tan = (y2-y1)/(x2-x1)
		state = (L,tan)
		#让每个键值对应的值都是列表
		if diagram.__contains__(state):
			diagram[state].append(i)
		else:
			diagram[state]=[i]
	#找到包含最多的点的状态向量
	max_num = 0
	for i in diagram:
		num = len(diagram[i])
		if num > max_num:
			waitForDrop = diagram[i]
	#将是该状态向量的特征点都删除
	find_old = np.delete(find_old,waitForDrop,0)
	find_new = np.delete(find_new,waitForDrop,0)

	return find_old,find_new

#前景目标的精提取
def calCenterAndDropMore(find_old,find_new):
	#计算新特征点中的中心点位置
	xcenter = find_new[:,0].sum()/find_new.shape[0]
	ycenter = find_new[:,1].sum()/find_new.shape[0]
	#设定一个阀值T，每个特征点到中心点的距离，如果大于T，则舍弃
	while(1):
		waitForDrop = []
		#[100,120]效果比较好
		T = 130
		for i,point in enumerate(find_new):
			xpoint,ypoint = point.ravel()
			distance = pow(xpoint-xcenter,2) + pow(ypoint-ycenter,2)
			if distance > pow(T,2):
				waitForDrop.append(i)
		#没有需要淘汰的特征点时，循环结束
		if len(waitForDrop) == 0:
			break
		#去除不符合条件的特征点
		find_old = np.delete(find_old,waitForDrop,0)
		find_new = np.delete(find_new,waitForDrop,0)
	return find_old,find_new

#效果并不好！也许并不是特征点的交集，而是光流的交集
#numpy 判断 二维矩阵是否含有某个元素

def judgeIfContained(square,target):
	state = (square==target)
	#匹配是互相独立的，需要每一列都是True才可以
	state = state[:,0] & state[:,1]
	return (True in state)

#取特征点的交集,确定前景目标
def intersectFeaturePoints(new,last):
	waitForDrop = []
	for i,point in enumerate(new):
		if judgeIfContained(last,point):
			continue
		else:
			waitForDrop.append(i)
	new = np.delete(new,waitForDrop,0)
	return new


#取光流的交集，来确定特征点
def intersectOpticalFlow(new,last,find_new,find_old):
	waitForDrop = []
	for i,item in enumerate(new):
		if item not in last:
			waitForDrop.append(i)
	find_new = np.delete(find_new,waitForDrop,0)
	find_old = np.delete(find_old,waitForDrop,0)
	#print(new_feature)
	return find_new,find_old