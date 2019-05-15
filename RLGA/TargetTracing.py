#基于稀疏光流的目标追踪
#20190430
import cv2
import numpy as np 
import forwardTargetExtract as fd

def getOpticalFlow(find_old,find_new):
	opticalflow = []
	for i,(new,old) in enumerate(zip(find_new,find_old)):
		x1,y1 = old.ravel()
		x2,y2 = new.ravel()
		#对每个点计算L^2和tan
		#L = format((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1),'.2f')
		tan = format((y2-y1)/(x2-x1),'.2f')
		#x = format(x2-x1,'.2f')
		#y = format(y2-y1,'.2f')
		temp = [tan]
		#print(temp)
		opticalflow.append(temp)
	return opticalflow

def targetTrace(feature_params,lk_params):
	env = cv2.VideoCapture('2.mp4')
	success,frame = env.read()
	mask = np.zeros_like(frame)
	firstFrame = True
	counter = 0
	while success:
		if counter > 10:
			break
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		old_corner = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
		success,frame = env.read()
		if success == False:
			break
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)

		find_new = new_corner[trace_st==1]
		find_old = old_corner[trace_st==1]

		#前景目标粗提取
		find_old,find_new = fd.getStateVector(find_old,find_new)
		#前景目标精提取
		find_old,find_new = fd.calCenterAndDropMore(find_old,find_new)

		if firstFrame:
			last_optical = getOpticalFlow(find_old,find_new)
			firstFrame = False
		else:
			new_optical = getOpticalFlow(find_old,find_new)
			find_new,find_old = fd.intersectOpticalFlow(new_optical,last_optical,find_new,find_old)
			last_optical = getOpticalFlow(find_old,find_new)

		counter+=1

	cv2.destroyAllWindows()
	env.release()

	return find_new.shape[0]

def setParamAndGetResult(params):
	
	feature_params = dict(maxCorners = 1000,
		qualityLevel = params[0],
		minDistance = params[1],
		blockSize = 7)
	
	lk_params = dict(winSize = (int(params[2]),int(params[2])),
		maxLevel = int(params[3]),
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,int(params[4]),params[5]))

	env = cv2.VideoCapture('2.mp4')
	fps = env.get(cv2.CAP_PROP_FPS)  
	size = (int(env.get(cv2.CAP_PROP_FRAME_WIDTH)),   
	        int(env.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
	writer = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)  
	success,frame = env.read()
	mask = np.zeros_like(frame)
	firstFrame = True

	while success:
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		old_corner = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
		success,frame = env.read()
		if success == False:
			break
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)

		find_new = new_corner[trace_st==1]
		find_old = old_corner[trace_st==1]

		#前景目标粗提取
		find_old,find_new = fd.getStateVector(find_old,find_new)
		#前景目标精提取
		find_old,find_new = fd.calCenterAndDropMore(find_old,find_new)

		'''if firstFrame:
									last_optical = getOpticalFlow(find_old,find_new)
									firstFrame = False
								else:
									new_optical = getOpticalFlow(find_old,find_new)
									find_new,find_old = fd.intersectOpticalFlow(new_optical,last_optical,find_new,find_old)
									last_optical = getOpticalFlow(find_old,find_new)'''
								
		for i,(new,old) in enumerate(zip(find_new,find_old)):
			x1,y1 = old.ravel()
			x2,y2 = new.ravel()
			mask = cv2.line(mask,(x2,y2),(x1,y1),[0,0,255],1)
			frame = cv2.circle(frame,(x2,y2),5,[0,255,0],-1)
		

		img = cv2.add(frame,mask)
		cv2.imshow('frame',img)
		cv2.waitKey(30)
		#cv2.waitKey(1000/int(fps))
		writer.write(img)

	cv2.destroyAllWindows()
	env.release()