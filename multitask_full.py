#import utilities
import os,sys
from sklearn.datasets import load_svmlight_file
from sklearn import random_projection
import numpy as np
import need
import operator
from scipy.sparse import vstack
from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix
from sets import Set
import log_full1 as model
import time
import collections
path1="candidates/"
path2="new-data-separated/"
path3="test-data/"
path4="sign-hashed/"

def load_data(filename,test_file,mention):
	X_temp,y_temp=load_svmlight_file(test_file)
	X_w,y_w=load_svmlight_file(filename,n_features=16074140)
	col=X_w.shape[0]-X_temp.shape[0]
	
	X_train=X_w[0:col-1]
	y_train=y_w[0:col-1]
	X_test=X_w[col:]
	y_test=y_w[col:]
	
	return X_train,y_train,X_test,y_test

def aggregate(X1,X2,y1,y2):
	X=vstack([X1,X2])
	y=np.concatenate((y1,y2),axis=0)
	return X,y
def aggregate_y(y1,y2):
	#X=vstack([X1,X2])
	y=np.concatenate((y1,y2),axis=0)
	return y
def sort_prior(mention,y):
	prior_dict=need.load_pickle(path1+mention+"-pind.p")
	#print prior_dict.keys()[0]
	ys=Set(y)
	#print len(ys)
	new_y=[]
	sorted_prior_dict = sorted(prior_dict.items(), key=operator.itemgetter(1),reverse=True)
	if len(ys)<=30:
		#print mention
		return ys
	i=0
	while len(new_y)<30:
		if int(sorted_prior_dict[i][0]) in ys:
			new_y.append(int(sorted_prior_dict[i][0]))
		i=i+1
	return new_y

def select_data_point(X,y,y1):
	cut_rows=[]
	
	for i in range(0,len(y)):
		if y[i] not in y1:
			cut_rows.append(i)
	new_y=np.delete(y,cut_rows,0)
	new_X=need.delete_row_lil(X.tolil(),cut_rows)
	return new_X,new_y
	
def make_y(y,max_cl):
	ys=Set(y)
	print len(ys)
	
	classes={}
	y_n=np.zeros((len(y),max_cl))
	ind=0
	for x in ys:
		classes[x]=ind
		ind=ind+1
	for i in range(0,len(y)):
		y_n[i,classes[y[i]]]=1
	return y_n,classes
def make_y_test(y,max_cl,classes):
	y_n=np.zeros((len(y),max_cl))
	ind=0	
	for i in range(0,len(y)):
		try:
			y_n[i,classes[y[i]]]=1
		except:
			y_n[i,0]=1
			print "Error for key: ",y[i]
	return y_n
def unique_col(X):
	col=Set()
	for i in range(0,X.shape[0]):
		row=X.getrow(i)
		for j in row.nonzero()[1]:
			col.add(j)
	col_list=[]
	for x in col:
		col_list.append(x)
	col_list=sorted(col_list)
	#print "col done..."
	return col_list

	
def mention_list():
	mentions=collections.defaultdict(list)
	mentions['1']=['New Zealand','Auckland']
	mentions['2']=['North','Korean','Korea','North Korea']
	mentions['3']=['South','Korean','Korea','South Korea','South Korean']
	mentions['4']=['Middle East','Middle','East','German','Wales']
	mentions['5']=['Jordan','Jordanian']
	mentions['6']=['Turkish','Turkey']
	mentions['7']=['Bulgarian','Bulgaria','SOFIA','Islam','League']
	mentions['8']=['British','Scottish','Britain','England','India','Roman']
	mentions['9']=['Europe','European']
	mentions['10']=['South Africa','South','South African','Africa','World Cup','Union']
	mentions['11']=['Scottish','Scotland','League']
	mentions['12']=['Bangladesh','Bangladeshi']
	mentions['13']=['Yugoslavia','Yugoslav','Serbia','Serb','BELGRADE','Hungarian','League']
	mentions['14']=['U.S.','National','U.S. Open','Union']
	mentions['15']=['Billy Mayfair','Mayfair']
	mentions['16']=['Democratic','Democrat','Democratic Convention']
	return mentions
def main():
	sel_mention=mention_list()
	mentions=sel_mention[str(sys.argv[1])]
	total_y=0
	lens=[]
	for i in range(0,len(mentions)):
		print i
		whole_file=path2+"whole-data-"+mentions[i]+".dat"
		test_file=path3+mentions[i]+"-test-data.dat"
		
		X_train=need.load_pickle(path4+mentions[i]+"-train-x.dat")	
		y_train=need.load_pickle(path4+mentions[i]+"-train-y.dat")
		print "loading done..."
		new_y=sort_prior(mentions[i].lower(),y_train)
		total_y=total_y+len(new_y)
		#print y_train.shape[0],y_test.shape[0]
		if i==0:
			X_train1,y_train1=select_data_point(X_train,y_train,new_y)
			print X_train1.shape
			lens.append(X_train1.shape[0])
		if i>=1:
			X_train2,y_train2=select_data_point(X_train,y_train,new_y)
			print X_train2.shape
			X_train1,y_train1=aggregate(X_train1,X_train2,y_train1,y_train2)
			lens.append(X_train2.shape[0])
	
	max_col= len(Set(y_train1))
	print max_col
	
	y,classes=make_y(y_train1,max_col)
	need.save_pickle(classes,"classes/"+"ind_classes_"+str(sys.argv[1])+".p")	
	
	#classes=need.load_pickle("ind_classes_1.p")
	print "preprocessing done..."

	x=X_train1
	
	for a in range(0,int(sys.argv[2])):
		print a
		one_time=time.time()
		count=0
		ind=0
		for i in range(0,len(mentions)):
			print mentions[i]
				
			whole_file=path2+"whole-data-"+mentions[i]+".dat"
			test_file=path3+mentions[i]+"-test-data.dat"
			
			y_train=need.load_pickle(path4+mentions[i]+"-train-y.dat")
			new_y=sort_prior(mentions[i].lower(),y_train)
			
	
			print "loading done..."
		
			
			ind_list=[]
			for cl in new_y:
				if classes[cl] not in ind_list:
					ind_list.append(classes[cl])
			mini_batch_size=1000
			
			start=0
			
			for j in range(0,int(lens[ind]/mini_batch_size)+1):
				start_time=time.time()
				if sum(lens[0:ind+1])-count < mini_batch_size:
					batch_x, batch_y = x[count:count+(sum(lens[0:ind+1])-count)],y[count:count+(sum(lens[0:ind+1])-count)]
					count=sum(lens[0:ind+1])
				else:
					batch_x, batch_y = x[count:count+mini_batch_size],y[count:count+mini_batch_size]
					count=count+mini_batch_size
				
				col_list=unique_col(batch_x)
				
				model.cost=model.train(batch_x[:,col_list],batch_y[:,ind_list],col_list,ind_list)	
				
			ind=ind+1
		print "Iter time: ",time.time()-one_time
	for i in range(0,len(mentions)):
		#print i
		print mentions[i]
		whole_file=path2+"whole-data-"+mentions[i]+".dat"
		test_file=path3+mentions[i]+"-test-data.dat"
		#X_train,y_train,x_t,y_temp=load_data(whole_file,test_file,mentions[i])
		y_train=need.load_pickle(path4+mentions[i]+"-train-y.dat")
		x_t=need.load_pickle(path4+mentions[i]+"-test-x.dat")
		#print x_t.shape
		y_temp=need.load_pickle(path4+mentions[i]+"-test-y.dat")
		ys=Set(y_train).union(Set(y_temp))
		new_y=sort_prior(mentions[i].lower(),y_train)
		ind_list=[]
		fw=open("multitask-results-full-1/"+mentions[i]+".txt","w")
		for x in new_y:
			ind_list.append(classes[x])
		#indices=need.load_pickle("ind-classes.p")
		#print len(ind_list)
		if len(ys)>1:
			y_t=make_y_test(y_temp,max_col,classes)
			mask=np.zeros((y_t.shape))
			for j in range(0,y_t.shape[0]):
				for k in range(0,len(ind_list)):
					mask[j,ind_list[k]]=1
			col_list=unique_col(x_t)
			prediction1=model.predict(x_t[:,col_list],col_list,ind_list)
			prediction=[]
			for j in range(0,len(prediction1)):
				prediction.append(ind_list[prediction1[j]])
			print x_t.shape,len(ys)
			print np.mean(np.argmax(y_t, axis=1) == prediction)
			test=(np.argmax(y_t,axis=1)==prediction)
			miss=[]
			for j in range(0,len(test)):
				if test[j]==False:
					miss.append(j)
			for j in range(0,len(test)):
				if j in miss:
					fw.write("prediction: "+str(prediction[j])+' ')
					fw.write("actual: "+ str(np.argmax(y_t,axis=1)[j])+'\n')
		else:
			print x_t.shape,len(ys)
			print "1.0"
if __name__=="__main__":
	main()
