

from sklearn.datasets import load_svmlight_file,dump_svmlight_file
import numpy as np
import time
import need
import os,sys
from sets import Set
import utilities
import scipy.sparse 
path1="candidates/"
path2="new-data-separated/"
path3="test-data/"

path4="sign-hashed/"
def load_data(filename,test_file,mention):
	
	X_temp,y_temp=load_svmlight_file(test_file)
	X_w,y_w=load_svmlight_file(filename,n_features=16074140)
	col=X_w.shape[0]-X_temp.shape[0]
	
	print X_w.shape
	dim=int(sys.argv[3])
	
	X_w1=utilities.create_hashed_features(X_w,mention,dim,utilities.hashfn,utilities.hashfn2)
	print "dimension reduction done.."
	X_train=X_w1[0:col-1]
	y_train=y_w[0:col-1]
	X_test=X_w1[col:]
	y_test=y_w[col:]
	return X_train,y_train,X_test,y_test
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
	print "col done..."
	return col_list
def main():
	mentions=np.load("short_list1.npy")
	for i in range(int(sys.argv[1]),int(sys.argv[2])):
		print i
		print mentions[i]
		whole_file=path2+"whole-data-"+mentions[i]+".dat"
		test_file=path3+mentions[i]+"-test-data.dat"
		start_time=time.time()
		X_train,y_train,X_test,y_test=load_data(whole_file,test_file,mentions[i])
		need.save_pickle(X_train,path4+mentions[i]+"-train-x.dat")
		need.save_pickle(y_train,path4+mentions[i]+"-train-y.dat")
		need.save_pickle(X_test,path4+mentions[i]+"-test-x.dat")
		#X_train,y_train=load_svmlight_file(path4+mentions[i]+"-train-x-1.dat")
		need.save_pickle(y_test,path4+mentions[i]+"-test-y.dat")
		#X_test,y_test=load_svmlight_file(path4+mentions[i]+"-test-x-1.dat")
		print X_train.shape
		print time.time()-start_time
if __name__=="__main__":
	main()
