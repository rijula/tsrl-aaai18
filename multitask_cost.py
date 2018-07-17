#import utilities
import os,sys
from sklearn.datasets import load_svmlight_file
from sklearn import random_projection
import numpy as np
import need
from numpy import linalg as LA
import operator
from scipy.sparse import vstack
from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix
#from sets import set
import hinge_full_n as model
import time
import collections
#path1="candidates/"
#path2="new-data-separated/"
#path3="test-data/"
path4="sign-hashed-new/"
path1="../preprocess/candidates/"
path2="../preprocess/new-data-separated/"
path3="../preprocess/test-data/"
def load_data(filename,test_file,mention):
	X_temp,y_temp=load_svmlight_file(test_file)
	X_w,y_w=load_svmlight_file(filename,n_features=16074140)
	col=X_w.shape[0]-X_temp.shape[0]
	#transformer = random_projection.SparseRandomProjection(100000)
	#transformer.fit(X_w)
	#X_new=transformer.transform(X_w)
	#print "dimension reduction done..."
	#print X_new.type()
	X_train=X_w[0:col-1]
	y_train=y_w[0:col-1]
	X_test=X_w[col:]
	y_test=y_w[col:]
	#dim=int(sys.argv[1])
	#X_train=utilities.create_hashed_features(X_train,mention,dim,utilities.hashfn,utilities.hashfn2)
	#X_test=utilities.create_hashed_features(X_test,mention,dim,utilities.hashfn,utilities.hashfn2)
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
	ys=set(y)
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
	ys=set(y)
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
	col=set()
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
'''def unique_col(X):
	col=set()
	for i in range(0,X.shape[0]):
		col.add(X.getrow(i).nonzero()[1])
	col_list=[]
	for x in col:
		col_list.append(x)'''
	
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
	sel_mention=need.load_pickle("new_reduced_test.p")
	
	mentions=sel_mention[int(sys.argv[1])]
	#mentions.remove('British-based')
	total_y=0
	lens=[]
	i=0
	for x in mentions:
		print x
		whole_file=path2+"whole-data-"+x+".dat"
		test_file=path3+x+"-test-data.dat"
		#X_train,y_train,X_test,y_test=load_data(whole_file,test_file,mentions[i])
		try:
			X_train=need.load_pickle(path4+x+"-train-x.dat")	
			y_train=need.load_pickle(path4+x+"-train-y.dat")
		except:
			exit(1)
		print "loading done..."
		new_y=sort_prior(x.lower(),y_train)
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
		i=i+1
	#need.save_pickle(X_train1,path2+"short_data-x.dat")
	#need.save_pickle(y_train1,path2+"short_data-y.dat")
	
	#x=need.load_pickle(path2+"short_data-x.dat")
	#full_y=need.load_pickle(path2+"short_data-y.dat")
	max_col= len(set(y_train1))
	print max_col
	
	y,classes=make_y(y_train1,max_col)
	need.save_pickle(classes,"classes/"+"ind_classes_"+str(sys.argv[1])+".p")	
	
	#classes=need.load_pickle("ind_classes_1.p")
	print "preprocessing done..."

	x=X_train1
	prev_norm=10000
	prev_cost=10000
	stop=int(sys.argv[2])
	flag=0
	for a in range(0,stop+30):
		print a
		one_time=time.time()
		count=0
		ind=0
		#lr=0.5
		lr=1/float(np.sqrt(a)+1)
		tot_cost=0
		full_gradient=np.zeros((x.shape[1],max_col))
		for m in mentions:
			print m
			#X_train=need.load_pickle(path4+mentions[i]+"-train-x.dat")	
			#y_train=need.load_pickle(path4+mentions[i]+"-train-y.dat")	
			whole_file=path2+"whole-data-"+m+".dat"
			test_file=path3+m+"-test-data.dat"
			#X_train,y_train,X_test,y_test=load_data(whole_file,test_file,mentions[i])
			#X_train=need.load_pickle(path4+mentions[i]+"-train-x.dat")
			y_train=need.load_pickle(path4+m+"-train-y.dat")
			if len(set(y_train))>1:
				new_y=sort_prior(m.lower(),y_train)
				#print y_train.shape[0],y_test.shape[0]
	
				print "loading done..."
		
				#x,y_temp=select_data_point(X_train,y_train,new_y)
				#print x.shape		
				#y = make_y_test(y_temp,57,classes)
				ind_list=[]
				for cl in new_y:
					if classes[cl] not in ind_list:
						ind_list.append(classes[cl])
				mini_batch_size=1000
			
				start=0
				#full_x=x[count:count+lens[ind]]
				#full_y=y[count:count+lens[ind]]
				#print full_x.shape
				#print count
				for j in range(0,int(lens[ind]/mini_batch_size)+1):
					start_time=time.time()
					if sum(lens[0:ind+1])-count < mini_batch_size:
						batch_x, batch_y = x[count:count+(sum(lens[0:ind+1])-count)],y[count:count+(sum(lens[0:ind+1])-count)]
						count=sum(lens[0:ind+1])
					else:
						batch_x, batch_y = x[count:count+mini_batch_size],y[count:count+mini_batch_size]
						count=count+mini_batch_size
					mask=np.zeros((batch_y.shape))
					for k in range(0,batch_y.shape[0]):
						for l in range(0,len(ind_list)):
							mask[k,ind_list[l]]=1
					col_list=unique_col(batch_x)
					#print col_list
					#print len(col_list)
					#print batch_x[:,col_list].shape
					model.cost=model.train(batch_x[:,col_list],batch_y[:,ind_list],col_list,ind_list,lr)	
					#full_cost = model.cost_print(full_x,full_y[:,ind_list],ind_list)
					
					#print model.cost
					#gradient=model.gradient_print(batch_x[:,col_list],batch_y[:,ind_list],col_list,ind_list)
					#for k in range(0,len(col_list)):
					#	for l in range(0,len(ind_list)):
					#		full_gradient[col_list[k],ind_list[l]]=full_gradient[col_list[k],ind_list[l]]+gradient[k,l]
	
					#tot_norm=tot_norm+np

					tot_cost=tot_cost+model.cost

				#print time.time()-start_time
			else:
				count=count+lens[ind]
			ind=ind+1
		if a==(stop+1):
			weights=model.weight_val()
			np.save("weights/"+str(sys.argv[1])+".npy",weights)
		if a>(stop+1):
			w_o=np.load("weights/"+str(sys.argv[1])+".npy")
			weights=model.weight_val()
			#print weights.shape
			w_n=(w_o+weights)/2
			#print w_n.shape
			np.save("weights/"+str(sys.argv[1])+".npy",w_n)
		if a-(stop+1)==31:
			break
		#gradient_norm=LA.norm(full_gradient)
		print tot_cost

		print "Iter time: ",time.time()-one_time
	fw1=open("results-news/"+str(sys.argv[1])+"-"+str(sys.argv[3])+".txt","w")
	for x in mentions:
		#print i
		print x
		whole_file=path2+"whole-data-"+x+".dat"
		test_file=path3+x+"-test-data.dat"
		#X_train,y_train,x_t,y_temp=load_data(whole_file,test_file,mentions[i])
		y_train=need.load_pickle(path4+x+"-train-y.dat")
		x_t=need.load_pickle(path4+x+"-test-x.dat")
		#print x_t.shape
		y_temp=need.load_pickle(path4+x+"-test-y.dat")
		ys=set(y_train).union(set(y_temp))
		new_y=sort_prior(x.lower(),y_train)
		ind_list=[]
		fw=open("multitask-results-full-testb/"+x+"-"+str(sys.argv[1])+"-"+str(sys.argv[3])+".txt","w")
		for y in new_y:
			#print y
			ind_list.append(classes[y])
		#print ind_list
		w_o=np.load("weights/"+str(sys.argv[1])+".npy")
		#print w_o.shape
		#indices=need.load_pickle("ind-classes.p")
		#print len(ind_list)
		if len(ys)>1:
			y_t=make_y_test(y_temp,max_col,classes)
			mask=np.zeros((y_t.shape))
			for j in range(0,y_t.shape[0]):
				for k in range(0,len(ind_list)):
					mask[j,ind_list[k]]=1
			col_list=unique_col(x_t)
			prediction1=model.predict(x_t[:,col_list],w_o,col_list,ind_list)
			#print prediction1
			prediction=[]
			for j in range(0,len(prediction1)):
				prediction.append(ind_list[prediction1[j]])
			print x_t.shape,len(ys)
			acc=np.mean(np.argmax(y_t, axis=1) == prediction)

			print acc
			fw1.write(x+" "+str(acc)+" "+str(x_t.shape[0])+'\n')
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
			fw1.write(x+" "+str(1.0)+" "+str(len(ys))+'\n')
			print x_t.shape,len(ys)
			print "1.0"
	os.remove("weights/"+str(sys.argv[1])+".npy")
if __name__=="__main__":
	main()
