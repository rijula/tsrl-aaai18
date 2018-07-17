import utilities
import os,sys
from sklearn.datasets import load_svmlight_file
import numpy as np
import need
import operator
from scipy.sparse import vstack
from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix
from sets import Set
#import real_theano as model
import time
import collections

path1="../preprocess/candidates/"
path2="../preprocess/new-data-separated/"
path3="../preprocess/test-data-new/"
path4="sign-hashed-new/"
def load_data(filename,test_file,mention):
	X_temp,y_temp=load_svmlight_file(test_file)
	X_w,y_w=load_svmlight_file(filename,n_features=16074140)
	col=X_w.shape[0]-X_temp.shape[0]
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

#for a mention sort_prior return a list of top 30 prior wise sorted entity
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
#for a mention training data select_data_point returns training data examples after removing the entity examples which entities are not in top 30 list
def select_data_point(X,y,y1):
	cut_rows=[]
	
	for i in range(0,len(y)):
		if y[i] not in y1:
			cut_rows.append(i)
	new_y=np.delete(y,cut_rows,0)
	new_X=need.delete_row_lil(X.tolil(),cut_rows)
	return new_X,new_y
# make_y returns one hot representation of the class-labels of train data
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
# make_y returns one hot representation of the class-labels of test data
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
# entity_overlap checks for mentions which has overlaps in candidate_entity set. It takes a mention's candidate entity set and check whether the set has any overlap with other mentions' entity set and saving it in a entity_ov dictionary. 
def entity_overlap(other_mentions,mention,entity_ov):
	entities=need.load_pickle("entities.p")
	test_count=0
	train_count=0
	mention_overlap=[]
	train_mention_overlap=[]
	for i in range(0,len(other_mentions)):
		test_overlap=Set(entities[mention+'test']).intersection(Set(entities[other_mentions[i]+'train']))
		train_overlap=Set(entities[mention+'train']).intersection(Set(entities[other_mentions[i]+'train']))
		if len(test_overlap)>0:
			test_count=test_count+1
			entity_ov[mention+'test'].append(other_mentions[i])
			mention_overlap.append(other_mentions[i])
		if len(train_overlap)>0:
			train_count=train_count+1
			#train_mention_overlap.append(other_mentions[i])
			entity_ov[mention+'train'].append(other_mentions[i])
			train_mention_overlap.append(other_mentions[i])
	#print mention_overlap
	#print train_mention_overlap
	#need.save_pickle(entity_overlap,"entity-overlap.p")
	return test_count,train_count

#taking the testb mentions' list and for each mention training and testing entities and call entity_overlap to find between the mentions and saving the overlap dictionary in "new-entities-overlap.p"
def main():
	#mentions=np.load("../preprocess/short_list1.npy")
	#mentions=mentions[0:210]
	f=open("test-ch.txt","r")
	entities=collections.defaultdict(list)
	mentions=[]
	for line in f:
		line=line.strip()
		line=line.split()
		mention=line[0]
		
		for i in range(1,len(line)-1):
			mention=mention+" "+line[i]
		mentions.append(mention)
	total_y=0
	cou_ov=0
	cou_trov=0
	#entities=collections.defaultdict(list)
	entity_ov=collections.defaultdict(list)
	for i in range(0,len(mentions)):
		print i
		print mentions[i]
		train_file=path4+mentions[i]+"-train-y.dat"
		test_file=path3+mentions[i]+"-test-data.dat"
		#X_train,y_train,X_test,y_test=load_data(whole_file,test_file,mentions[i])
		'''try:
			y_train=need.load_pickle(train_file)
			X_test,y_test=load_svmlight_file(test_file)
			print "loading done..."
			new_y=sort_prior(mentions[i].lower(),y_train)
			entities[mentions[i]+'train'] = new_y
			entities[mentions[i]+'test'] = y_test
		except:

			entities[mentions[i]+'train'] = []
			entities[mentions[i]+'test'] = []'''
		#print "Training set: ", new_y
		#print "Testing set: ", Set(y_test)
		other_mentions=np.delete(mentions,i)
		ovcount,trovcount=entity_overlap(other_mentions,mentions[i],entity_ov)
		if ovcount >0:
			cou_ov=cou_ov+1
			print "ovcount is greater than zero: " ,mentions[i]
		if trovcount >0:
			cou_trov=cou_trov+1
			print "trovcount is greater than zero: ",mentions[i]
		print len(mentions)
	print "cou_ov :",cou_ov
	print "cou_trov :",cou_trov
	need.save_pickle(entity_ov,"new-entities-overlap.p")
	#need.save_pickle(entities,"entities.p")
if __name__=="__main__":
	main()
