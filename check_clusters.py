import os,sys
import need
import collections
import operator
import csv
path1="../preprocess/candidates/"
path2="sign-hashed-w/"
def find_clusters(key,ent):
	mention=key.replace("test","")
	mentions_list=set()
	mentions_list.add(mention)
	for val1 in ent[key]:
		mentions_list.add(val1)
		
	return mentions_list
def sort_prior(mention,y):
	prior_dict=need.load_pickle(path1+mention.lower()+"-pind.p")
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
def print_dict(dname):
	for key in dname.keys():
		print key, ": ", dname[key]
def delete_or_not(mentions,ent,n):
	key_list=[]
	key_list_same=[]
	for key in ent.keys():
		if key!=n:
			if bool(mentions.difference(ent[key]))==False and len(mentions)>=len(ent[key]):
				#print key
				key_list.append(key)
	return key_list
def delete_or_not1(mentions,ent,n):
	key_list=[]
	key_list_same=[]
	for key in ent.keys():
		if key!=n:
			if bool(mentions.difference(ent[key]))==False and len(mentions)<=len(ent[key]):
				#print key
				#key_list.append(key)
				return key
	return -1
def map_ind_name(ind):
	prior_name = need.load_pickle("classes.p")
	for key in prior_name.keys():
		if prior_name[key]==str(ind):
			return key
def main1():
	sel_mention=need.load_pickle("reduced1-sum-test.p")
	test_keys=need.load_pickle("test-keys.p")
	agg_keys=collections.defaultdict(list)
	train_entities=collections.defaultdict(list)
	test_entities=collections.defaultdict(list)
	for key in sel_mention.keys():
		agg_keys[test_keys[key]]=sel_mention[key]
	need.save_pickle(agg_keys,"agg-keys.p")
	for key in sel_mention.keys():
		mentions=sel_mention[key]
		for x in mentions:
			try:
				y_train=need.load_pickle(path2+x+"-train-y.dat")
				new_y=sort_prior(x,y_train)
				y_test=need.load_pickle(path2+x+"-test-y.dat")
				test=set(y_test)
				for y in new_y:
					#print int(y)
					y=int(y)
					ins=map_ind_name(y)
					if ins not in train_entities[key]:
						train_entities[key].append(ins)
				for z in test:
					z=int(z)
					ins=map_ind_name(z)
					if ins not in test_entities[key]:
						test_entities[key].append(ins)
					
			except:
				print "error on, ",x
		#print train_entities[key]
	print "Train entities"
	#print_dict(train_entities)
	need.save_pickle(train_entities,"train-entities.p")
	print "Test entities"
	print_dict(test_entities)
	need.save_pickle(test_entities,"test-entities.p")
def main2():
	agg_keys=need.load_pickle("new-keys.p")
	cluster=need.load_pickle("reduced1-sum-test.p")
	train_entities=need.load_pickle("train-entities.p")
	#test_entities=need.load_pickle("test-entities.p")
	classes=need.load_pickle("classes.p")
	with open('tripartite-new.csv', 'w') as csvfile:
		fieldnames = ['cluster-id', 'cluster-mentions','entity']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		i=0
		flag=0
		for key in agg_keys.keys():
			print i
			sort_cl=sorted(cluster[key])
			for x in sort_cl:
				try:
					y_train=need.load_pickle(path2+x+"-train-y.dat")
					new_y=sort_prior(x,y_train)
					for y in train_entities[key]:
						if float(classes[y]) in new_y:
							writer.writerow({'cluster-id':'C'+str(i),'cluster-mentions':x,'entity':y})
				except:
					print "error on: ",key
					flag=1
			if flag==0:
				i=i+1
			else:
				flag=0
def main3():
	ent=need.load_pickle("new-mentions-sum-test.p")
	old_keys_m=collections.defaultdict(list)
	mentions_sum=collections.defaultdict(list)
	old_keys=need.load_pickle("new-mentions-sum-keys.p")

	duplicate_keys1=[]
	duplicate_keys2=[]
	j=0
	for key in ent.keys():
		if key not in duplicate_keys1:
			mentions_sum[j]=ent[key]
			old_keys_m[j].append(old_keys[key])
			key_list=delete_or_not(set(ent[key]),ent,key)
			for x in key_list:	
				duplicate_keys1.append(x)
				old_keys_m[j].append(old_keys[x])
			j=j+1
	'''for key in ent.keys():
		if key not in duplicate_keys2:
			mentions_sum[j]=ent[key]
			old_keys_m[j].append(old_keys[key])
			key_list1=delete_or_not1(set(ent[key]),ent,key)
			duplicate_keys2.append(key_list1)
	#		if key_list!=-1:
	#			old_keys_m[j].append(old_keys[key_list])
			j=j+1'''
	need.save_pickle(old_keys_m,"old-keys-m.p")
	need.save_pickle(mentions_sum,"new_reduced_test.p")
	print_dict(old_keys_m)
	#print_dict(mentions_sum)
	#print duplicate_keys
	#for x in duplicate_keys2:
	#	if x not in duplicate_keys1 and x !=-1:
	#		print x
def main():
	'''ent=need.load_pickle("mentions1-sum-test.p")
	test_keys=need.load_pickle("test-keys.p")
	new_test_keys=collections.defaultdict(list)
	mentions_sum=collections.defaultdict(list)
	#mentions_sum={}
	i=0
	j=0	
	duplicate_keys=[]
	for key in ent.keys():
		if key not in duplicate_keys:
			#print j,":",key
			#print key
			mentions_sum[j]=ent[key]
			new_test_keys[j].append(test_keys[key])
			key_list=delete_or_not(ent[key],ent,key)
			#print key_list
			for x in key_list:
				duplicate_keys.append(x)
				new_test_keys[j].append(test_keys[x])
			#print duplicate_keys
			j=j+1'''
	'''for key in ent.keys():
		if key.find("test")!=-1 and key.find("train")==-1:
			mentions_sum[i]=find_clusters(key,ent)
			#print i,":",key.replace("test","")
			i=i+1'''
		
	'''need.save_pickle(mentions_sum,"reduced1-sum-test.p")
	print_dict(mentions_sum)
	need.save_pickle(new_test_keys,"new-keys.p")'''
	#print_dict(new_test_keys)

	#print_dict(need.load_pickle("entities-overlap.p"))'''
	ent=need.load_pickle("new-entities-overlap.p")
	mentions_sum_train=collections.defaultdict(list)
	mentions_sum={}
	i=0
	#for key in ent.keys():
	#	if key.find("test")!=-1 and key.find("train")==-1:
			#print key
	#		ind=len(key)-4
			#print ind
			#mentions_sum[i]=key.replace("test","",ind)
	#		mentions_sum[i]=''.join(key.rsplit('test', 1))
			#print i,": ",mentions_sum[i]
	#		i=i+1
	for key in ent.keys():
		if key.find("test")!=-1:
			mentions_sum_train[i]=find_clusters(key,ent)	
			mentions_sum[i]=''.join(key.rsplit('test',1))
			i=i+1

	#unique_mentions=set()
	#for key in ent.keys():
	#	if key.find("train")!=-1:
	#		unique_mentions.add(key.replace("train",""))
	#		for val in ent[key]:
	#			unique_mentions.add(val)
	#print_dict(mentions_sum)
	need.save_pickle(mentions_sum,"new-mentions-sum-keys.p")

	#for x in unique_mentions:
	#	print x

	


if __name__=="__main__":
	main3()
