from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
import cPickle as pickle
import collections
import tensorflow as tf
def delete_row_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])
    return mat
def delete_row_lil(mat, i):
	if not isinstance(mat, lil_matrix):
		raise ValueError("works only for LIL format -- use .tolil() first")
	mat.rows = np.delete(mat.rows, i)
	mat.data = np.delete(mat.data, i)
	mat._shape = (mat._shape[0] - len(i), mat._shape[1])
	return mat.tocsr()
#row = np.array([0, 0, 1, 2, 2, 2])
#col = np.array([0, 2, 2, 0, 1, 2])
#data = np.array([1, 2, 3, 4, 5, 6])
#X=csr_matrix((data, (row, col)), shape=(3, 3))
#print X.todense()
#print delete_row_lil(X.tolil(),[0,2]).todense()
def load_pickle(filename):
    features=pickle.load(open(filename,"rb"))
    return features
def save_pickle(name,write_file):
    pickle.dump(name, open(write_file,"wb"),pickle.HIGHEST_PROTOCOL)
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
def make_features(X,Xf,n,f,st):
	for i in range(0,X.shape[0]):
		currrow=X.getrow(i)
		for j in currrow.nonzero()[1]:
			Xf[st,(n-1)*f+j]=X[i,j]
		st=st+1
	return Xf
def mention_list():
	mentions=collections.defaultdict(list)
	mentions['1']=['New Zealand','Auckland']
	mentions['2']=['North','Korean','Korea','North Korea']
	mentions['3']=['South','Korean','Korea','South Korean','South Korea']
	mentions['4']=['Middle East','Middle','East','German','Wales']
	mentions['5']=['Jordan','Jordanian']
	mentions['6']=['Turkish','Turkey']
	mentions['7']=['Bulgarian','Bulgaria','SOFIA','Islam','League']
	mentions['8']=['Scottish','Britain','England','India','Roman','British']
	mentions['9']=['Europe','European']
	mentions['10']=['South Africa','South','Africa','World Cup','Union','South African']
	mentions['11']=['Scottish','Scotland','League']
	mentions['12']=['Bangladesh','Bangladeshi']
	mentions['13']=['Yugoslavia','Yugoslav','Serbia','Serb','BELGRADE','Hungarian','League']
	mentions['14']=['U.S.','National','U.S. Open','Union']
	mentions['15']=['Billy Mayfair','Mayfair']
	mentions['16']=['Democratic','Democrat','Democratic Convention']
	return mentions
def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])
def map_ind_name(ind):
	prior_name = need.load_pickle("classes.p")
	for key in prior_name.keys():
		if prior_name[key]==str(ind):
			return key
