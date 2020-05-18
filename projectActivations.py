import numpy as np
import os
import pypcd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys 
from sklearn.neighbors import KDTree
from progress.bar import Bar

def load_pcd_data_binary(filename):
	pc=pypcd.PointCloud.from_path(filename)
	xyz = np.empty((pc.points, 3), dtype=np.float)
	rgb=np.empty((pc.points, 1), dtype=np.int)
	normals=np.empty((pc.points, 3), dtype=np.float)
	xyz[:, 0] = pc.pc_data['x']
	xyz[:, 1] = pc.pc_data['y']
	xyz[:, 2] = pc.pc_data['z']
	try:
		rgb = pc.pc_data['rgb']
	except Exception as e:
		error_msg=e
	try:
		normals[:,0]=pc.pc_data['normal_x']
		normals[:,1]=pc.pc_data['normal_y']
		normals[:,2]=pc.pc_data['normal_z']
	except Exception as e:
		error_msg=e

	return xyz,rgb,normals

def getTrainingScene(data_file,scene_file):
	if '.pcd' in scene_file or '.ply' in scene_file:
		if os.path.exists(data_file):
			data_file=data_file
	else:
		try_data_file=data_file+'.ply'
		if os.path.exists(try_data_file):
			#print(try_data_file)
			data_file=try_data_file
		#maybe pcd extension missing
		else:
			try_data_file=data_file+'.pcd'
			if os.path.exists(try_data_file):
				data_file=try_data_file
	# if scene_file not in unique_scenes:
	# 	unique_scenes[scene_file]=interaction
	# else:
	# 	continue
	if '.pcd' in data_file:
		cloud_training,_=load_pcd_data_binary(data_file)
	else:
		cloud_training,_=load_pcd_data_binary(data_file)
	return cloud_training

def load_pcd_data(filename,cols=(01,2),dataType=float):
	tmp_cloud=np.loadtxt(filename,skiprows=11,usecols=cols,dtype=dataType)
	return tmp_cloud

def projectActivations():
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	ids_target=np.nonzero(new_c>=samples)[0]

	path=os.path.abspath('/home/er13827/space/testing/tmp.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])
	# fig = plt.figure(figsize=(6, 6))
	# plt.ion()
	# ax = fig.add_subplot(111, projection='3d')

	for i in range(ids_target.size-1,-1,-1):
		interaction=ids_target[i]
		#recover activations
		name='../data/activations/OriginalActivations_'+str(interaction)+'.npy'
		activations=np.load(name)
		aff_dir=labels[interaction,0]
		query_object=labels[interaction,2]
		data_file=path[:pos]+"/"+aff_dir+"/ibs_full_"+labels[interaction,1]+"_"+query_object+".txt"
		print(data_file)
		with open(data_file) as f:
			content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content] 
		scene_file=content[0].split(":")[1]
		tmp=content[8].split(":")[1]
		datapoint=tmp.split(',')
		test_point=np.expand_dims(np.asarray([float(x) for x in datapoint]),axis=0)
		data_file=path[:pos]+"/"+aff_dir+"/"+scene_file
		#training_cloud=getTrainingScene(data_file,scene_file)
		#translate activations using test_point
		activations=activations+test_point
		field=path[:pos]+'/'+aff_dir+'/'+labels[interaction,1]+'_'+query_object+'_field_clean.pcd'
		#read original sample lines
		one_sample_file=path[:pos]+'/'+aff_dir+'/ibs_sample_512_'+labels[interaction,1]+'_'+query_object+'_better.pcd'
		sample_data=load_pcd_data(one_sample_file,cols=(0,1,2,3,4,5))
		#print(sample_data[-1,:])
		if not os.path.exists(field):
			print('Field not found %s'%(field))
			continue
		data,_,normals=load_pcd_data_binary(field)
		#search the NN in the field for every activation point		
		print('Building tree')
		kdt=KDTree(data,metric='euclidean')
		print('Searching for %d activations NN'%(activations.shape[0]))
		toExtract=np.zeros((activations.shape[0],1),dtype=np.int32)
		for k in range(activations.shape[0]):
			anActivation=np.expand_dims(activations[k,:],axis=0)
			_, ind = kdt.query(anActivation, k=1)
			toExtract[k,0]=ind[0,0]
		#remove repeated ids
		non_repeated=np.unique(toExtract)
		print('To extract %d, actually %d'%(toExtract.shape[0],non_repeated.shape[0]))
		new_sample=np.empty((non_repeated.shape[0]+2,activations.shape[1]))
		new_sample_normals=np.empty((non_repeated.shape[0]+2,activations.shape[1]))
		norm_mags=np.empty((non_repeated.shape[0],1))

		new_sample[:non_repeated.size,:]=data[non_repeated,:]
		new_sample_normals[:non_repeated.size,:]=normals[non_repeated,:]
		norm_mags[:,0]=np.linalg.norm(normals[non_repeated,:],axis=1)

		sorted_norms=np.argsort(norm_mags[:,0])
		new_sample=new_sample[sorted_norms,...]
		new_sample_normals=new_sample_normals[sorted_norms,...]
		# field_ids=np.arange(data.shape[0])
		# np.random.shuffle(field_ids)
		# field_ids=field_ids[:5000]
		#ax.scatter(training_cloud[:2048,0],training_cloud[:2048,1],training_cloud[:2048,2],s=1,c='r')
		# ax.scatter(new_sample[:,0],new_sample[:,1],new_sample[:,2],s=5,c='b')
		# ax.scatter(data[field_ids[:],0],data[field_ids[:],1],data[field_ids[:],2],s=1,c='c')
		# ax.scatter(activations[:,0],activations[:,1],activations[:,2],s=5,c='g')
		# plt.pause(1)
		# plt.draw()
		# ax.clear()
		new_sample[-2,:]=sample_data[-2,:3]
		new_sample[-1,:]=sample_data[-1,:3]
		new_sample_normals[-1,:]=sample_data[-1,3:]
		new_sample_normals[-2,:]=sample_data[-2,3:]
		actual_data_array=np.zeros(new_sample_normals.shape[0], dtype={'names':('x', 'y', 'z','v1','v2','v3'),
                          'formats':('f4', 'f4', 'f4','f4','f4','f4')})
		actual_data_array['x']=new_sample[:,0]
		actual_data_array['y']=new_sample[:,1]
		actual_data_array['z']=new_sample[:,2]
		actual_data_array['v1']=new_sample_normals[:,0]
		actual_data_array['v2']=new_sample_normals[:,1]
		actual_data_array['v3']=new_sample_normals[:,2]
		#new_cloud_data=np.concatenate((new_sample,new_sample_normals),axis=1)
		new_cloud = pypcd.PointCloud.from_array(actual_data_array)
		nname='../data/activations/ibs_sample_cnn_sample_'+labels[interaction,1]+'_'+labels[interaction,2]+'.pcd'
		new_cloud.save_pcd(nname,compression='ascii')
		del new_cloud
		activations_cloud=pypcd.make_xyz_point_cloud(activations)
		nname='../data/activations/activations_'+str(interaction)+'.pcd'
		activations_cloud.save_pcd(nname)
		print("Done")

def cleanTensors():
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	ids_target=np.nonzero(new_c>=samples)[0]
	path=os.path.abspath('/home/er13827/space/testing/tmp.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	fig = plt.figure(figsize=(8, 8))
	plt.ion()
	ax = fig.add_subplot(111, projection='3d')
	#ax = fig.add_subplot(111)
	for i in range(ids_target.size):
		interaction=ids_target[i]
		aff_dir=labels[interaction,0]
		query_object=labels[interaction,2]
		field=path[:pos]+'/'+aff_dir+'/'+labels[interaction,1]+'_'+query_object+'_field.pcd'
		if not os.path.exists(field):
			print('Field not found %s'%(field))
			continue
		data,_=load_pcd_data_binary(field)
		kdt=KDTree(data,metric='euclidean')
		clean_field_ids=np.zeros((data.shape[0],1),dtype=np.int32)-1
		bar = Bar('Cleaning',max=data.shape[0])
		field_ids=np.arange(data.shape[0])
		np.random.shuffle(field_ids)
		ax.scatter(data[:,0],data[:,1],data[:,2],'.',c='c')
		neighbors=np.zeros(clean_field_ids.shape,dtype=np.int32)
		for point_id in range(data.shape[0]):
			aPoint=np.expand_dims(data[point_id,:],axis=0)
			ind,dist =kdt.query_radius(aPoint,r=0.07,count_only =False,return_distance = True)
			ind=ind[0].reshape(1,-1)
			dist=dist[0].reshape(1,-1)
			# a=np.mean(dist[0,:])
			# b=np.median(dist[0,:])
			neighbors[point_id,0]=ind.shape[1]
			bar.next()
			#if dist.shape[1]<1000:
			# 	bar.next()
			# 	continue
			# else:
			# 	clean_field_ids[point_id,0]=point_id
			# 	bar.next()
				# print(ind.shape)
				# print(dist.shape)
				# field_ids=np.arange(dist.shape[1])
				# np.random.shuffle(field_ids)
				# if dist.shape[1]>10:
				# 	field_ids=field_ids[:10]
				# # print('%d/%d'%(point_id,data.shape[0]))
				# vecinity=ax.scatter(data[ind[0,field_ids],0],data[ind[0,field_ids],1],data[ind[0,field_ids],2],s=4,c='b')
				# sampledP=ax.scatter(aPoint[0,0],aPoint[0,1],aPoint[0,2],s=20,c='r')
				# title="%.4f,%.4f" %(a,b)
				# plt.title(title)
				# plt.pause(.1)
				# plt.draw()
				# vecinity.remove()
				# sampledP.remove()
		bar.finish()
		ok=False
		min_neighbors=500
		while not ok:
			good=np.nonzero(neighbors[:,0]<min_neighbors)[0]
			points=ax.scatter(data[good,0],data[good,1],data[good,2],s=10,c='r')
			plt.pause(10)
			plt.draw()
			#plt.show()
			min_neighbors=input('Min neighbors ')
			if min_neighbors<1:
				ok=True
			points.remove()


		#ax.hist(neighbors[:,0])
		#plt.draw()
		#plt.show()
		# actual_ids=np.nonzero(clean_field_ids[:,0]>=0)[0]
		# new_clean_field=data[clean_field_ids[actual_ids,0],:]
		# new_cloud = pypcd.make_xyz_point_cloud(new_clean_field)
		# nname='../data/activations/clean_field_'+str(interaction)+'.pcd'
		# new_cloud.save_pcd(nname)
def computeNewDescriptors():
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	ids_target=np.nonzero(new_c>=samples)[0]

	path=os.path.abspath('/home/er13827/space/testing/tmp.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])
	for i in range(ids_target.size):
		interaction=ids_target[i]
		new_sample_file='../data/activations/ibs_sample_cnn_sample_'+labels[interaction,1]+'_'+labels[interaction,2]+'.pcd'
		data_path='~/space/testing/'+labels[interaction,0]+'/'
		command='./processDescriptor '+new_sample_file+' '+data_path
		#print(command)
		os.system(command)

if __name__ == '__main__':
	#projectActivations()
	#cleanTensors()
	computeNewDescriptors()