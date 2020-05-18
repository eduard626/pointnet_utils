import numpy as np
from plyfile import (PlyData)
import sys
import os
from prep_affordanceData import (save_h5,load_h5,load_pcd_data,sample_cloud)
import glob
import pypcd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
from progress.bar import Bar

max_rad=0.806884
n_points=2048*2
n_samples=128
n_orientations=8


def getVoxel(seedPoint,rad,cloud):
	kdt = BallTree(cloud, leaf_size=5,metric='euclidean')
	#print('Extracting with rad %f'%rad)
	ind = kdt.query_radius(seedPoint.reshape(1,-1),r=rad)
	point_ids=np.expand_dims(ind,axis=0)[0,0].reshape(1,-1)
	print(point_ids.shape)
	#print(scene_cloud[point_ids[0,:],:].shape)
	return cloud[point_ids[0,:],:]

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
	Input:
    BxNx3 array, original batch of point clouds
    Return:
	BxNx3 array, rotated batch of point clouds"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		cosval = np.cos(rotation_angle)
    	sinval = np.sin(rotation_angle)
    	rotation_matrix = np.array([[cosval, -sinval,0],
                                    [sinval, cosval,0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data


def getMultiAffordanceData(file):
	path=os.path.abspath(file)
	pos=path.rfind('/')
	tokens=path[pos+1:].split('_')
	descriptor_id=tokens[6]
	scene_name=tokens[2]
	scene_name=path[:pos]+'/'+scene_name+'_d.pcd'
	file_descriptor=path[:pos]+'/tmp'+descriptor_id+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])
	fileId=tokens[-1]
	tokens=fileId.split('.')
	fileId=tokens[0]
	print(fileId)
	res_data_file=path[:pos]+'/'+fileId+'_goodPointsX.pcd'
	res_points_file=path[:pos]+'/'+fileId+'_goodPoints.pcd'

	data=load_pcd_data(res_data_file,cols=None)
	#print(data.shape)
	points=load_pcd_data(res_points_file,cols=(0,1,2))
	real_c_data=load_pcd_data(res_points_file,cols=(3,),dataType=np.uint32)
	#real_c_data=np.array(colors[:,-1],dtype=np.int32)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)

	real_c_data=np.concatenate((red,green,blue),axis=1)

	perPoint=np.sum(real_c_data,axis=1)
	bounds=np.cumsum(perPoint)


	howMany=np.zeros((labels.shape[0],1),dtype=np.int32)
	for i in range(labels.shape[0]):
		success=np.nonzero(data[:,0]==i)[0]
		howMany[i]=success.size
	ids_target=np.nonzero(howMany>n_samples)[0]
	print('Real found: %d'%ids_target.size)
	# re
	st_i=0
	end_i=bounds[0]
	print('Getting single affordance-instance per point')
	bar = Bar('Processing', max=bounds.shape[0])
	for i in range(bounds.shape[0]-1):
		if points[i,-1]>0.3:
			if i>0:
				st_i=bounds[i]
				end_i=bounds[i+1]
			someData=data[st_i:end_i,...]
			#get unique aff_ids
			ids=np.unique(someData[:,0])
			aff_ids=np.intersect1d(ids,ids_target)
			if aff_ids.shape[0]==0:
				continue
			toKeep=np.zeros((aff_ids.shape[0],7))
			for j in range(aff_ids.shape[0]):
				affData=np.nonzero(someData[:,0]==aff_ids[j])[0]
				keep=np.argmax(someData[affData,2])
				toKeep[j,:3]=points[i,...]
				toKeep[j,3:6]=someData[affData[keep],:3]
				toKeep[j,6]=i
			if i>0:
				newData=np.concatenate((newData,toKeep),axis=0)
			else:
				newData=toKeep
		bar.next()
	bar.finish()

	print('Recompute real targets')
	for i in range(labels.shape[0]):
		success=np.nonzero(newData[:,3]==i)[0]
		howMany[i]=success.size
	ids_target=np.nonzero(howMany>n_samples)[0]

	print('Real found: %d'%ids_target.size)
	ids_target=np.nonzero(howMany>n_samples)[0]
	print('Real found: %d'%ids_target.size)
	if n_orientations>1:
		name='AffordancesDataset_augmented_names.txt'
	else:
		name='AffordancesDataset_names.txt'
	aff_initials=[]
	with open(name, "w") as text_file:
		for i in range(ids_target.shape[0]):
			text_file.write("%d:%s-%s\n" % (i,labels[ids_target[i],0],labels[ids_target[i],2]))
			#aff_initials.append(labels[ids_target[i],0][0])
	#aff_initials=set(aff_initials)
	#print(aff_initials)
	#sys.exit()


	#Test 4 affordances case, where all instances of interaction account for single affordance classe
	aff_lims=np.array([0,8,17,91,92])
	#sample 128 points for every affordance, regardsless of their id
	sampled_ids=np.zeros((ids_target.size,n_samples))
	for i in range(ids_target.shape[0]):
		interesting_ids=np.nonzero(newData[:,3]==ids_target[i])[0]
		sorted_ids=np.argsort(newData[interesting_ids,5])
		sorted_ids=interesting_ids[sorted_ids[::-1]]
		sampled_ids[i,...]=newData[sorted_ids[:n_samples],-1]


	t=np.unique(sampled_ids.reshape(1,-1))
	dataPoints=np.zeros((t.size,3),dtype=np.float32)
	dataPoints_labels=np.zeros((t.size,5),dtype=np.uint8)
	initials=[]
	for i in range(t.size):
		#get all affordances for this point
		ids=np.nonzero(newData[:,-1]==t[i])[0]
		labels=np.zeros(ids.shape[0],dtype=np.uint8)
		for j in range(ids.shape[0]):
			labels[j]=np.nonzero(aff_lims>newData[ids[j],3])[0][0]
		labels=np.unique(labels)
		dataPoints[i]=newData[ids[0],:3]
		dataPoints_labels[i,labels]=1
		#extract voxel
	if n_orientations>1:
		name='dataPointsAffordances_augmented.h5'
	else:
		name='dataPointsAffordances.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,dataPoints,dataPoints_labels,'float32','uint8')

	#get dense cloud
	dense_sceneCloud=pypcd.PointCloud.from_path(scene_name).pc_data
	pc_array = np.array([[x, y, z] for x,y,z in dense_sceneCloud])

	#generate pointclouds that were not detected to test against single example training
	good_points_file=path[:pos]+'/'+fileId+'_goodPointsIds.pcd'
	sampled_points_file=path[:pos]+'/'+fileId+'_samplePointsIds.pcd'
	sampled_ids=np.sort(load_pcd_data(sampled_points_file,cols=(0,),dataType=np.int32))
	good_ids=np.sort(load_pcd_data(good_points_file,cols=(0,),dataType=np.int32))	
	non_affordance=np.setdiff1d(np.arange(sampled_ids.shape[0]),good_ids)
	sampled_points_file=path[:pos]+'/'+fileId+'_samplePoints.pcd'
	sampled_points=load_pcd_data(sampled_points_file,cols=(0,1,2))
	np.random.shuffle(non_affordance)
	print('Getting 1024 negative examples ')
	#shuffle negative examples ids
	bar = Bar('Processing', max=1024)
	negative_examples=np.zeros((1024,n_points,3),dtype=np.float32)
	for i in range(1024):
		point=pc_array[non_affordance[i],...]
		voxel=getVoxel(point,max_rad,pc_array)
		sample=sample_cloud(voxel,n_points)
		negative_examples[i,...]=sample
		bar.next()
	bar.finish()
	negative_labels=100*np.ones((1024,1),dtype=np.uint8)
	print('Got %d negative examples'%(negative_examples.shape[0]))
	print(negative_examples[0,0,:])
	name='AffordancesDataset_negatives.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,negative_examples,negative_labels,'float32','uint8')


	print('Sampling actual voxels from %s of %d points'%(scene_name,pc_array.shape[0]))
	dataSet_data=np.zeros((dataPoints.shape[0]*n_orientations,n_points,3),dtype=np.float32)
	dataSet_labels=np.zeros((dataPoints_labels.shape[0]*n_orientations,dataPoints_labels.shape[1]),dtype=np.uint8)
	print(dataSet_data.shape)
	count=0
	#data_type 0->centered
	data_type=1
	aff_names=np.array(['Non','Filling','Hanging','Placing','Sitting'])
	#extract voxels and pointclouds for dataset
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(False)
	for aff in range(dataPoints.shape[0]):
		t_names=np.nonzero(dataPoints_labels[aff])[0]
		print('%d/%d Training example for %s'%(aff,dataPoints.shape[0],np.array_str(aff_names[t_names])) )
		point=dataPoints[aff,:]
		#print(point.shape)
		voxel=getVoxel(point,max_rad,pc_array)
		if voxel.shape[0]<n_points:
			sample=aVoxel
		else:
			sample=sample_cloud(voxel,n_points)
		if data_type==0:
			centered_sample=sample-point
		else:
			centered_sample=sample
		#rotate this voxels n_orientations around Z (up)
		for j in range(n_orientations):
			rotated_voxel=rotate_point_cloud_by_angle(np.expand_dims(centered_sample,axis=0),j*2*np.pi/n_orientations).squeeze()
			dataSet_data[count,...]=rotated_voxel
			dataSet_labels[count,...]=dataPoints_labels[aff,...]
			count+=1
			if j==0:
				ax.scatter(rotated_voxel[:,0],rotated_voxel[:,1],rotated_voxel[:,2],s=3)
				plt.pause(0.2)
				plt.draw()			
	
	if n_orientations>1:
		name='AffordancesDataset_augmented.h5'
	else:
		name='AffordancesDataset.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,dataSet_data,dataSet_labels,'float32','uint8')

	
	return dataPoints,dataPoints_labels

def createMiniDatasetMulti(train_size,test_size,t_affordances=[0,1,2,3,4],positives_file='AffordancesDataset_augmented.h5',negatives_file='AffordancesDataset_negatives.h5',info_file='AffordancesDataset_augmented_names.txt'):
	# sample traing_size random for each class
	# check repeated
	the_affordances=np.expand_dims(np.asarray(t_affordances),0)
	names=np.genfromtxt(info_file,dtype='str',skip_header=0,delimiter=':')
	names=names[:,1]
	aff_initials=sorted(list(set([x[0] for x in names])))
	actual_initials=[]
	positive_data,positive_labels=load_h5(positives_file)
	negative_data,negative_labels=load_h5(negatives_file)
	for i in range(1,the_affordances.size):
		id_=the_affordances[0,i]
		thisIds=np.nonzero(positive_labels[:,id_])[0]
		print(thisIds.size)
		#select train and test
		np.random.shuffle(thisIds)
		train_ids=thisIds[:train_size]
		test_ids=thisIds[train_size:train_size+test_size]
		if i>1:
			#check for repeated
			new_=np.setdiff1d(train_ids,all_train_ids)
			all_train_ids=np.concatenate((all_train_ids,new_),axis=0)
			new_=np.setdiff1d(test_ids,all_test_ids)
			all_test_ids=np.concatenate((all_test_ids,new_),axis=0)
		else:
			all_train_ids=train_ids
			all_test_ids=test_ids
		actual_initials.append(aff_initials[id_-1])
	negative_ids_train=np.arange(train_size)
	negative_ids_test=np.arange(train_size,train_size+test_size)
	negative_labels_train=np.zeros((train_size,the_affordances.size))
	negative_labels_train[:,0]=1
	negative_labels_test=np.zeros((test_size,the_affordances.size))
	negative_labels_test[:,0]=1
	all_train_ids=all_train_ids.reshape(-1,1)
	all_test_ids=all_test_ids.reshape(-1,1)
	#print(all_train_ids.shape)

	train_data=np.concatenate((positive_data[all_train_ids.squeeze(),...],negative_data[negative_ids_train,...]),axis=0)
	train_labels=np.concatenate((positive_labels[all_train_ids,the_affordances],negative_labels_train),axis=0)
	#train_ids=np.arange(train_data.shape[0])
	#np.random.shuffle(train_ids)
	test_data=np.concatenate((positive_data[all_test_ids.squeeze(),...],negative_data[negative_ids_test,...]),axis=0)
	test_labels=np.concatenate((positive_labels[all_test_ids,the_affordances],negative_labels_test),axis=0)

	name='mini3_AffordancesDataset_train_'+''.join(actual_initials)+'_'+str(train_size)+'.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,train_data,train_labels,'float32','uint8')
	name='mini3_AffordancesDataset_test_'+''.join(actual_initials)+'_'+str(train_size)+'.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,test_data,test_labels,'float32','uint8')
	return train_data,train_labels,test_data,test_labels
	#print('Test data %d %d')

def createDataSet(file):
	path=os.path.abspath(file)
	pos=path.rfind('/')
	tokens=path[pos+1:].split('_')
	descriptor_id=tokens[6]
	scene_name=tokens[2]
	scene_name=path[:pos]+'/'+scene_name+'_d.pcd'
	file_descriptor=path[:pos]+'/tmp'+descriptor_id+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])
	fileId=tokens[-1]
	tokens=fileId.split('.')
	fileId=tokens[0]
	print(fileId)
	res_data_file=path[:pos]+'/'+fileId+'_goodPointsX.pcd'
	res_points_file=path[:pos]+'/'+fileId+'_goodPoints.pcd'

	data=load_pcd_data(res_data_file,cols=None)
	#print(data.shape)
	points=load_pcd_data(res_points_file,cols=(0,1,2))
	real_c_data=load_pcd_data(res_points_file,cols=(3,),dataType=np.uint32)
	#real_c_data=np.array(colors[:,-1],dtype=np.int32)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)

	real_c_data=np.concatenate((red,green,blue),axis=1)

	perPoint=np.sum(real_c_data,axis=1)
	bounds=np.cumsum(perPoint)
	#print(bounds)
	howMany=np.zeros((labels.shape[0],1),dtype=np.int32)
	all_data=np.zeros((data.shape[0],6))

	for i in range(all_data.shape[0]):
		point_id=np.nonzero(bounds>i)[0][0]
		all_data[i,:3]=points[point_id,:]
		all_data[i,3:]=data[i,:3]


	for i in range(labels.shape[0]):
		success=np.nonzero(all_data[:,3]==i)[0]
		success2=np.nonzero(all_data[success,2]>0.3)[0]
		howMany[i]=success2.size

	ids_target=np.nonzero(howMany>n_samples)[0]
	print('Real found: %d'%ids_target.size)
	if n_orientations>1:
		name='AffordancesDataset_augmented_names.txt'
	else:
		name='AffordancesDataset_names.txt'
	with open(name, "w") as text_file:
		for i in range(ids_target.shape[0]):
			text_file.write("%d:%s-%s\n" % (i,labels[ids_target[i],0],labels[ids_target[i],2]))
	#
	#print(labels[ids_target,1:])

	all_points=np.zeros((ids_target.size,n_samples,3))
	all_points_score=np.zeros((ids_target.size,n_samples))
	for i in range(ids_target.shape[0]):
		#get the 3D point for the response
		success=np.nonzero((all_data[:,3]==ids_target[i])&(all_data[:,2]>0.3))[0]
		sorted_ids=np.argsort(all_data[success,5])
		print('Sampling for %s %s in %d points(%f,%f)'%(labels[ids_target[i],0],labels[ids_target[i],2],success.size,np.max(all_data[success,5]),np.min(all_data[success,5])))
		sorted_ids=sorted_ids[::-1]
		for j in range(n_samples):
			all_points[i,j,:]=all_data[success[sorted_ids[j]],:3]
			all_points_score[i,j]=all_data[success[sorted_ids[j]],5]
		#print('Min %f max %f'%(all_points_score[i,0],all_points_score[i,-1]))
	labels_d=np.arange(ids_target.size)
	print('Sampled points maxZ %f minZ %f'%(np.max(all_points[:,:,2].reshape(1,-1)),np.min(all_points[:,:,2].reshape(1,-1))) )

	#sys.exit()

	if n_orientations>1:
		name='dataPointsAffordances_augmented.h5'
	else:
		name='dataPointsAffordances.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,all_points,labels_d,'float32','uint8')


	#get dense cloud
	dense_sceneCloud=pypcd.PointCloud.from_path(scene_name).pc_data
	pc_array = np.array([[x, y, z] for x,y,z in dense_sceneCloud])

	#generate pointclouds that were not detected to test against single example training
	good_points_file=path[:pos]+'/'+fileId+'_goodPointsIds.pcd'
	sampled_points_file=path[:pos]+'/'+fileId+'_samplePointsIds.pcd'
	sampled_ids=np.sort(load_pcd_data(sampled_points_file,cols=(0,),dataType=np.int32))
	good_ids=np.sort(load_pcd_data(good_points_file,cols=(0,),dataType=np.int32))	
	non_affordance=np.setdiff1d(np.arange(sampled_ids.shape[0]),good_ids)
	sampled_points_file=path[:pos]+'/'+fileId+'_samplePoints.pcd'
	sampled_points=load_pcd_data(sampled_points_file,cols=(0,1,2))
	np.random.shuffle(non_affordance)
	print('Getting 1024 negative examples ')
	#shuffle negative examples ids
	bar = Bar('Processing', max=1024)
	negative_examples=np.zeros((1024,n_points,3),dtype=np.float32)
	for i in range(1024):
		point=pc_array[non_affordance[i],...]
		voxel=getVoxel(point,max_rad,pc_array)
		minP=np.min(voxel,0);
		maxP=np.max(voxel,0);
		dist=np.linalg.norm(maxP-minP,axis=0)/2
		print('RAD %f rad %f estimation %f'%(dist,max_rad,max_rad*np.sqrt(3)))
		sample=sample_cloud(voxel,n_points)
		negative_examples[i,...]=sample
		bar.next()
	bar.finish()
	negative_labels=100*np.ones((1024,1),dtype=np.uint8)
	print('Got %d negative examples'%(negative_examples.shape[0]))
	print(negative_examples[0,0,:])
	name='AffordancesDataset_negatives.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,negative_examples,negative_labels,'float32','uint8')
	#sys.exit()


	print('Sampling actual voxels from %s of %d points'%(scene_name,pc_array.shape[0]))
	dataSet_data=np.zeros((all_points.shape[0]*all_points.shape[1]*n_orientations,n_points,3),dtype=np.float32)
	dataSet_labels=np.zeros((all_points.shape[0]*all_points.shape[1]*n_orientations,1),dtype=np.uint8)
	print(dataSet_data.shape)
	count=0
	#data_type 0->centered
	data_type=1
	#extract voxels and pointclouds for dataset
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(False)
	for aff in range(all_points.shape[0]):
		print('Training examples for %s %s'%(labels[ids_target[aff],0],labels[ids_target[aff],2]))
		bar = Bar('Processing', max=all_points.shape[1])
		for n_sample in range(all_points.shape[1]):
			point=all_points[aff,n_sample,:].reshape(3,-1)
			#print(point.shape)
			voxel=getVoxel(point,max_rad,pc_array)
			if voxel.shape[0]<n_points:
				sample=aVoxel
			else:
				sample=sample_cloud(voxel,n_points)
			if data_type==0:
				centered_sample=sample-point
			else:
				centered_sample=sample
			#rotate this voxels n_orientations around Z (up)
			for j in range(n_orientations):
				rotated_voxel=rotate_point_cloud_by_angle(np.expand_dims(centered_sample,axis=0),j*2*np.pi/n_orientations).squeeze()
				dataSet_data[count,...]=rotated_voxel
				dataSet_labels[count]=labels_d[aff]
				count+=1
			if n_sample==0:
				ax.scatter(rotated_voxel[:,0],rotated_voxel[:,1],rotated_voxel[:,2],s=3)
				plt.pause(0.2)
				plt.draw()
			bar.next()
		bar.finish()
	if n_orientations>1:
		name='AffordancesDataset_augmented.h5'
	else:
		name='AffordancesDataset.h5'
	if os.path.exists(name):
		os.system('rm %s' % (name))
	save_h5(name,dataSet_data,dataSet_labels,'float32','uint8')

def getDataset(file):
	#split dataset into smaller batches/files
	all_data,all_labels=load_h5(file)
	#shuffle them to add 'randomness'
	all_ids=np.arange(all_data.shape[0])
	np.random.shuffle(all_ids)
	all_data=all_data[all_ids,...]
	all_labels=all_labels[all_ids]
	print(all_data.shape)
	print(all_labels.shape)
	n_splits=all_labels.shape[0]/(496*4)
	print(n_splits)
	for i in range(n_splits):
		name='AffordancesDataset_file'+str(i)+'.h5'
		start_id=i*(496*4)
		end_id=(i+1)*(496*4)
		toSaveData=all_data[start_id:end_id,...]
		toSaveLabels=all_labels[start_id:end_id]
		print('%s %d %d'%(name,start_id,end_id))
		if os.path.exists(name):
			os.system('rm %s' % (name))
		save_h5(name,toSaveData,toSaveLabels,'float32','uint8')




def getMiniDataset(class_ids,train_size,test_size,file='AffordancesDataset_augmented.h5',negatives_file='AffordancesDataset_negatives.h5',return_data=False,info_file='AffordancesDataset_augmented_names.txt'):
	#if return data is true then no data is saved
	# and data/labels are returned to caller
	
	names=np.genfromtxt(info_file,dtype='str',skip_header=0,delimiter=':')
	#print(names)
	real_ids=np.array([int(x) for x in names[:,0]])
	#print(real_ids)
	all_data,all_labels=load_h5(file)
	#print(np.unique(all_labels))
	if (test_size+train_size)>all_labels.shape[0]:
		print('Max data size is '%all_labels.shape[0])
		sys.exit()
	if test_size<0:
		test_size=all_labels.shape[0]-train_size

	#print(all_data.shape)
	train_ids=np.zeros((class_ids.shape[0]*train_size,1),dtype=np.int32)
	test_ids=np.zeros((class_ids.shape[0]*test_size,1),dtype=np.int32)
	#some_ids_new=np.zeros((class_ids.shape[0],1),dtype=np.uint8)
	new_labels_train=np.zeros((class_ids.shape[0]*train_size,1),dtype=np.uint8)
	new_labels_test=np.zeros((class_ids.shape[0]*test_size,1),dtype=np.uint8)
	aff_initial=[]
	for i in range(class_ids.shape[0]):
		ids=np.nonzero(all_labels==class_ids[i])[0]
		#print(all_labels[ids])
		#take 32 from each class to test
		test=np.arange(ids.shape[0],dtype=np.int32)
		np.random.shuffle(test)
		start_id=i*train_size
		end_id=(i+1)*train_size
		train_ids[start_id:end_id,0]=ids[test[:train_size]]
		new_labels_train[start_id:end_id,0]=i+1
		start_id=i*test_size
		end_id=(i+1)*test_size
		test_ids[start_id:end_id,0]=ids[test[train_size:train_size+test_size]]
		new_labels_test[start_id:end_id,0]=i+1
		aff_initial.append(names[class_ids[i],1][0])
	print(aff_initial)
	#print(ids_train)
	#train_ids=np.asarray(ids_train,dtype=np.uint8).reshape(-1,1)
	train_ids=np.squeeze(train_ids)
	test_ids=np.squeeze(test_ids)
	#print(train_ids.T)
	#print(test_ids.T)
	#sys.exit()
	#test_ids=np.squeeze(np.asarray(ids_test,dtype=np.uint8).reshape(-1,1))
	print('Training set %d'%train_ids.shape[0])
	print('Testing set %d'%test_ids.shape[0])
	new_data_train=all_data[train_ids,...]
	new_data_test=all_data[test_ids,...]
	#concatenate here the negatives
	negative_data,negative_labels=load_h5(negatives_file)
	new_data_train=np.concatenate((new_data_train,negative_data[:train_size]),axis=0)
	new_labels_train=np.concatenate((new_labels_train,np.zeros((train_size,1))),axis=0)

	train_shuffle=np.arange(new_data_train.shape[0])
	np.random.shuffle(train_shuffle)
	new_data_train=new_data_train[train_shuffle,...]
	new_labels_train=new_labels_train[train_shuffle]


	
	name='mini_AffordancesDataset_train_'+''.join(aff_initial)+'_'+str(train_size)+'.h5'
	if not return_data:
		if os.path.exists(name):
				os.system('rm %s' % (name))
		save_h5(name,new_data_train,new_labels_train,'float32','uint8')

	new_data_test=np.concatenate((new_data_test,negative_data[train_size:train_size+test_size]),axis=0)
	new_labels_test=np.concatenate((new_labels_test,np.zeros((test_size,1))),axis=0)

	train_shuffle=np.arange(new_data_test.shape[0])
	np.random.shuffle(train_shuffle)
	new_data_test=new_data_test[train_shuffle,...]
	new_labels_test=new_labels_test[train_shuffle]

	print('Training data ')
	print(new_data_train.shape)
	print(new_labels_train)
	print('Test data ')
	print(new_data_test.shape)
	print(new_labels_test.shape)

	name='mini_AffordancesDataset_test_'+''.join(aff_initial)+'_'+str(train_size)+'.h5'
	if not return_data:
		if os.path.exists(name):
				os.system('rm %s' % (name))
		save_h5(name,new_data_test,new_labels_test,'float32','uint8')
		# save the original class ids to keep track of the affordances involved in this dataset
		name='mini_AffordancesDataset_names_'+''.join(aff_initial)+'_'+str(train_size)+'.txt'
		with open(name, "w") as text_file:
			for i in range(class_ids.shape[0]):
				print('%d:%s' % (i+1,names[class_ids[i],1]))
				text_file.write("%d:%s\n" % (i+1,names[class_ids[i],1]))
	else:
		for i in range(class_ids.shape[0]):
			print('%d:%s' % (i+1,names[class_ids[i],1]))


	'''fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(False)
	for i in range(new_labels_test.shape[0]):
		ax.scatter(new_data_test[i,:,0],new_data_test[i,:,1],new_data_test[i,:,2],s=10)
		#print(names[class_ids[new_labels_test[i,0]],1])
		ax.set_title(names[class_ids[new_labels_test[i,0]],1]+' '+str(new_labels_test[i,0]))
		plt.pause(5)
		plt.draw()'''
	if return_data:
		return new_data_train,new_labels_train,new_data_test,new_labels_test
	else:
		return 0,0,0,0

def createMiniDatasets(train_size,test_size,positives_file='AffordancesDataset_augmented.h5',negatives_file='AffordancesDataset_negatives.h5',info_file='AffordancesDataset_augmented_names.txt',target_affordance='Filling'):
	# This function creates binary datasets for every affordance in the csv file
	# train_size and test_size are per class
	positive_data,_=load_h5(positives_file)
	print(positive_data.shape)
	negative_data,negative_labels=load_h5(negatives_file)
	if train_size>negative_data.shape[0] or test_size>negative_data.shape[0]:
		print('Number of examples exceeded')
		sys.exit()
	info=np.genfromtxt(info_file,dtype='str',skip_header=0,delimiter=':')
	real_ids=np.array([int(x) for x in info[:,0]])
	bar = Bar('Processing', max=real_ids.shape[0])
	# if need all binary datasets, make target_affordance an empty string
	#target_affordance=''
	count=1
	if target_affordance:
		print('Getting data for %s'%(target_affordance))
	else:
		print('Getting all data ')
	data_train=np.array([],dtype=np.float32).reshape(0,n_points,3)
	data_test=np.array([],dtype=np.float32).reshape(0,n_points,3)
	labels_train=np.array([],dtype=np.uint8).reshape(0,1)
	labels_test=np.array([],dtype=np.uint8).reshape(0,1)
	for j in range(real_ids.shape[0]):
		current_aff=info[j,1]
		if target_affordance:
			if target_affordance not in current_aff:
				continue
		# this file is supposed to have 128 examples per affordance x 8 orientations
		start_i=j*(128*8)
		end_i=(j+1)*(128*8)
		thisAffordance_data=positive_data[start_i:end_i,...]
		train_ids=np.random.randint(thisAffordance_data.shape[0],size=train_size)
		test_ids=np.setdiff1d(np.arange(thisAffordance_data.shape[0]),train_ids)

		test_ids=test_ids[:test_size]		

		#save training data
		sample_negative=np.arange(negative_data.shape[0])
		np.random.shuffle(sample_negative)
		data=np.concatenate((thisAffordance_data[train_ids,...],negative_data[sample_negative[:train_size],...]),axis=0)
		labels=np.concatenate((np.ones((train_size,1)),np.zeros((train_size,1))),axis=0)
		if target_affordance:
			#concat tmp data with training data
			data_train=np.concatenate((data,data_train),axis=0)
			labels_train=np.concatenate((count*labels,labels_train),axis=0)
		else:
			data_train=data
			labels_train=labels
		#shuffle the data
		shuffle_ids=np.arange(labels_train.shape[0])
		np.random.shuffle(shuffle_ids)
		data_train=data_train[shuffle_ids,...]
		labels_train=labels_train[shuffle_ids]
		if not target_affordance:
			name='binary_AffordancesDataset_train'+str(j)+'_'+str(train_size)+'.h5'
			if os.path.exists(name):
				os.system('rm %s'%(name))
			save_h5(name,data_train,labels_train,'float32','uint8')


		# save test data
		data=np.concatenate((thisAffordance_data[test_ids,...],negative_data[sample_negative[train_size:train_size+test_size],...]),axis=0)
		#print(thisAffordance_data[test_ids,...].shape[0])
		labels=np.concatenate((np.ones((test_size,1)),np.zeros((test_size,1))),axis=0)
		if target_affordance:
			data_test=np.concatenate((data,data_test),axis=0)
			labels_test=np.concatenate((count*labels,labels_test),axis=0)
			#count+=1
		else:
			data_test=data
			labels_test=labels
		shuffle_ids=np.arange(labels_test.shape[0])
		np.random.shuffle(shuffle_ids)
		data_test=data_test[shuffle_ids,...]
		labels_test=labels_test[shuffle_ids]
		if not target_affordance:
			name='binary_AffordancesDataset_test'+str(j)+'_'+str(train_size)+'.h5'
			if os.path.exists(name):
				os.system('rm %s'%(name))
			save_h5(name,data_test,labels_test,'float32','uint8')
		bar.next()
	bar.finish()
	if target_affordance:
		print('Saving test data for %s '%(target_affordance))
		# before saving, remove unbalance in negatives
		# since there will be X (affordances) times more negatives
		'''ids_to_remove=np.nonzero(labels_test==0)[0]
		ids_to_remove=ids_to_remove[test_size:]
		ids_to_keep=np.setdiff1d(np.arange(labels_test.shape[0]),ids_to_remove)
		data_test=data_test[ids_to_keep,...]
		labels_test=labels_test[ids_to_keep]'''

		#Same for positives
		print(data_test.shape)
		print(labels_test.shape)
		name='miniAffordancesDataset_test_'+target_affordance+'_'+str(train_size)+'.h5'
		if os.path.exists(name):
			os.system('rm %s'%(name))
		save_h5(name,data_test,labels_test,'float32','uint8')
		name='miniAffordancesDataset_train_'+target_affordance+'_'+str(train_size)+'.h5'
		print('Saving train data for %s '%(target_affordance))
		'''ids_to_remove=np.nonzero(labels_train==0)[0]
		ids_to_remove=ids_to_remove[train_size:]
		ids_to_keep=np.setdiff1d(np.arange(labels_train.shape[0]),ids_to_remove)
		data_train=data_train[ids_to_keep,...]
		labels_train=labels_train[ids_to_keep]'''
		print(data_train.shape)
		print(labels_train.shape)
		if os.path.exists(name):
			os.system('rm %s'%(name))
		save_h5(name,data_train,labels_train,'float32','uint8')

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2,len(s)+1))


if __name__ == "__main__":
	if len(sys.argv)<2:
		print "Need results Affordance_object_id.pcd"
		sys.exit()
	if ".pcd" not in sys.argv[1]:
		print "Need pcd file"
		sys.exit()
	from itertools import *
	createDataSet(sys.argv[1])
	#getDataset('AffordancesDataset.h5')
	'''targets=np.array([0,8,15,62],dtype=np.uint8)
	nt=list(powerset(targets))
	print(nt)
	for i in range(len(nt)):
		print('==================')
		print(nt[i])
		targets=np.asarray(nt[i])
		getMiniDataset(targets.T,128,512)'''

	#getMultiAffordanceData(sys.argv[1])
	#targets=np.arange(4)
	#targets=np.array([0,15],dtype=np.uint8)
	#getMiniDataset(targets.T,1,512)
	#createMiniDatasets(1,512,target_affordance='Placing')
    