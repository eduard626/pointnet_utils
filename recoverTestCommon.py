import numpy as np
import os
import sys
import pypcd
import glob
from sklearn.neighbors import KDTree
from progress.bar import Bar
import h5py

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
def sample_cloud(data,n_samples):
	idx = np.arange(data.shape[0])
	np.random.shuffle(idx)
	return data[idx[0:n_samples], :]

def getVoxel(seedPoint,rad,tree):
	#print('Extracting with rad %f'%rad)
	ind = tree.query_radius(seedPoint.reshape(1,-1),r=rad)
	point_ids=np.expand_dims(ind,axis=0)[0,0].reshape(1,-1)
	#print(point_ids.shape)
	#print(scene_cloud[point_ids[0,:],:].shape)
	return point_ids[0,:]
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def recoverTestSet():
	# read sample from agglo samples
	#path='/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/'
	path='/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/'
	individual_path='/home/er13827/deepNetwork/skynetDATA/Eduardo/Individual/'
	#id_='1540921185'
	id_='1540909926'
	sampled_file=path+id_+'_samplePoints.pcd'
	#scene='kitchen5_d.pcd'
	scene='real-kitchen1.pcd'
	scene_file=path+scene
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	max_rad=0.806884
	ids_target=np.nonzero(new_c>=samples)[0]
	
	
	input_cloud,_,_=load_pcd_data_binary(scene_file)
	sampled_points,_,_=load_pcd_data_binary(sampled_file)

	big_data_set_data=np.zeros((sampled_points.shape[0],1024,3),dtype=np.float32)
	pointsPerCloud=1024
	kdt=KDTree(input_cloud,metric='euclidean')
	bar = Bar('Extracting data cloud',max=sampled_points.shape[0])
	for i in range(sampled_points.shape[0]):
		test_point=sampled_points[i,...]
		voxel_ids=getVoxel(test_point,max_rad,kdt)
		voxel=input_cloud[voxel_ids,:]
		if voxel.shape[0]<pointsPerCloud:
			#pad with repeated points
			to_sample_from=np.arange(voxel.shape[0])
			np.random.shuffle(to_sample_from)
			how_many_to_sample=pointsPerCloud-voxel.shape[0]
			repeat_ids=to_sample_from[:how_many_to_sample]
			repeated_voxel=input_cloud[repeat_ids,:]
			sample=np.concatenate((repeated_voxel,voxel),axis=0)-test_point
		else:
			sample=sample_cloud(voxel,pointsPerCloud)-test_point
		big_data_set_data[i,...]=sample
		bar.next()
	bar.finish()

	big_data_set_labels=np.zeros((sampled_points.shape[0],ids_target.size+1),dtype=np.int32)
	file_descriptor='tmp992.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	all_good_ids=np.zeros(sampled_points.shape[0],dtype=np.int32)
	bar=Bar('Extracting labels',max=ids_target.size)
	for i in range(ids_target.size):
		target_file=individual_path+labels[i,1]+'_'+labels[i,2]+'_real-kitchen1_*_13_*.pcd'
		some_files=glob.glob(target_file)
		main_file=''
		if len(some_files)>0:
			main_file=some_files[0]
			file_id=main_file.split('_')[-1].split('.')[0]
			good_points_file=individual_path+file_id+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(good_points_file)
			good_ids=pc.pc_data['id']
			big_data_set_labels[good_ids,i]=1
			all_good_ids[good_ids]=1
		bar.next()
	bar.finish()
	# check negatives
	non_positive=np.nonzero(all_good_ids)
	if len(non_positive)>0:
		big_data_set_labels[non_positive[0],-1]=1
	save_h5('common_test2.h5',big_data_set_data,big_data_set_labels,'float32','int32')


def recoverPredictions(scene='kitchen5'):
	path='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/'
	if scene=='kitchen5':
		copy_path='/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/'
		sampled=copy_path+'1540921185_samplePoints.pcd'
		predictions_file=path+'predicted.npy'
		sampled_ids_to_copy='/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/1540921185_samplePointsIds.pcd'
	else:
		copy_path='/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/'
		sampled=copy_path+'1540909926_samplePoints.pcd'
		predictions_file=path+'predicted2.npy'
		sampled_ids_to_copy='/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/1540909926_samplePointsIds.pcd'
	predictions=np.load(predictions_file)
	n_affordances=predictions.shape[1]-1
	#read predictions for sampled points
	points,_,_=load_pcd_data_binary(sampled)
	goodPoints=np.zeros((points.shape[0],6),dtype=np.float32)
	goodPointsX=np.zeros((points.shape[0]*n_affordances,3),dtype=np.int32)
	goodPointsIds=np.zeros((points.shape[0],1),dtype=np.int32)
	st=0
	ed=0
	bar=Bar('Recovering good predictions',max=predictions.shape[0])
	for i in range(predictions.shape[0]):
		predicted_positive=np.nonzero(predictions[i,:n_affordances])
		if len(predicted_positive)>0:
			predicted_positive=predicted_positive[0]
			ed=st+predicted_positive.size
			#print(st,ed,predicted_positive)
			goodPoints[i,:3]=points[i,:]
			goodPoints[i,3]=predicted_positive.size
			goodPointsX[st:ed,0]=predicted_positive
			goodPointsIds[i,0]=i
			st=ed
		bar.next()
	bar.finish()
	# save data as expected by metrics code
	# recover actual predictions
	actually_predicted=np.nonzero(goodPoints[:,3])[0]
	goodPoints=goodPoints[actually_predicted,...]
	goodPointsX=goodPointsX[:ed,...]
	goodPointsIds=goodPointsIds[actually_predicted,...]
	import time;
	ts = int(time.time())
	name=copy_path+'All_affordances_'+scene+'_3D_8_40_900_'+str(ts)+'.pcd'
	print(name)
	#make a dumb file
	os.system('touch '+name)
	# good points ->x,y,z,responses
	name=copy_path+str(ts)+'_goodPoints.pcd'
	print(name)
	actual_data_array=np.zeros(goodPoints.shape[0], dtype={'names':('x', 'y', 'z','rgb'),
                          'formats':('f4', 'f4', 'f4','u4')})
	actual_data_array['x']=goodPoints[:,0]
	actual_data_array['y']=goodPoints[:,1]
	actual_data_array['z']=goodPoints[:,2]
	#print(goodPoints[:10,3:].astype(np.uint8))
	#rgb=pypcd.encode_rgb_for_pcl(goodPoints[:,3:].astype(np.uint8))
	rgb = np.array((goodPoints[:,3].astype(np.int32) << 16) | (goodPoints[:,4].astype(np.int32) << 8) | (goodPoints[:,5].astype(np.int32) << 0),dtype=np.uint32)
	actual_data_array['rgb']=rgb
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	# good points x ->aff_id,score,oritentation
	name=copy_path+str(ts)+'_goodPointsX.pcd'
	print(name)
	actual_data_array=np.zeros(goodPointsX.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=goodPointsX[:,0]
	actual_data_array['y']=goodPointsX[:,1]
	actual_data_array['z']=goodPointsX[:,2]
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	# ids
	# name=copy_path+str(ts)+'_goodPointsIds.pcd'
	# print(name)
	# actual_data_array=np.zeros(goodPointsIds.shape[0], dtype={'names':('id'),
 #                          'formats':('i4')})
	# actual_data_array['id']=goodPointsIds[:,0]
	# new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	# new_cloud.save_pcd(name,compression='ascii')
	#sampled points
	# good points x ->aff_id,score,oritentation
	name=copy_path+str(ts)+'_samplePoints.pcd'
	print(name)
	actual_data_array=np.zeros(points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=points[:,0]
	actual_data_array['y']=points[:,1]
	actual_data_array['z']=points[:,2]
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')

	# copy samplePointsIds
	name=copy_path+str(ts)+'_samplePointsIds.pcd'
	print(name)
	command='cp '+sampled+' '+name
	os.system(command)


if __name__ == '__main__':
	recoverPredictions('real-kitchen1')