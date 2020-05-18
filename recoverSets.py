import socket
import os
import numpy as np
from prep_affordanceData import (save_h5,load_h5,load_pcd_data,load_pcd_data_binary,sample_cloud)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys 
from sklearn.neighbors import KDTree


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
	Input:
    BxNx3 array, original batch of point clouds
    Return:
	BxNx3 array, rotated batch of point clouds"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	cosval = np.cos(rotation_angle)
	sinval = np.sin(rotation_angle)
   	rotation_matrix = np.array([[cosval, -sinval,0],
                                   [sinval, cosval,0],
                                   [0, 0, 1]])
   	shape_pc = batch_data
   	rotated_data = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data

host=socket.gethostname()
if host=='it057384':
	MODEL_DIR='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/'
	NEW_DATA_DIR='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/New_data/'
	INPUT_DATA_DIR='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet/'
else:
	MODEL_DIR='/media/hal/DATA/Eduardo/PointNet2/Results_rad_centered/'
	NEW_DATA_DIR='/media/hal/DATA/Eduardo/PointNet2/New_data/'
	INPUT_DATA_DIR='/media/hal/DATA/Eduardo/PointNet/'

new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
samples=512
points=4096
max_rad=0.806884
#build a grid and kdtree
print('Building tree and grid')
x_=np.arange(-max_rad,max_rad,0.01)
y_=np.arange(-max_rad,max_rad,0.01)
z_=np.arange(-max_rad,max_rad,0.01)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
x=x.reshape(1,-1)
y=y.reshape(1,-1)
z=z.reshape(1,-1)

grid=np.concatenate((x,y,z),axis=0).T
kdt=KDTree(grid,metric='euclidean')

traininig_examples=512
#sys.exit()

ids_target=np.nonzero(new_c>=samples)[0]
# fig2 = plt.figure(figsize=(6, 6))
# plt.ion()
# ax = fig2.add_subplot(111, projection='3d')
# ax2 = fig2.add_subplot(132, projection='3d')
# ax3 = fig2.add_subplot(133, projection='3d')

# fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
c=0
for i in range(ids_target.size):
	counts=np.zeros((grid.shape[0],1))
	interaction=ids_target[i]
	#check that pointsets exist
	activations_file=MODEL_DIR+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_activations.npy'
	pointsets1_file=MODEL_DIR+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_pointsets1.npy'
	pointsets2_file=MODEL_DIR+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_pointsets2.npy'
	data_presented_file=MODEL_DIR+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_presented.npy'
	
	if os.path.exists(activations_file) and os.path.exists(data_presented_file):
		print('Processing %d'%(interaction))
	else:
		continue
	#read data tested ids
	ids_presented=np.load(data_presented_file).astype(int)
	#pointsets1=np.load(pointsets1_file)
	#pointsets2=np.load(pointsets2_file)
	activations=np.load(activations_file)
	#print(pointsets1.shape)
	#this goes 0-1023
	tmp_ids=np.nonzero(ids_presented==0)[0]
	#print(tmp_ids.size)
	#ids_presented=ids_presented[tmp_ids]
	#get the original data
	original_data_points_file=NEW_DATA_DIR+'dataPoints_'+str(interaction)+'.h5'
	original_data_clouds_file=NEW_DATA_DIR+'dataClouds_'+str(interaction)+'.h5'
	input_clouds_file=INPUT_DATA_DIR+'binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(traininig_examples)+'.h5'
	data_presented_original_ids_file=NEW_DATA_DIR+'binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(traininig_examples)+'_shuffledIds.npy'
	#print(data_presented_original_ids_file)
	#this goes 0-1023
	#_,original_ids=load_h5(data_presented_original_ids_file)
	original_ids=np.load(data_presented_original_ids_file)
	input_clouds,_=load_h5(input_clouds_file)
	original_points,_=load_h5(original_data_points_file)
	#original_clouds,_=load_h5(original_data_clouds_file)
	#find indices of corresponding clouds
	#ids=np.nonzero(original_ids>511)[0]
	#for j in range(ids_presented.shape[0]):
	#print(original_ids[:10])
	#print(ids_presented[:10])
	for j in range(tmp_ids.size):
		anId=tmp_ids[j]
		#affordance 'positive' examples are the last 512 in the dataset
		real_id=original_ids[anId]-512
		pointcloud_id=real_id
		if pointcloud_id!=511:
			continue
		one_original_cloud=input_clouds[anId,...]
		# print(one_original_cloud.shape)
		fired_points_ids=np.unique(activations[anId,...]).reshape(1,-1)
		fired_points=one_original_cloud[fired_points_ids[0,:],:]
		# print(fired_points[:3,:].shape)
		# print('-----------------')
		#fired_points2=pointsets2[anId,...]
		#get the pointcloud number
		#print('%d %d %d'%(j,anId,real_id))
		original_data=original_points[pointcloud_id,...]
		if pointcloud_id!=511:
			back_points1=rotate_point_cloud_by_angle(fired_points,-original_data[4]*(2*np.pi)/8)
		else:
			back_points1=fired_points
		#back_points2=rotate_point_cloud_by_angle(fired_points2,-original_data[4]*(2*np.pi)/8)
		#back_points3=rotate_point_cloud_by_angle(back_points2,original_data[4]*(2*np.pi)/8)
		#print('Score: %f Angle %d'%(original_data[5],original_data[4]))
		if pointcloud_id==511:
			print('Original training cloud')
			print(fired_points_ids.size)
			name='../data/activations/OriginalActivations2048_'+str(interaction)
			np.save(name,back_points1)
		# 	ax.scatter(one_original_cloud[:,0],one_original_cloud[:,1],one_original_cloud[:,2],s=1,c='b')
		# 	ax.scatter(fired_points[:,0],fired_points[:,1],fired_points[:,2],s=10,c='r')
		# 	plt.pause(0.5)
		# 	plt.draw()
		# 	ax.clear()
	# 	for k in range(back_points1.shape[0]):
	# 		activated=back_points1[k,:].reshape(1,-1)
	# 		if activated[0,0]==0 and activated[0,1]==0 and activated[0,2]==0:
	# 			continue
	# 		#print(activated.shape)
	# 		_, ind = kdt.query(activated, k=1)
	# 		counts[ind[0,0]]+=1

	# 	# print(fired_points.shape)
		
	# 	#ax2.scatter(back_points1[:,0],back_points1[:,1],back_points1[:,2],s=3,c='r')
	# 	#ax.scatter(back_points2[:,0],back_points2[:,1],back_points2[:,2],s=3,c='g')
	# 	# ax.scatter(back_points3[:,0],back_points3[:,1],back_points3[:,2],s=10,c='r')
	# 	# plt.pause(5)
	# 	# plt.draw()
	# 	# ax.clear()
	# 	#ax2.clear()
	# 	#ax3.clear()
	# populated=np.nonzero(counts>4)[0]
	# name='../data/activations/pointset_'+str(interaction)
	# np.save(name,grid[populated,:])
	# name='../data/activations/counts_'+str(interaction)
	# np.save(name,counts)
# 	axs[c].hist(counts[populated],bins=100)
# 	if c==0:
# 		ax.scatter(grid[populated,0],grid[populated,1],grid[populated,2],s=3)
# 	elif c==1:
# 		ax2.scatter(grid[populated,0],grid[populated,1],grid[populated,2],s=3)
# 	elif c==2:
# 		ax3.scatter(grid[populated,0],grid[populated,1],grid[populated,2],s=3)
# 	c+=1
	# print('Non empty= %d'%(populated.size))
	# print('Non empty (1)= %d'%np.nonzero(counts)[0].size)
	# print('Non empty (2)= %d'%np.nonzero(counts>1)[0].size)
	# print('Non empty (3)= %d'%np.nonzero(counts>2)[0].size)
	# print('Non empty (4)= %d'%np.nonzero(counts>3)[0].size)
# plt.show()
#plt