import numpy as np
import h5py
from compareIT import (load_h5,load_pcd_data_binary)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import socket
import os
import matplotlib.pyplot as plt
from progress.bar import Bar
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import sys
import pypcd

host=socket.gethostname()

if host=="it057384":
	data_paths=['/home/er13827/deepNetwork/halDATA/Eduardo/PointNet/','/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/','/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/']
else:
	data_paths=['/media/hal/DATA/Eduardo/PointNet/','/media/hal/DATA/Eduardo/PointNet2/Results_rad/','/media/hal/DATA/Eduardo/PointNet2/Results_rad_centered/']

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
	Input:
    BxNx3 array, original batch of point clouds
    Return:
	BxNx3 array, rotated batch of point clouds"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	cosval = np.cos(rotation_angle)
	sinval = np.sin(rotation_angle)
   	rotation_matrix = np.array([[cosval, sinval,0],
                                   [-sinval, cosval,0],
                                   [0, 0, 1]])
   	shape_pc = batch_data
   	rotated_data = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data

def readTrainingScene(data_file):
	with open(data_file) as f:
			content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	tmp=content[0].split(":")[1]
	return tmp

def readTrainingSamplePoint(data_file):
	with open(data_file) as f:
			content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	tmp=content[8].split(":")[1]
	datapoint=tmp.split(',')
	test_point=np.expand_dims(np.asarray([float(x) for x in datapoint]),axis=0)
	return test_point
def setPlotLims(ax1,ax2,ax3,lim):
	ax1.set_xlim(-lim,lim)
	ax1.set_ylim(-lim,lim)
	ax1.set_zlim(-lim,lim)
	ax2.set_xlim(-lim,lim)
	ax2.set_ylim(-lim,lim)
	ax2.set_zlim(-lim,lim)
	ax3.set_xlim(-lim,lim)
	ax3.set_ylim(-lim,lim)
	ax3.set_zlim(-lim,lim)

def save_as_h5(h5_filename, data, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.close()

def load_from_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    return data

def load_ply_data(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def recoverAggloSets():
	cell_size=0.01
	#995 1cm cells without clipping
	#994 1cm cells with clipping
	#993 0.7cm cells with clipping
	#992 0.5cm cells with clipping
	descriptor=994
	if not os.path.exists('agglo_all_data_clipped.h5'):
		name=data_paths[0]+'MultilabelDataSet_splitTest2.h5'
		input_clouds,input_labels=load_h5(name)
		input_ids_name=data_paths[1]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/inputIds.npy'
		inputIds=np.load(input_ids_name)
		new_name=data_paths[1]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/recoveredActivations.npy'
		if not os.path.exists(new_name):
			name=data_paths[1]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/points_sampled.npy'
			pointSampled=np.load(name)
			name=data_paths[1]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/pointIds.npy'
			pointSampledIds=np.load(name)
			name=data_paths[1]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/activationIds.npy'
			activationIds=np.load(name)
		
			#all activations should have at most 128 points with ids in [0-32)	
			all_activations=np.zeros((input_clouds.shape[0],pointSampled.shape[1],128),dtype=np.int32)
			print('Input clouds')
			print(input_clouds.shape)
			print('Points sampled')
			print(pointSampled.shape)
			print('Sampled ids')
			print(pointSampledIds.shape)
			print('Activations')
			print(activationIds.shape)
			#
			#oneSampleOk=np.zeros(oneActivation.shape)-1
			# For every point sampled in the first layer
			bar = Bar('Recovering data ',max=input_clouds.shape[0])
			for j in range(input_clouds.shape[0]):
				oneSample=pointSampledIds[j,...]
				oneActivation=activationIds[j,...]
				for i in range(oneActivation.shape[0]):
					point_ids_per_sample=oneActivation[i,:]
					#print('%d Per sampled point %d'%(i,point_ids_per_sample.size))
					#print(point_ids_per_sample[:30])
					all_activations[j,i,...]=oneSample[i,point_ids_per_sample]
					#print('%d in sample %d'%(i,oneSample[i,point_ids_per_sample].size))
					#print(oneSample[i,point_ids_per_sample[:30]])
				bar.next()
			bar.finish()
			print(all_activations.shape)	
			del pointSampled,pointSampledIds,activationIds
			np.save(new_name,all_activations)
		else:
			all_activations=np.load(new_name)

		# for x in range(all_activations.shape[0]):
		# 	print('%d Unique %d for %d Affordances'%(x,np.unique(all_activations[x,...]).size,np.nonzero(input_labels[x,...])[0].size))

		# Read the orientations of this clouds with ids of the split
		if not os.path.exists('splitTest2_orientations.npy'):
			print('Recovering orientations')
			orientations=np.zeros((input_labels.shape[0],input_labels.shape[1]-1))
			test_ids=np.load('MultilabelDataSet_splitTest2.npy')
			# SplitTest2 -> kitchen5+real-kitchen1
			files=['MultilabelDataSet_kitchen5_Orientations.npy','MultilabelDataSet_real-kitchen1_Orientations.npy']
			someOrientations=np.load(files[0])
			orientations_here=np.nonzero(test_ids<someOrientations.shape[0])[0]
			real_ids=test_ids[orientations_here]
			orientations[orientations_here,...]=someOrientations[real_ids,...]
			someOrientations2=np.load(files[1])
			orientations_here=np.nonzero(test_ids>=someOrientations.shape[0])[0]
			real_ids=test_ids[orientations_here]-someOrientations.shape[0]
			orientations[orientations_here,...]=someOrientations2[real_ids,...]
			np.save('splitTest2_orientations.npy',orientations)
		else:
			print('Reading orientations')
			orientations=np.load('splitTest2_orientations.npy')

		#Read affordances/tensors All scenes have same names
		names_labels=np.expand_dims(np.genfromtxt('common_namesreal-kitchen1.csv',dtype='str'),axis=1)
		max_rad=0.806884
		# build a grid and kdtree
		# same for every affordance
		print('grid')
		x_=np.arange(-max_rad,max_rad,cell_size)
		y_=np.arange(-max_rad,max_rad,cell_size)
		z_=np.arange(-max_rad,max_rad,cell_size)
		x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
		x=x.reshape(1,-1)
		y=y.reshape(1,-1)
		z=z.reshape(1,-1)

		grid=np.concatenate((x,y,z),axis=0).T
		print('Done')
		#ax = fig.add_subplot(111,projection="3d")
		n_affordances=names_labels.size-1
		object_sizes=np.zeros((n_affordances,1),dtype=np.float32)

		if not os.path.exists('pop_cell_ids_clipped.npy'):
			print('Building tree ')
			kdt=KDTree(grid,metric='euclidean')
			print('done')
			cell_ids=np.zeros((grid.shape[0],n_affordances),dtype=np.int32)
			bar = Bar('Checking pop cells',max=n_affordances)
			for i in range(names_labels.size-1):
				tokens=names_labels[i,0].split('-')
				if len(tokens)>2:
					aff=tokens[0]
					obj=tokens[1]+'-'+tokens[2]
				else:
					aff=tokens[0]
					obj=tokens[1]
				if aff=='Place':dirr='Placing'
				elif aff=='Fill':dirr='Filling'
				elif aff=='Hang':dirr='Hanging'
				elif aff=='Sit':dirr='Sitting'
				tensor_file='/home/er13827/space/testing/'+dirr+'/'+aff+'_'+obj+'_field_clean.pcd'
				#read tensor cloud
				cloud,_,_=load_pcd_data_binary(tensor_file)
				data_file='/home/er13827/space/testing/'+dirr+'/ibs_full_'+aff+'_'+obj+'.txt'
				#read data file -> scene point to translate everything
				test_point=readTrainingSamplePoint(data_file)
				#read object size for later clipping
				object_cloud_file='/home/er13827/space/testing/'+dirr+'/'+obj+'.ply'
				#print(object_cloud_file)
				o_points=load_ply_data(object_cloud_file)
				maxP=np.max(o_points,axis=0).reshape(1,-1)
				minP=np.min(o_points,axis=0).reshape(1,-1)
				a_size=np.linalg.norm(maxP-minP,axis=1)
				object_sizes[i,0]=a_size
				#print(a_size)
				#translate cloud back to origin
				cloud=cloud-test_point
				#clip pointcloud inside sphere with a_size radi
				distances=np.linalg.norm(cloud,axis=1)
				inside_sphere=np.nonzero(distances<=(a_size/2))[0]
				cloud=cloud[inside_sphere,:]
				#fit the grid to the tensor and get cells
				_,ind=kdt.query(cloud,k=1)
				real_activations=np.unique(ind[:,0])
				cell_ids[real_activations,i]+=1
				# ax.scatter(grid[real_activations,0],grid[real_activations,1],grid[real_activations,2],s=1,c='b')
				# ax.scatter(0,0,0,s=25,c='r')
				# ax.set_title(aff+'-'+obj)
				# plt.pause(3)
				# plt.draw()
				# ax.clear()
				bar.next()
			bar.finish()
			np.save('pop_cell_ids_clipped.npy',cell_ids)
		else:
			cell_ids=np.load('pop_cell_ids_clipped.npy')
		print(cell_ids.shape)
		# for i in range(cell_ids.shape[1]):
		# 	pop_cells=np.nonzero(cell_ids[:,i])[0]
		# 	print('Affordance %d Cells %d'%(i,pop_cells.size))
		#for every activation get the closest grid point 
		plot=False
		if plot:
			fig = plt.figure(figsize=(15, 7))
			plt.ion()
			#ax = fig.add_subplot(131,projection="3d")
			ax2 = fig.add_subplot(121,projection="3d")
			ax3 = fig.add_subplot(122,projection="3d")
			#just to skip file checking below
			name='all_projected_activations_clipped.npy'
		else:
			name='all_projected_activations_clipped.npy'
			# ax.view_init(azim=135)
			# ax2.view_init(azim=135)
			# ax3.view_init(azim=135)
		if not os.path.exists(name):
			responses=np.zeros(cell_ids.shape,dtype=np.int16)
			bar = Bar('Projecting activations into grid',max=all_activations.shape[0])
			for i in range(all_activations.shape[0]):
				rotations=orientations[i,...]
				thisActivations=all_activations[i,...]
				real_ids=np.unique(thisActivations)
				real_ids=inputIds[i,real_ids]
				activations_3d=input_clouds[i,real_ids,:]
				#print(activations_3d.shape)
				affordances_actually_here=np.nonzero(input_labels[i,:n_affordances])[0]
				for j in range(affordances_actually_here.size):
					affordance_id=affordances_actually_here[j]
					#rotate back the actications
					anAngle=-rotations[affordance_id]*(2*np.pi)/8
					aCloud=rotate_point_cloud_by_angle(activations_3d,anAngle)
					#pop cells goes from 0 to gridSize
					pop_cells=np.nonzero(cell_ids[:,affordance_id])[0]
					#build a search tree with only those cells activated with current affordance
					active_grid=grid[pop_cells,:]
					kdt=KDTree(active_grid,metric='euclidean')
					# get the closest cell for every activation point
					_,ind=kdt.query(aCloud,k=1)
					#ids from 0 to popCells size
					ind=np.unique(ind[:,0])
					#print(ind.size,affordance_id)
					responses[pop_cells[ind],affordance_id]+=1

					if plot:
						#setPlotLims(ax,ax2,ax3,max_rad)
						# ax.scatter(activations_3d[:,0],activations_3d[:,1],activations_3d[:,2],s=1,c='b')
						# ax.set_title(str(j)+' '+str(rotations[affordance_id]))
						# ax.scatter(0,0,0,s=25,c='r')
						somePoint=np.array([0,-0.8,0])
						# ax.plot([0,somePoint[0]],[0,somePoint[1]],[0,somePoint[2]],linewidth=2, markersize=12,color='g')
						rotatedPoint=rotate_point_cloud_by_angle(somePoint,anAngle)
						ax2.scatter(aCloud[:,0],aCloud[:,1],aCloud[:,2],s=1,c='b')
						ax2.scatter(0,0,0,s=25,c='r')
						ax2.plot([0,rotatedPoint[0,0]],[0,rotatedPoint[0,1]],[0,rotatedPoint[0,2]],linewidth=2, markersize=12,color='g')
						ax2.set_title(str(j)+' '+str(anAngle))
						ax3.scatter(active_grid[:,0],active_grid[:,1],active_grid[:,2],s=1,c='b')
						#ax3.plot([0,somePoint[0]],[0,somePoint[1]],[0,somePoint[2]],linewidth=2, markersize=12,color='g')
						nn_cloud=grid[pop_cells[ind],:]
						ax3.scatter(nn_cloud[:,0],nn_cloud[:,1],nn_cloud[:,2],s=20,c='g')
						if rotations[affordance_id]!=0 and rotations[affordance_id]!=4:
							plt.pause(10)
						else:
							plt.pause(1)
						plt.draw()
						#ax.clear()
						ax2.clear()
						ax3.clear()
				bar.next()
			bar.finish()
			np.save('all_projected_activations_clipped.npy',responses)
		else:
			print('Reading all projected activations')
			responses=np.load('all_projected_activations_clipped.npy')
	
		fired_up=np.count_nonzero(responses,axis=0)
		#print(fired_up)
		trainig_instances=np.count_nonzero(input_labels[:,0:n_affordances],axis=0)
		#print(trainig_instances)
		# fig = plt.figure(figsize=(7, 7))
		# plt.ion()
		# ax = fig.add_subplot(111)
		#ax.view_init(azim=135)
		common_cells=np.zeros(responses.shape,dtype=np.int8)
		for i in range(n_affordances):
			# average response per cell
			this_responses=responses[:,i]/float(trainig_instances[i])
			
			#get the cells that fired up at least half of the time
			pop=np.nonzero(this_responses>=0.5)[0]
			#print('%s %d'%(names_labels[i,0],pop.size))
			common_cells[pop,i]=1
		del responses,all_activations
		tmp=np.count_nonzero(common_cells,axis=1)
		tmp_ids=np.nonzero(tmp)[0]
		smaller_grid=grid[tmp_ids,:]
		common_cells=common_cells[tmp_ids,...]
		agglo_points=np.zeros(smaller_grid.shape)
		print(common_cells.shape,smaller_grid.shape)
		real_size=np.sum(np.sum(common_cells,axis=1))
		all_data=np.empty((real_size,6))
		all_data_extra=np.empty((real_size,5))
		agglo_data=np.zeros((agglo_points.shape[0],1),dtype=np.int32)
		#sys.exit()
		start_i=0
		bar = Bar('Going through data',max=common_cells.shape[0])
		#read again checking common points
		for i in range(common_cells.shape[0]):
			cell_activations=np.nonzero(common_cells[i,:])[0]
			#for every affordance here, find NN in everytensor
			#and update centroid
			# print('Sampling from the following affordaces:')
			# print(names_labels[cell_activations,0])
			cell_centre=smaller_grid[i,:].reshape(1,-1)
			cell_data=np.zeros((cell_activations.size,6),dtype=np.float32)
			cell_data_extra=np.zeros((cell_activations.size,5),dtype=np.float32)
			agglo_data[i,0]=cell_activations.size
			end_i=start_i+cell_activations.size
			for j in range(cell_activations.size):
				an_interaction=cell_activations[j]
				tokens=names_labels[an_interaction,0].split('-')
				if len(tokens)>2:
					aff=tokens[0]
					obj=tokens[1]+'-'+tokens[2]
				else:
					aff=tokens[0]
					obj=tokens[1]
				if aff=='Place':dirr='Placing'
				elif aff=='Fill':dirr='Filling'
				elif aff=='Hang':dirr='Hanging'
				elif aff=='Sit':dirr='Sitting'
				tensor_file='/home/er13827/space/testing/'+dirr+'/'+aff+'_'+obj+'_field_clean.pcd'
				#read tensor cloud
				cloud,_,normals=load_pcd_data_binary(tensor_file)
				norm_mags=np.linalg.norm(normals,axis=1)
				data_file='/home/er13827/space/testing/'+dirr+'/ibs_full_'+aff+'_'+obj+'.txt'
				#read data file -> scene point to translate everything
				test_point=readTrainingSamplePoint(data_file)
				#translate cloud back to origin
				cloud=cloud-test_point
				#get NN in tensor
				kdt=KDTree(cloud,metric='euclidean')
				_,ind=kdt.query(cell_centre,k=1)
				keypoint_id=ind[0,0]
				#get 3d point and vector 
				#keypoint_data=np.concatenate((,),axis=1)
				cell_data[j,:3]=cloud[keypoint_id,:]
				cell_data[j,3:]=normals[keypoint_id,:]
				#id from tensor
				cell_data_extra[j,0]=keypoint_id
				#id of orientation
				cell_data_extra[j,1]=0
				#id of affordance
				cell_data_extra[j,2]=an_interaction
				#max vector in tensor
				cell_data_extra[j,3]=np.max(norm_mags)
				#min vector in tensor
				cell_data_extra[j,4]=np.min(norm_mags)
			#recompute centroid
			agglo_points[i,:]=np.mean(cell_data[:,:3],axis=0)
			all_data[start_i:end_i,...]=cell_data
			all_data_extra[start_i:end_i,...]=cell_data_extra
			start_i=end_i
			bar.next()
		bar.finish()
		save_as_h5('agglo_all_data_clipped.h5',all_data,'float32')
		save_as_h5('agglo_all_data_extra_clipped.h5',all_data_extra,'float32')
		save_as_h5('agglo_points_clipped.h5',agglo_points,'float32')
		save_as_h5('agglo_data_clipped.h5',agglo_data,'int32')
	else:
		print('Reading agglo data')
		all_data=load_from_h5('agglo_all_data_clipped.h5')
		all_data_extra=load_from_h5('agglo_all_data_extra_clipped.h5')
		agglo_points=load_from_h5('agglo_points_clipped.h5')
		agglo_data=load_from_h5('agglo_data_clipped.h5')
	#will need to rotate everything 
	bigger_data_points=np.empty((all_data.shape[0]*8,3))
	bigger_agglo_points=np.empty((agglo_points.shape[0]*8,3))
	start_i1=0
	start_i2=0
	oris=np.zeros((all_data_extra.shape[0]*8,1))
	for i in range(8):
		end_i1=start_i1+all_data.shape[0]
		angle=i*(2*np.pi/8)
		bigger_data_points[start_i1:end_i1,...]=rotate_point_cloud_by_angle(all_data[:,:3],angle)
		oris[start_i1:end_i1,0]=i
		end_i2=start_i2+agglo_points.shape[0]
		bigger_agglo_points[start_i2:end_i2,...]=rotate_point_cloud_by_angle(agglo_points,angle)
		start_i2=end_i2
		start_i1=end_i1
	#create agglo data for iT code
	#centroids for NN search
	name='New'+str(descriptor)+'_Approx_descriptor_8.pcd'
	actual_data_array=np.zeros(bigger_agglo_points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=bigger_agglo_points[:,0]
	actual_data_array['y']=bigger_agglo_points[:,1]
	actual_data_array['z']=bigger_agglo_points[:,2]
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	#members per ccell
	name='New'+str(descriptor)+'_Approx_descriptor_8_members.pcd'
	new_agglo_data=np.expand_dims(np.tile(agglo_data[:,0],8),axis=1)
	#print(new_agglo_data.shape)
	cum_sum=np.cumsum(new_agglo_data,axis=0)-new_agglo_data
	#cum_sum=np.expand_dims(cum_sum,axis=1)
	print(cum_sum.shape)
	actual_data_array=np.zeros(new_agglo_data.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=new_agglo_data[:,0]
	actual_data_array['y']=cum_sum[:,0]
	actual_data_array['z']=0
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	# extra info -> aff_id,ori_id,pv_id
	name='New'+str(descriptor)+'_Approx_descriptor_8_extra.pcd'
	actual_data_array=np.zeros(bigger_data_points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=np.tile(all_data_extra[:,2]+1,8)
	actual_data_array['y']=oris[:,0]
	actual_data_array['z']=0
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	# raw points -> all 3d locations represented by the agglo
	name='New'+str(descriptor)+'_Approx_descriptor_8_points.pcd'
	actual_data_array=np.zeros(bigger_data_points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=bigger_data_points[:,0]
	actual_data_array['y']=bigger_data_points[:,1]
	actual_data_array['z']=bigger_data_points[:,2]
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	#vdata -> mags, weights 
	# firts need to remap weights because I did not do it above
	n_affordances=np.unique(all_data_extra[:,2]).size
	names_labels=np.expand_dims(np.genfromtxt('common_namesreal-kitchen1.csv',dtype='str'),axis=1)
	all_names=np.empty((names_labels.shape[0],3),dtype='object')
	print('Total Affordances %d'%(n_affordances))
	point_counts_data=np.empty((n_affordances,8),dtype=np.float32)
	vdata=np.empty((all_data.shape[0],2))
	for i in range(n_affordances):
		#get the points in this affordance
		ids=np.nonzero(all_data_extra[:,2]==i)[0]
		#get the max and min values
		maxV=np.max(all_data_extra[ids,3])
		minV=np.min(all_data_extra[ids,4])
		vectors=all_data[ids,3:]
		vectors_norm=np.linalg.norm(vectors,axis=1)
		weights=(vectors_norm-minV)*((1-0)/(maxV-minV))+0
		vdata[ids,0]=vectors_norm
		vdata[ids,1]=1-weights
		#get also the per-affordance points to build the point_counts file
		point_counts_data[i,:]=ids.size
		tokens=names_labels[i,0].split('-')
		if len(tokens)>2:
			aff=tokens[0]
			obj=tokens[1]+'-'+tokens[2]
		else:
			aff=tokens[0]
			obj=tokens[1]
		all_names[i,1]=aff
		all_names[i,2]=obj
		if aff=='Place':all_names[i,0]='Placing'
		elif aff=='Fill':all_names[i,0]='Filling'
		elif aff=='Hang':all_names[i,0]='Hanging'
		elif aff=='Sit':all_names[i,0]='Sitting'
		#save the new "sample" taken from the agglo representation
		data_file='/home/er13827/space/testing/'+all_names[i,0]+'/ibs_full_'+all_names[i,1]+'_'+all_names[i,2]+'.txt'
		#read data file -> scene point to translate everything
		test_point=readTrainingSamplePoint(data_file)
		points=all_data[ids,:3]+test_point
		name=all_names[i,1]+'_'+all_names[i,2]+'_agglo_sample'+str(descriptor)+'.pcd'
		actual_data_array=np.zeros(points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
		actual_data_array['x']=points[:,0]
		actual_data_array['y']=points[:,1]
		actual_data_array['z']=points[:,2]
		new_cloud = pypcd.PointCloud.from_array(actual_data_array)
		new_cloud.save_pcd(name,compression='ascii')



	name='New'+str(descriptor)+'_Approx_descriptor_8_vdata.pcd'
	actual_data_array=np.zeros(bigger_data_points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=np.tile(vdata[:,0],8)
	actual_data_array['y']=np.tile(vdata[:,1],8)
	actual_data_array['z']=0
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	#raw vectors -> provenance vectors
	name='New'+str(descriptor)+'_Approx_descriptor_8_vectors.pcd'
	actual_data_array=np.zeros(bigger_data_points.shape[0], dtype={'names':('x', 'y', 'z'),
                          'formats':('f4', 'f4', 'f4')})
	actual_data_array['x']=np.tile(all_data[:,3],8)
	actual_data_array['y']=np.tile(all_data[:,4],8)
	actual_data_array['z']=np.tile(all_data[:,5],8)
	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
	new_cloud.save_pcd(name,compression='ascii')
	print(name)
	name='point_count'+str(descriptor)+'.dat'
	f = open(name, 'w+b')	
	b=np.array(point_counts_data.shape,dtype=np.uint32).reshape(1,-1)
	print(point_counts_data)
	b=np.fliplr(b)
	print(b)
	binary_format = bytearray(b)
	f.write(binary_format)
	binary_format = bytearray(point_counts_data.T)
	f.write(binary_format)
	f.close()
	print(name)
	name='tmp'+str(descriptor)+'.csv'
	with open(name, "w") as text_file:
			text_file.write("Directory,Affordance,Object\n")
			for i in range(n_affordances):
				text_file.write("%s,%s,%s\n" % (all_names[i,0],all_names[i,1],all_names[i,2]))
	print(name)



def recoverSets():
	name=data_paths[0]+'binaryOc_AffordancesDataset_test1_512.h5'
	input_clouds,input_labels=load_h5(name)
	new_name=data_paths[2]+'AFF_1_BATCH_16_EXAMPLES_512_DATA_binary_OC/dump/recoveredActivations.npy'
	data_presented_original_ids_file='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/New_data/binaryOc_AffordancesDataset_test1_512_shuffledIds.npy'
	original_ids=np.load(data_presented_original_ids_file)
	target=np.nonzero(original_ids==1023)[0]
	#print(target,original_ids[target])
	input_ids_name=data_paths[2]+'AFF_1_BATCH_16_EXAMPLES_512_DATA_binary_OC/dump/inputIds.npy'
	inputIds=np.load(input_ids_name)
	name=data_paths[2]+'AFF_1_BATCH_16_EXAMPLES_512_DATA_binary_OC/dump/points_sampled.npy'
	pointSampled=np.load(name)
	if not os.path.exists(new_name):
		name=data_paths[2]+'AFF_1_BATCH_16_EXAMPLES_512_DATA_binary_OC/dump/pointIds.npy'
		pointSampledIds=np.load(name)
		name=data_paths[2]+'AFF_1_BATCH_16_EXAMPLES_512_DATA_binary_OC/dump/activationIds.npy'
		activationIds=np.load(name)

		#all activations should have at most 128 points with ids in [0-32)	
		all_activations=np.zeros((input_clouds.shape[0],pointSampled.shape[1],128),dtype=np.int32)
		print('Input clouds')
		print(input_clouds.shape)
		print('Points sampled')
		print(pointSampled.shape)
		print('Sampled ids')
		print(pointSampledIds.shape)
		print('Activations')
		print(activationIds.shape)
		#
		#oneSampleOk=np.zeros(oneActivation.shape)-1
		# For every point sampled in the first layer
		bar = Bar('Recovering data ',max=input_clouds.shape[0])
		for j in range(input_clouds.shape[0]):
			oneSample=pointSampledIds[j,...]
			oneActivation=activationIds[j,...]
			for i in range(oneActivation.shape[0]):
				point_ids_per_sample=oneActivation[i,:]
				#print('%d Per sampled point %d'%(i,point_ids_per_sample.size))
				#print(point_ids_per_sample[:30])
				all_activations[j,i,...]=oneSample[i,point_ids_per_sample]
				#print('%d in sample %d'%(i,oneSample[i,point_ids_per_sample].size))
				#print(oneSample[i,point_ids_per_sample[:30]])
			bar.next()
		bar.finish()
		print(all_activations.shape)	
		del pointSampled,pointSampledIds,activationIds
		np.save(new_name,all_activations)
	else:
		all_activations=np.load(new_name)

	max_rad=0.806884
	#build a grid and kdtree
	print('Building tree and grid')
	data_file='/home/er13827/space/testing/Filling/ibs_full_Fill_bowl.txt'
	with open(data_file) as f:
			content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	tmp=content[8].split(":")[1]
	datapoint=tmp.split(',')
	test_point=np.expand_dims(np.asarray([float(x) for x in datapoint]),axis=0)

	clean_tensor_file='/home/er13827/space/testing/Filling/Fill_bowl_field_clean.pcd'
	cloud,_,_=load_pcd_data_binary(clean_tensor_file)
	cloud=cloud-test_point
	someIds=np.arange(cloud.shape[0])
	np.random.shuffle(someIds)
	kdt=KDTree(cloud,metric='euclidean')
	# for x in range(all_activations.shape[0]):
	# 	print('%d Unique %d for %d Affordances'%(x,np.unique(all_activations[x,...]).size,np.nonzero(input_labels[x,...]==0)[0].size))
	print('%d Unique %d for %d Affordances'%(target,np.unique(all_activations[target,...]).size,np.nonzero(input_labels[target,...]==0)[0].size))
	actual_input_cloud=np.squeeze(input_clouds[target,inputIds[target,...],:])
	activations_3d=actual_input_cloud[np.unique(all_activations[target,...]),:]
	tensor_ids=np.zeros((activations_3d.shape[0],1),dtype=np.int32)
	print(activations_3d.shape)
	for j in range(activations_3d.shape[0]):
		activated=activations_3d[j,:].reshape(1,-1)
		_, ind = kdt.query(activated, k=1)
		tensor_ids[j,0]=ind[0,0]
	tensor_ids=np.unique(tensor_ids)
	tensor_activations=cloud[tensor_ids,...]
	print(tensor_activations.shape)

	print(actual_input_cloud.shape)
	actual_sampled=np.squeeze(pointSampled[target,...])
	print(actual_sampled.shape)
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111,projection='3d')
	#ax.scatter(input_clouds[target,:,0],input_clouds[target,:,1],input_clouds[target,:,2],s=1,c='b')
	ax.scatter(actual_input_cloud[:,0],actual_input_cloud[:,1],actual_input_cloud[:,2],s=5,c='r')
	ax.scatter(actual_sampled[:,0],actual_sampled[:,1],actual_sampled[:,2],s=10,c='g')
	tensor_activations=cloud[someIds[:2048],...]
	ax.scatter(tensor_activations[:,0],tensor_activations[:,1],tensor_activations[:,2],s=20,c='c')
	plt.show()


if __name__ == '__main__':
	recoverAggloSets()
	#recoverSets()
