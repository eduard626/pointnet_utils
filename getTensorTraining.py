from __future__ import print_function
import numpy as np
from plyfile import (PlyData)
import sys
import os
from prep_affordanceData import (save_h5,load_h5,load_pcd_data,load_pcd_data_binary,sample_cloud)
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
from progress.bar import Bar
import csv

max_rad=0.806884
n_points=2048*2
n_samples=128
n_orientations=8
TRAIN_EXAMPLES=1


result_dirs=['/home/er13827/space/testing/','/home/er13827/space/Affordances/Hal/voronoi/build/']
extra_dirs=['/home/er13827/deepNetwork/skynetDATA/Eduardo/ScanNet/','/home/er13827/deepNetwork/halDATA/Eduardo/ScanNet/','/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build']


X = 0
Y = 1
Z = 2
A_ID=3
O_ID=4
SCORE=5

#some_counts=np.array([863936 ,3584856 , 243671 ,3867619 ,2970017 ,3131876 ,2329841 , 962446 , 317499 , 155735 , 312400 , 235135 , 773958 ,    185 , 154982 , 226570 , 113706 ,1331054 , 846818 ,  33079 , 310646 ,1276737 ,    591 , 799662 , 356899 , 272537 , 689864 , 256414 ,      1 , 194977 , 792847 ,  91732 , 953394 ,1306723 , 791602 ,      0 , 523711 ,1374852 , 890657 ,2419878 ,1286064 , 183447 , 216221 ,  29845 , 436333 , 459637 ,  69910 ,      0 ,1050976 ,      0 ,  64009 ,1121742 ,  49403 ,1266039 ,      0 , 606988 , 727020 ,1279063 ,1585769 , 710830 ,1477731 ,1688965 , 115653 , 495778 , 337779 ,1328770 , 403318 ,1175671 ,1406571 ,  51643 , 585446 ,  40487 ,1191233 , 575456 ,   1535 , 825952 , 612953 , 308265 ,   3752 , 363969 ,1084893 ,    185 , 244859 , 557498 , 917565 , 153345 , 189705 , 594064 ,1330469 , 418284 ,     25 ,  89291])

def getVoxel(seedPoint,rad,tree):
	#print('Extracting with rad %f'%rad)
	ind = tree.query_radius(seedPoint.reshape(1,-1),r=rad)
	point_ids=np.expand_dims(ind,axis=0)[0,0].reshape(1,-1)
	#print(point_ids.shape)
	#print(scene_cloud[point_ids[0,:],:].shape)
	return point_ids[0,:]

def walk_extra_dirs(path=extra_dirs):
	file_dict=[]
	addDir=False
	dirToAdd=''
	for i in range(len(path)):
		walk_dir=path[i]
		walk_dir = os.path.abspath(walk_dir)
		for root, subdirs, files in os.walk(walk_dir):
			for filename in files:
				if "All_affordances" not in filename:
					continue
				else:
					file_dict.append(root+'/')
	return file_dict



def getResults(descriptor_id):
	list_of_files=dict()
	extended_results=result_dirs+walk_extra_dirs()
	for i in range(len(extended_results)):
		aDir=extended_results[i]+'All_affordances_*_'+str(descriptor_id)+'_*.pcd'
		someFiles=glob.glob(aDir)
		for file in someFiles:
			tokens=file.split('_')
			last_token=tokens[-1].split('.')
			file_id=last_token[0]
			#check that data file exist
			dataPoints=extended_results[i]+file_id+'_goodPoints.pcd'
			dataFile=extended_results[i]+file_id+'_goodPointsX.pcd'
			if not os.path.exists(dataPoints) or not os.path.exists(dataFile):
				continue
			else:
				if file_id not in list_of_files:
					list_of_files[file_id]=extended_results[i]
				else:
					continue
		#list_of_files.append(someFiles)
		#print(someFiles)
	return list_of_files
	#list_of_files = [file for sublist in list_of_files for file in sublist]


def computeResultStats(descriptor_id):
	file_ids=getResults(descriptor_id)
	print('Found %d actual results'%(len(file_ids)))
	path=os.path.abspath(result_dirs[0])
	print(path)
	file_descriptor=path+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])
	counts=np.zeros((labels.shape[0],1),dtype=np.int32)
	countsFile="Counts_"+str(descriptor_id)+".csv"
	if not 'some_counts' in globals():
		# collect some data about affordances found here
		counter=0
		bar = Bar('Creating new data', max=len(file_ids))
		for file_id in file_ids:
			#read results
			some_results=file_ids[file_id]+file_id+"_goodPointsX.pcd"
			#print('File to read: %s'%some_results)
			some_results_points=file_ids[file_id]+file_id+"_goodPoints.pcd"
			newDataName=file_ids[file_id]+file_id+"_newData.csv"
			#if not os.path.exists(newDataName):
			try:
				# read_routine=1
				# with open(some_results_points) as fp:
				# 	for i, line in enumerate(fp):
				# 		if i == 10:
				# 			words=line.split(" ")
				# 			if words[1]!="ascii":
				# 				read_routine=2
				# 			break
				data,_=load_pcd_data_binary(some_results)
				points,real_c_data=load_pcd_data_binary(some_results_points)
			except Exception as e:
				print('Encoding error in %s'%(file_ids[file_id]+file_id))
				continue
				bar.next()
				
			#real_c_data=np.array(colors[:,-1],dtype=np.int32)
			red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			real_c_data=np.concatenate((red,green,blue),axis=1)
			perPoint=np.sum(real_c_data,axis=1)
			bounds=np.cumsum(perPoint)

			#Only get points above a height
			minZ=np.min(points[:,2])
				
			all_data=np.zeros((data.shape[0],6))
			start_id=0
			end_id=bounds[0]
			for i in range(bounds.shape[0]):
				if i>0:
					start_id=bounds[i-1]
				else:
					start_id=0
				end_id=bounds[i]
				all_data[start_id:end_id,:3]=points[i,:]
				all_data[start_id:end_id,3:]=data[start_id:end_id,:3]
					
			valid_ids=np.nonzero(all_data[:,Z]>=(minZ+0.3))[0]
			data=all_data[valid_ids,:]
			np.savetxt(newDataName, data, delimiter=",",fmt='%1.6f')
			#else:
				#data=np.genfromtxt(newDataName,delimiter=",",dtype='float32')
				#np.savetxt(newDataName,data,delimiter=",",fmt='%1.6f')
			counter+=1
			counts_tmp=np.bincount(data[:,A_ID].astype(int),minlength=counts.shape[0])
			counts_tmp=np.expand_dims(counts_tmp,axis=1)
			counts+=counts_tmp
			bar.next()
		bar.finish()
	else:
		counts=some_counts
	with open(countsFile, "w") as text_file:
		for i in range(labels.shape[0]):
			text_file.write("%d,%s-%s,%d\n" % (i,labels[i,0],labels[i,2],counts[i]))
	


def readCounts(file='Counts_13.csv'):
	interactions=np.genfromtxt(file,dtype='str',delimiter=',',usecols=(1))
	counts=np.genfromtxt(file,dtype='int32',delimiter=',',usecols=(2))
	names=[]
	[names.append([]) for x in range(counts.shape[0])]
	file_ids=getResults(13)
	newCounts=np.zeros((counts.shape[0],1))
	bar = Bar('Reading new data files',max=len(file_ids)+1)
	for file_id in file_ids:
		newDataName=file_ids[file_id]+file_id+"_newData.csv"
		#print(newDataName)
		if os.path.exists(newDataName):
			try:
				data=np.genfromtxt(newDataName,delimiter=",",dtype='float32',usecols=(3,5))
			except Exception as e:
				print('File empty? \n %s'%e)
				bar.next()
				continue
			if data.size<1:
				print('File empty? \n %s'%newDataName)
				bar.next()
				continue
			for aff_id in range(counts.shape[0]):
				#print(aff_id)
				ids=np.nonzero(data[:,0]==aff_id)[0]
				if ids.size > 0:
					if interactions[aff_id]=="Riding-biker2":
						minV=0.5
					else:
						minV=np.percentile(data[ids,1],30)
					relevant_ids=np.nonzero(data[ids,1]>=minV)[0]
					if relevant_ids.size>0:
						fileWithData=file_ids[file_id]+file_id
						names[aff_id].append(fileWithData)
						newCounts[aff_id,0]+=relevant_ids.size
		bar.next()
	bar.finish()
	sorted_ids=np.argsort(newCounts)
	print(newCounts)
	for i in range(newCounts.shape[0]):
		print('%s %d'%(interactions[sorted_ids[i]],newCounts[sorted_ids[i]].astype(int)))
	return newCounts,names
	
def sampleFromFile(affordance,list_of_files,number_of_samples,pointsPerCloud=4096):
	file_options=np.arange(len(list_of_files))
	files_to_sample=np.random.randint(len(list_of_files),size=(1,number_of_samples))
	repeated=np.bincount(files_to_sample[0,:],minlength=len(list_of_files))
	actually_sample_files=np.nonzero(repeated)[0]
	dataPoints=np.empty((number_of_samples, 6), dtype=np.float)
	dataClouds=np.empty((number_of_samples,pointsPerCloud,3),dtype=np.float32)
	start_id=0
	actually_sampled=0
	outOfPoints=False
	bar = Bar('Sampling ',max=number_of_samples)
	for i in range(actually_sample_files.size):
		file=list_of_files[actually_sample_files[i]]+"_newData.csv"
		pos=file.rfind('/')+1
		if "space/" in file:
			#Need to search for the exact file
			pos_id=list_of_files[actually_sample_files[i]].rfind('/')+1
			target_file_id=list_of_files[actually_sample_files[i]][pos_id:]
			path_to_scene=file[:pos_id]+'All_affordances_*_'+target_file_id+'.pcd'
			someFile=glob.glob(path_to_scene)
			tokens=someFile[0].split('_')
			cloud_file=list_of_files[actually_sample_files[i]][:pos_id]+tokens[2]
			if "real" in tokens[2]:
				cloud_file=cloud_file+".pcd"
			else:
				cloud_file=cloud_file+"_d.pcd"
				#if "readingroom" in cloud_file:
				#print(list_of_files[actually_sample_files[i]])
				#print(cloud_file)
				#sys.exit()
		else:
			pos_id=list_of_files[actually_sample_files[i]].rfind('/')+1
			target_file_id=list_of_files[actually_sample_files[i]][pos_id:]
			if "DATA" in file[:pos_id]:
				path_to_scene=file[:pos_id]+'*_clean.pcd'
				someFile=glob.glob(path_to_scene)
				cloud_file=someFile[0]
			else:
				path_to_scene=file[:pos_id]+'All_affordances_*_'+target_file_id+'.pcd'
				someFile=glob.glob(path_to_scene)
				tokens=someFile[0].split('_')			
				cloud_file=list_of_files[actually_sample_files[i]][:pos_id]+tokens[2]+'.pcd'
				#print(cloud_file)
				#sys.exit()
		sample_from_file=repeated[actually_sample_files[i]]
		data=np.genfromtxt(file,delimiter=",",dtype='float32')
		target_ids=np.nonzero(data[:,A_ID].astype(int)==affordance)[0]
		sorted_subset=np.argsort(data[target_ids,SCORE])
		sorted_subset=sorted_subset[::-1]
		j=0
		k=0
		complete_sample=False
		if not os.path.exists(cloud_file):
			print('No input cloud %s'%(cloud_file))
			return np.empty((0,6)),np.empty((0,0,0))
		cloud,_=load_pcd_data_binary(cloud_file)
		kdt = BallTree(cloud, leaf_size=5,metric='euclidean')
		while not complete_sample:
			#take points until conplete set
			dataPoints[start_id+j,:]=data[target_ids[sorted_subset[k]],:]
			point=dataPoints[start_id+j,:3]
			voxel_ids=getVoxel(point,max_rad,kdt)
			voxel=cloud[voxel_ids,:]
			actual_voxel_size=voxel.shape[0]
			if actual_voxel_size<(pointsPerCloud/4):
				#bad point, get a new one
				if k==0:
					print("\n File %s"%(cloud_file))
				outputText="Voxel "+str(voxel.shape[0])+" "+str(k)+"/"+str(sorted_subset.shape[0])
				print(outputText,end='\r')
					#print('\nFile: %s bad point %d/%d\r'%(someFile[0],k,sorted_subset.shape[0]))
					#print('bad point %d of %d Voxel: %d'%(k,sorted_subset.shape[0],voxel.shape[0]))
				k+=1
				if k>=sorted_subset.shape[0]:
					outOfPoints=True
					print('Exhausted File')
					break
			else:
				if actual_voxel_size>=pointsPerCloud:
					sample=sample_cloud(voxel,pointsPerCloud)
				else:
					print('padding')
					padding=point+np.zeros((pointsPerCloud-actual_voxel_size,3),dtype=np.float32)
					sample=np.concatenate((padding,voxel),axis=0)
				#center cloud
				dataClouds[start_id+j,...]=sample-point
				j+=1
				#print('\tVoxel size (%d,%d) SampleSize(%d,%d) start_id %d +j %d'%(voxel.shape[0],voxel.shape[1],sample.shape[0],sample.shape[1],start_id,j))
			if j==sample_from_file:
				complete_sample=True
		if not outOfPoints:
			start_id+=sample_from_file
			actually_sampled+=sample_from_file
			bar.next(sample_from_file)
		else:
			break;
	bar.finish()
	if outOfPoints or actually_sampled!=number_of_samples:
		return np.empty((0,6)),np.empty((0,0,0))
	else:
		return dataPoints,dataClouds




def rotate_point_cloud_by_angle(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
	Input:
    BxNx3 array, original batch of point clouds
    Return:
	BxNx3 array, rotated batch of point clouds"""
	
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	print(batch_data.shape)
	for k in range(batch_data.shape[0]):
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
    	rotation_matrix = np.array([[cosval, -sinval,0],
                                    [sinval, cosval,0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data

def load_ply_data(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def getSingleTraining(file):
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
	# print(fileId)
	# # Need only those affordances that have
	# # over 128 good predictions in this result file

	# res_data_file=path[:pos]+'/'+fileId+'_goodPointsX.pcd'
	# res_points_file=path[:pos]+'/'+fileId+'_goodPoints.pcd'

	# data=load_pcd_data(res_data_file,cols=None)
	# #print(data.shape)
	# points,real_c_data=load_pcd_data_binary(res_points_file)
	# #real_c_data=load_pcd_data(res_points_file,cols=(3,),dataType=np.uint32)
	# #real_c_data=np.array(colors[:,-1],dtype=np.int32)
	# red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	# green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	# blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)

	# real_c_data=np.concatenate((red,green,blue),axis=1)

	# perPoint=np.sum(real_c_data,axis=1)
	# bounds=np.cumsum(perPoint)
	# #print(bounds)
	# howMany=np.zeros((labels.shape[0],1),dtype=np.int32)
	# all_data=np.zeros((data.shape[0],6))

	# for i in range(all_data.shape[0]):
	# 	point_id=np.nonzero(bounds>i)[0][0]
	# 	all_data[i,:3]=points[point_id,:]
	# 	all_data[i,3:]=data[i,:3]


	# for i in range(labels.shape[0]):
	# 	success=np.nonzero(all_data[:,3]==i)[0]
	# 	#success2=np.nonzero(all_data[success,2]>0.2)[0]
	# 	howMany[i]=success.size

	# ids_target=np.nonzero(howMany>n_samples)[0]
	# print('Real found: %d'%ids_target.size)
	# print(ids_target)
	#sys.exit()



	new_c=np.genfromtxt('filtered_counts2.csv',delimiter=',',dtype='int')
	with open('file_lists2.csv', 'r') as f:
		reader=csv.reader(f)
		new_n=list(reader)

	samples=32
	points=4096
	ids_target=np.nonzero(new_c>=samples)[0]
	print('Actually using %d affordances'%(ids_target.size))

	fig = plt.figure()
	plt.ion()
	ax = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')
	unique_scenes=dict()
	k=10
	#ax.hold(False)
	if k>1:
		bar = Bar('Creating original single example training dataset', max=ids_target.shape[0])
		for i in range(ids_target.shape[0]):
			interaction=ids_target[i]
			path_to_data=os.path.abspath('../data')
			name=path_to_data+'/affordances/binaryOc_AffordancesDataset_train'+str(interaction)+'_'+str(TRAIN_EXAMPLES)+'.h5'
			if os.path.exists(name):
				continue
			#find training data
			aff_dir=labels[interaction,0]
			query_object=labels[interaction,2]
			data_file=path[:pos]+"/"+aff_dir+"/ibs_full_"+labels[interaction,1]+"_"+query_object+".txt"
			with open(data_file) as f:
				content = f.readlines()
				# you may also want to remove whitespace characters like `\n` at the end of each line
			content = [x.strip() for x in content] 
			scene_file=content[0].split(":")[1]
			tmp=content[8].split(":")[1]
			datapoint=tmp.split(',')
			test_point=np.expand_dims(np.asarray([float(x) for x in datapoint]),axis=0)
			data_file=path[:pos]+"/"+aff_dir+"/"+scene_file
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
				cloud_training=load_pcd_data(data_file)
			else:
				cloud_training=load_ply_data(data_file)
			data=np.zeros((2,n_points,3),dtype=np.float32)
			data_labels=np.zeros((2,1),dtype=np.int32)
			boundingBoxDiag=np.linalg.norm(np.min(cloud_training,0)-np.max(cloud_training,0))
			#print('%s Diagonal %f Points %d'%(scene_file,boundingBoxDiag,cloud_training.shape[0]))
			#sample a voxel with rad from test-point
			kdt = BallTree(cloud_training, leaf_size=5,metric='euclidean')
			voxel_ids=getVoxel(test_point,max_rad,kdt)
			voxel=cloud_training[voxel_ids,:]
			sample=sample_cloud(voxel,n_points)
			sample_cloud_training=sample_cloud(cloud_training,n_points*2)
			#genereate a negative example with noise around test_point
			low=test_point[0,0]-max_rad
			high=test_point[0,0]+max_rad
			tmp1=(high - low) * np.random.random_sample((n_points, 1)) + (low)
			low=test_point[0,1]-max_rad
			high=test_point[0,1]+max_rad
			tmp2=(high - low) * np.random.random_sample((n_points, 1)) + (low)
			low=test_point[0,2]-max_rad
			high=test_point[0,2]+max_rad
			tmp3=(high - low) * np.random.random_sample((n_points, 1)) + (low)
			negative_cloud_training=np.concatenate((tmp1,tmp2,tmp3),axis=1)
			data[0,...]=sample-test_point
			data_labels[0,...]=np.zeros((1,1),dtype=np.int32)
			data[1,...]=negative_cloud_training-test_point
			data_labels[1,...]=np.ones((1,1),dtype=np.int32)
			#name=path_to_data+'/affordances/binaryOc_AffordancesDataset_train'+str(interaction)+'_'+str(TRAIN_EXAMPLES)+'.h5'
			#print(name)
			save_h5(name,data,data_labels,'float32','uint8')
			ax.scatter(sample_cloud_training[:,0],sample_cloud_training[:,1],sample_cloud_training[:,2],s=1,c='b')
			ax.scatter(sample[:,0],sample[:,1],sample[:,2],s=3,c='b')
			ax2.scatter(negative_cloud_training[:,0],negative_cloud_training[:,1],negative_cloud_training[:,2],s=3,c='r')
			plt.pause(1)
			plt.draw()
			ax.clear()
			ax2.clear()
			bar.next()
		bar.finish()
	name='../data/affordances/names.txt'
	with open(name, "w") as text_file:
		for i in range(ids_target.shape[0]):
			text_file.write("%d:%s-%s\n" % (i,labels[ids_target[i],0],labels[ids_target[i],2])) 
		#ax.hold(True)


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
