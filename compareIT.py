import os
import numpy as np
import socket
import pypcd
import glob
import sys
import h5py
import matplotlib.pyplot as plt
from progress.bar import Bar
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import BallTree
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

host=socket.gethostname()

if host=="it057384":
	data_paths=['/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/','/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/']
else:
	data_paths=['/home/hal/Eduardo/voronoi/build/','/home/skynet/Eduardo/GPU_Tensor/voronoi/build/']


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

def load_ply_data(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

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

def genereateNoisyData(anchor,rad,d_points,x_samples):
	low=anchor[0,0]-rad
	high=anchor[0,0]+rad
	tmp1=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
	low=anchor[0,1]-rad
	high=anchor[0,1]+rad
	tmp2=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
	low=anchor[0,2]-rad
	high=anchor[0,2]+rad
	tmp3=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
	data=np.concatenate((tmp1,tmp2,tmp3),axis=2)
	return data


def load_agglo_res(scene_index=0,agglo_descriptors=(999,13)):
	if scene_index>=len(data_paths):
		print('Check scene index')
		sys.exit()
	target_files=[]
	target_ids=[]
	for descriptor_id in agglo_descriptors:
		file1=data_paths[scene_index]+'All_affordances_*_'+str(descriptor_id)+'*.pcd'
		resFiles=glob.glob(file1)
		#recentFile=''
		#recentId=''
		if len(resFiles)>1:
			largest=0
			for file in resFiles:
				tok=file.split('_')[-1].split('.')[0]
				if int(tok)>largest:
					largest=int(tok)
					recentFile=file
					recentId=tok
		else:
			recentFile=resFiles[0]
			tok=recentFile.split('_')[-1].split('.')[0]
			recentId=tok
		target_files.append(recentFile)
		target_ids.append(recentId)
	print(target_ids)
	#sample by cnn and it is are the same, so need only to read one of those.
	sampled_cnn_file=data_paths[scene_index]+target_ids[0]+'_samplePointsIds.pcd'
	print(sampled_cnn_file)
	pc=pypcd.PointCloud.from_path(sampled_cnn_file)
	sampled_cnn=pc.pc_data['id']
	sampled=np.arange(sampled_cnn.size)

	#read good data for cnn
	cnn_file=data_paths[scene_index]+target_ids[0]+'_goodPoints.pcd'
	point_data_cnn,real_c_data,_=load_pcd_data_binary(cnn_file)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	real_c_data=np.concatenate((red,green,blue),axis=1)
	perPointDetectionsCNN=np.sum(real_c_data,axis=1)
	boundsCNN=np.cumsum(perPointDetectionsCNN)
	cnn_file=data_paths[scene_index]+target_ids[0]+'_goodPointsX.pcd'
	data_cnn,_,_=load_pcd_data_binary(cnn_file)
	good_cnn1_file=data_paths[scene_index]+target_ids[0]+'_goodPointsIds.pcd'
	pc=pypcd.PointCloud.from_path(good_cnn1_file)
	good_cnn1_ids=pc.pc_data['id']

	actual_boundsCNN=np.zeros((data_cnn.shape[0],1),dtype=np.int32)
	start_i=0
	for i in range(boundsCNN.shape[0]):
		end_i=boundsCNN[i]
		actual_boundsCNN[start_i:end_i,0]=i
		start_i=end_i
	#do a similar thing for it
	it_file=data_paths[scene_index]+target_ids[1]+'_goodPoints.pcd'
	point_data_it,real_c_data,_=load_pcd_data_binary(it_file)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	real_c_data=np.concatenate((red,green,blue),axis=1)
	perPointDetectionsIT=np.sum(real_c_data,axis=1)
	boundsIT=np.cumsum(perPointDetectionsIT)
	it_file=data_paths[scene_index]+target_ids[1]+'_goodPointsX.pcd'
	data_it,_,_=load_pcd_data_binary(it_file)
	good_it_file=data_paths[scene_index]+target_ids[1]+'_goodPointsIds.pcd'
	pc=pypcd.PointCloud.from_path(good_it_file)
	good_it_ids=pc.pc_data['id']
	actual_boundsIT=np.zeros((data_it.shape[0],1),dtype=np.int32)
	start_i=0
	for i in range(boundsIT.shape[0]):
		end_i=boundsIT[i]
		actual_boundsIT[start_i:end_i,0]=i
		start_i=end_i

	#affordances in descriptor
	itDescriptor_file=data_paths[scene_index]+"tmp"+str(agglo_descriptors[1])+".csv"
	cnnDescriptor_file=data_paths[scene_index]+"tmp"+str(agglo_descriptors[0])+".csv"
	labels_it=np.genfromtxt(itDescriptor_file,dtype='str',skip_header=1,delimiter=',')
	labels_cnn=np.genfromtxt(cnnDescriptor_file,dtype='str',skip_header=1,delimiter=',')
	affordance_ids=np.zeros((labels_it.shape[0],2),dtype=np.int32)-1
	c=0
	for i in range(labels_it.shape[0]):
		for j in range(labels_cnn.shape[0]):
			lab_it=labels_it[i,0]+'-'+labels_it[i,2]
			lab_cnn=labels_cnn[j,0]+'-'+labels_cnn[j,2]
			if lab_it==lab_cnn:
				affordance_ids[c,0]=i
				affordance_ids[c,1]=j
				c+=1
				break
	tmp_ids=np.nonzero(affordance_ids[:,0]>=0)[0]
	affordance_ids=affordance_ids[tmp_ids,:]
	data_out=np.zeros((tmp_ids.size,5),dtype=np.float32)
	bar = Bar('Going through data',max=tmp_ids.size)
	for i in range(tmp_ids.size):
		#recover cnn data
		affordance_id_cnn=affordance_ids[i,1]
		#filtered by height
		# filtered_height_ids=np.nonzero(point_data_cnn[:,2]>=0.3)[0]
		# #get the
		affordance_ids_cnn=np.nonzero(data_cnn[:,0]==affordance_id_cnn)[0]
		#these are from 0 to goodPoints size, need to remap to sampledIds
		some_ids=np.unique(actual_boundsCNN[affordance_ids_cnn,0])
		pP=good_cnn1_ids[some_ids]
		pN=np.setdiff1d(sampled,pP)
		#recover it data
		affordance_id_it=affordance_ids[i,0]
		affordance_ids_it=np.nonzero(data_it[:,0]==affordance_id_it)[0]
		#these are from 0 to goodPoints size, need to remap to sampledIds
		some_ids=np.unique(actual_boundsIT[affordance_ids_it,0])
		aP=good_it_ids[some_ids]
		if aP.size==0:
			bar.next()
			continue
		aN=np.setdiff1d(sampled,aP)

		TP=np.intersect1d(pP,aP)
		#print('%s-%s %d %d'%(labels_cnn[affordance_ids[i,1],0],labels_cnn[affordance_ids[i,1],1],TP.size,aP.size))
		data_out[i,3]=TPR=TP.size/float(aP.size)
		FP=np.setdiff1d(pN,aN)
		data_out[i,4]=FPR=FP.size/float(aP.size)
		TN=np.intersect1d(pN,aN)
		FN=np.setdiff1d(pN,aN)
		TNR=TN.size/float(aN.size)
		FNR=FN.size/float(aN.size)
		data_out[i,1]=precision=TP.size/float(TP.size+FP.size)
		data_out[i,2]=recall=TP.size/float(TP.size+FN.size)
		data_out[i,0]=accuracy=(TP.size+TN.size)/float(aP.size+aN.size)

		bar.next()
	bar.finish()
	return data_out


def cnnStats(agglo_descriptor_ids=(999,13)):
	some_data=load_agglo_res(scene_index=1,agglo_descriptors=agglo_descriptor_ids)
	np.savetxt('test.csv',some_data,delimiter=',',fmt='%.4f',header='accuracy,precision,recall,TPR,FPR')
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	ids_target=np.nonzero(new_c>=samples)[0]

	path=os.path.abspath('/home/er13827/space/testing/tmp.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')

	scenes=['real-kitchen1','kitchen5']
	scenes_extension=['pcd','ply']
	data_points=np.empty((ids_target.size,2),dtype=np.float32)
	data_points_extra=np.empty((ids_target.size,3),dtype=np.float32)
	bar = Bar('Reading results',max=ids_target.size)
	data_name='basic_stats.npy'
	data_name_2='basic_stats_affordance.npy'
	data_name_3='basic_stats_extra.npy'
	data_name_4='basic_stats_extra_affordance.npy'
	per_affordance_data_points=np.zeros((4,2),dtype=np.float32)
	per_affordance_data_points_extra=np.zeros((4,3),dtype=np.float32)
	per_affordance_interactions=np.zeros((4,1),dtype=np.int32)
	last_read=''
	c=-1
	plot_labels=[x[0]+'-'+x[-1] for x in labels[ids_target,:]]
	plot_labels2=['Filling','Hanging','Placing','Sitting']
	if not os.path.exists(data_name) or not os.path.exists(data_name_2) or not os.path.exists(data_name_3) or not os.path.exists(data_name_4):
		for i in range(ids_target.size):
			#get target interaction
			interaction=ids_target[i]
			affordance=labels[interaction,1]
			obj=labels[interaction,2]
			#cnn scene 1
			cnn_file_1=affordance+'_'+obj+'_'+scenes[0]+'_*_999_*.pcd'
			path_to_scene=data_paths[0]+cnn_file_1
			someFile=glob.glob(path_to_scene)
			cnn_id1=someFile[0].split('_')[-1].split('.')[0]
			sampled_cnn1_file=data_paths[0]+cnn_id1+'_samplePointsIds.pcd'
			pc=pypcd.PointCloud.from_path(sampled_cnn1_file)
			sampled_ids_cnn1=pc.pc_data['id']
			sampled_cnn1=np.arange(sampled_ids_cnn1.shape[0])
			del sampled_ids_cnn1
			good_cnn1_file=data_paths[0]+cnn_id1+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(good_cnn1_file)
			good_cnn1_ids=pc.pc_data['id']
			#cnn scene 2
			cnn_file_2=affordance+'_'+obj+'_'+scenes[1]+'_*_999_*.pcd'
			path_to_scene=data_paths[1]+cnn_file_2
			someFile=glob.glob(path_to_scene)
			cnn_id2=someFile[0].split('_')[-1].split('.')[0]
			sampled_cnn2_file=data_paths[1]+cnn_id2+'_samplePointsIds.pcd'
			pc=pypcd.PointCloud.from_path(sampled_cnn2_file)
			sampled_ids_cnn2=pc.pc_data['id']
			sampled_cnn2=np.arange(sampled_ids_cnn2.shape[0])
			del sampled_ids_cnn2
			good_cnn2_file=data_paths[1]+cnn_id2+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(good_cnn2_file)
			good_cnn2_ids=pc.pc_data['id']
			#it scene 1
			it_file_1=affordance+'_'+obj+'_'+scenes[0]+'_*_13_*.pcd'
			path_to_scene=data_paths[0]+it_file_1
			someFile=glob.glob(path_to_scene)
			it_id1=someFile[0].split('_')[-1].split('.')[0]
			sampled_it1_file=data_paths[0]+it_id1+'_samplePointsIds.pcd'
			pc=pypcd.PointCloud.from_path(sampled_it1_file)
			sampled_ids_it1=pc.pc_data['id']
			sampled_it1=np.arange(sampled_ids_it1.shape[0])
			del sampled_ids_it1
			good_it1_file=data_paths[0]+it_id1+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(good_it1_file)
			good_it1_ids=pc.pc_data['id']
			#it scene 2
			it_file_2=affordance+'_'+obj+'_'+scenes[1]+'_*_13_*.pcd'
			path_to_scene=data_paths[1]+it_file_2
			someFile=glob.glob(path_to_scene)
			it_id2=someFile[0].split('_')[-1].split('.')[0]
			sampled_it2_file=data_paths[1]+it_id2+'_samplePointsIds.pcd'
			pc=pypcd.PointCloud.from_path(sampled_it2_file)
			sampled_ids_it2=pc.pc_data['id']
			sampled_it2=np.arange(sampled_ids_it2.shape[0])
			del sampled_ids_it2
			good_it2_file=data_paths[1]+it_id2+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(good_it2_file)
			good_it2_ids=pc.pc_data['id']

			TP1=np.intersect1d(good_cnn1_ids,good_it1_ids)
			TPR1=TP1.size/float(good_it1_ids.size)
			FP1=np.setdiff1d(good_cnn1_ids,good_it1_ids)
			FPR1=FP1.size/float(good_it1_ids.size)
			pN1=np.setdiff1d(sampled_cnn1,good_cnn1_ids)
			aN1=np.setdiff1d(sampled_it1,good_it1_ids)
			TN1=np.intersect1d(pN1,aN1)
			FN1=np.setdiff1d(pN1,aN1)
			TNR1=TN1.size/float(aN1.size)
			FNR1=FN1.size/float(aN1.size)
			precision1=TP1.size/float(TP1.size+FP1.size)
			recall1=TP1.size/float(TP1.size+FN1.size)
			accuracy1=(TP1.size+TN1.size)/float(good_it1_ids.size+aN1.size)

			pN2=np.setdiff1d(sampled_cnn2,good_cnn2_ids)
			aN2=np.setdiff1d(sampled_it2,good_it2_ids)
			TN2=np.intersect1d(pN2,aN2)
			FN2=np.setdiff1d(pN2,aN2)
			TNR2=TN1.size/float(aN2.size)
			FNR2=FN1.size/float(aN2.size)
			TP2=np.intersect1d(good_cnn2_ids,good_it2_ids)
			TPR2=TP2.size/float(good_it2_ids.size)
			FP2=np.setdiff1d(good_cnn2_ids,good_it2_ids)
			FPR2=FP2.size/float(good_it2_ids.size)
			precision2=TP2.size/float(TP2.size+FP2.size)
			recall2=TP2.size/float(TP2.size+FN2.size)
			accuracy2=(TP2.size+TN2.size)/float(good_it2_ids.size+aN2.size)

			avg_prec=np.mean([precision1,precision2])
			avg_recall=np.mean([recall2,recall1])
			avg_accuracy=np.mean([accuracy1,accuracy2])
			avg_TPR=avg_recall
			avg_FPR=np.mean([FPR1,FPR2])
			#print('%s %s mean-prec:%.2f mean-recall:%.2f'%(affordance,obj, avg_prec,avg_recall))
			data_points[i,0]=avg_prec
			data_points[i,1]=avg_recall
			data_points_extra[i,0]=avg_accuracy
			data_points_extra[i,1]=avg_TPR
			data_points_extra[i,2]=avg_FPR

			if last_read==affordance:
				per_affordance_data_points[c,1]+=avg_recall
				per_affordance_data_points[c,0]+=avg_prec
				per_affordance_data_points_extra[c,0]+=avg_accuracy
				per_affordance_data_points_extra[c,1]+=avg_TPR
				per_affordance_data_points_extra[c,2]+=avg_FPR
				per_affordance_interactions[c,0]+=1
			else:
				c+=1
				per_affordance_data_points[c,1]+=avg_recall
				per_affordance_data_points[c,0]+=avg_prec
				per_affordance_data_points_extra[c,0]+=avg_accuracy
				per_affordance_data_points_extra[c,1]+=avg_TPR
				per_affordance_data_points_extra[c,2]+=avg_FPR
				per_affordance_interactions[c,0]+=1
				last_read=affordance

			bar.next()
		np.save(data_name,data_points)
		per_affordance_data_points=per_affordance_data_points/per_affordance_interactions
		per_affordance_data_points_extra=per_affordance_data_points_extra/per_affordance_interactions
		np.save(data_name_2,per_affordance_data_points)
		np.save(data_name_3,data_points_extra)
		np.save(data_name_4,per_affordance_data_points_extra)
		bar.finish()
	else:
		data_points=np.load(data_name)
		per_affordance_data_points=np.load(data_name_2)
		data_points_extra=np.load(data_name_3)
		per_affordance_data_points_extra=np.load(data_name_4)
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)
	ax.scatter(data_points[:,1],data_points[:,0],c='g',label='perInteractionPR')
	ax2.scatter(per_affordance_data_points[:,1],per_affordance_data_points[:,0],c='b',label='perAffordacePR')
	ax3.plot(np.array([0,1]),np.array([0,1]),ls='--')
	ax4.plot(np.array([0,1]),np.array([0,1]),ls='--')
	ax3.scatter(data_points_extra[:,2],data_points_extra[:,1],c='r',label='perInteractionROC')
	ax4.scatter(per_affordance_data_points_extra[:,2],per_affordance_data_points_extra[:,1],c='b',label='perAffordaceROC')
	for i in range(ids_target.size):
		if data_points[i,0]<0.95:
			print('%s %f'%(plot_labels[i],data_points[i,0]))
			ax.text(data_points[i,1], data_points[i,0]+.02, plot_labels[i], horizontalalignment='right', size='medium', color='black', weight='semibold')
	for i in range(per_affordance_data_points.shape[0]):
		ax2.text(per_affordance_data_points[i,1], per_affordance_data_points[i,0]+.02, plot_labels2[i], horizontalalignment='center', size='medium', color='black', weight='semibold')
	print('=========================')
	for i in range(ids_target.size):
		if data_points_extra[i,2]>0.3 or data_points_extra[i,1]<0.7:
			print('%s %f'%(plot_labels[i],data_points_extra[i,1]))
			ax3.text(data_points_extra[i,2]+0.02, data_points_extra[i,1], plot_labels[i], horizontalalignment='left', size='medium', color='black', weight='semibold')

	for i in range(per_affordance_data_points_extra.shape[0]):
		ax4.text(per_affordance_data_points_extra[i,2], per_affordance_data_points_extra[i,1]+.02, plot_labels2[i], horizontalalignment='center', size='medium', color='black', weight='semibold')
	#ax.set_xticks(ind)
	#ax.set_xticklabels(plot_labels,rotation=45,ha="right")
	#ax.plot(ind,counts[sorted_ids,0],c='b',label='2048')
	#ax.plot(ind,counts[sorted_ids,1],c='g',label='1024')
	#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	ax.set_xlim(0,1)
	#ax3.set_xlim(0,1)
	ax2.set_xlim(0,1)
	ax3.set_xlim(0,1)
	ax4.set_xlim(0,1)

	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax2.set_xlabel('Recall')
	ax2.set_ylabel('Precision')
	ax3.set_xlabel('FPR')
	ax3.set_ylabel('TPR')
	ax4.set_xlabel('FPR')
	ax4.set_ylabel('TPR')

	ax.set_ylim(0,1)
	#ax3.set_ylim(0,1)
	ax2.set_ylim(0,1)
	ax3.set_ylim(0,1)
	ax4.set_ylim(0,1)
	ax.grid()
	ax2.grid()
	ax3.grid()
	ax4.grid()
	ax.set_axisbelow(True)
	ax2.set_axisbelow(True)
	ax3.set_axisbelow(True)
	ax4.set_axisbelow(True)
	plt.tight_layout()
	plt.show()

def getMultiLabelData(scene_id=0,agglo_descriptor=13, plot=False):
	n_training_examples=2048
	data_paths_new=['/home/er13827/deepNetwork/skynetDATA/Eduardo/Inividual/','/home/er13827/deepNetwork/skynetDATA/Eduardo/Inividual/','/home/er13827/deepNetwork/skynetDATA/Eduardo/Inividual/','/home/er13827/deepNetwork/skynetDATA/Eduardo/Individual/']
	scenes=['real-kitchen1','kitchen5','living-room6','real-kitchen2']
	base_name=data_paths_new[scene_id]+'[!A]*_'+scenes[scene_id]+'_*_'+str(agglo_descriptor)+'_*.pcd'
	res_files=glob.glob(base_name)
	res_files=sorted(res_files)
	# for i in range(len(res_files)):
	# 	print(res_files[i])
	#sys.exit()

	# some_ids_res=np.array([0,8,16,83])
	# res_files=[res_files[x] for x in some_ids_res]
	#print(new_res_files)
	#sys.exit()
	scene=data_paths_new[scene_id]+scenes[scene_id]
	scenes_extension=['pcd','ply','ply','pcd']
	#display_cloud,_,_=load_pcd_data_binary(scene+'.'+scenes_extension[scene_id])
	if scenes_extension[scene_id]=="ply":
		input_cloud=load_ply_data(scene+'.'+scenes_extension[scene_id])
		print('read cloud %s'%scene+'.'+scenes_extension[scene_id])
		scene=scene+'_d.pcd'
	else:
		input_cloud,_,_=load_pcd_data_binary(scene+'.'+scenes_extension[scene_id])
		print('read cloud %s'%scene+'.'+scenes_extension[scene_id])
		scene=scene+'.pcd'
		
	input_cloud_dense,_,_=load_pcd_data_binary(scene)
	print('read scene %s'%scene)
	kdt = BallTree(input_cloud_dense, leaf_size=5,metric='euclidean')
	min_z_scene=np.min(input_cloud[:,2])
	min_z=min_z_scene+0.3
	pointsPerCloud=4096
	max_rad=0.806884
	if plot:
		fig = plt.figure(figsize=(10, 10))
		plt.ion()
		ax = fig.add_subplot(111, projection='3d')
	# scene_to_disp=np.arange(display_cloud.shape[0])
	# np.random.shuffle(scene_to_disp)
	# scene_ids=scene_to_disp[:1000]
	# ax.scatter(display_cloud[scene_ids,0],display_cloud[scene_ids,1],display_cloud[scene_ids,2],s=1,c='b')
	useful_counts=np.zeros((len(res_files),1),dtype=np.int32)
	#n_sampled=input_cloud.shape[0]//3
	ORI_ID=4
	AFF_ID=3
	SCORE=5
	GOOD_ID=6
	SAMPLE_ID=7
	if not os.path.exists('common'+scenes[scene_id]+'.npy'):
		common=np.zeros((input_cloud.shape[0],len(res_files)),dtype=np.int32)
		negative_common=np.zeros((input_cloud.shape[0],len(res_files)),dtype=np.int32)
		sorted_common=np.zeros((input_cloud.shape[0],len(res_files)),dtype=np.int32)-1
		common_orientations=np.zeros((input_cloud.shape[0],len(res_files)),dtype=np.int32)
		bar = Bar('Reading results',max=len(res_files))
		interaction_labels=[]
		for j in range(len(res_files)):
			file=res_files[j]
			file_id=file.split('_')[-1].split('.')[0]
			pos=file.rfind('/')+1
			aff=file[pos:].split('_')[0]
			obj=file[pos:].split('_')[1]
			interaction_labels.append(aff+'-'+obj)
			tmp_file=data_paths_new[scene_id]+file_id+'_samplePointsIds.pcd'
			#assuming everyone has the same sample points
			pc=pypcd.PointCloud.from_path(tmp_file)
			if j>0:
				if pc.pc_data['id'].shape[0]!=sampled_ids.shape[0]:
					print('Odd?')
			sampled_ids=pc.pc_data['id']
			#print(type(sampled_ids[0]))
			tmp_file=data_paths_new[scene_id]+file_id+'_goodPointsIds.pcd'
			pc=pypcd.PointCloud.from_path(tmp_file)
			good_ids=pc.pc_data['id']
			
			#print(type(good_ids[0]))
			tmp_file=data_paths_new[scene_id]+file_id+'_goodPointsX.pcd'
			data,_,_=load_pcd_data_binary(tmp_file)
			tmp_file=data_paths_new[scene_id]+file_id+'_goodPoints.pcd'
			points,real_c_data,_=load_pcd_data_binary(tmp_file)
			red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
			real_c_data=np.concatenate((red,green,blue),axis=1)
			perPointDetections=np.sum(real_c_data,axis=1)
			bounds=np.cumsum(perPointDetections)
			start_i=0
			#actual_bounds=np.zeros((data.shape[0],1),dtype=np.int32)
			#large_data=np.array(np.zeros((data.shape[0],8)),dtype='f4,f4,f4,f4,f4,f4,u4,u4')
			large_data=np.zeros((data.shape[0],8),dtype=np.float32)
			large_data[:,3:6]=data
			for i in range(bounds.size):
				end_i=bounds[i]
				#goodPoint id
				large_data[start_i:end_i,GOOD_ID]=i
				large_data[start_i:end_i,:3]=points[i,:]
				#samplePoint id from 0 to inputCloud size
				large_data[start_i:end_i,SAMPLE_ID]=sampled_ids[good_ids[i]]
				start_i=end_i
			del bounds, perPointDetections,data,points,red,green,blue

			ok_heights=np.nonzero(large_data[:,2]>=min_z)[0]
			large_data=large_data[ok_heights,...]
			#print(np.min(large_data[:,GOOD_ID]),np.max(large_data[:,GOOD_ID]))
			#print(np.min(large_data[:,SAMPLE_ID]),np.max(large_data[:,SAMPLE_ID]))
			#print('Point id Min %d Max: %d'%(np.min(large_data[:,GOOD_ID]),np.max(large_data[:,GOOD_ID])))
			#print('Point id Min %d Max: %d'%(np.min(large_data[:,SAMPLE_ID]),np.max(large_data[:,SAMPLE_ID])))
			#filter based on score
			minV=np.percentile(large_data[:,SCORE],85)
			cutoff=np.percentile(large_data[:,SCORE],40)
			top_ids=np.nonzero(large_data[:,SCORE]>=minV)[0]
			bottom_ids=np.nonzero(large_data[:,SCORE]<=cutoff)[0]
			#print('Score before %.4f'%(large_data[top_ids[0],SCORE]))
			sorted_top_ids=np.argsort(large_data[top_ids,SCORE])
			sorted_bottom_ids=np.argsort(large_data[bottom_ids,SCORE])

			top_ids=top_ids[sorted_top_ids]
			bottom_ids=bottom_ids[sorted_bottom_ids]
			#reverse them to have them in decreasing order
			top_ids=top_ids[::-1]
			miniDATA=large_data[top_ids,...]
			miniData_neg=large_data[bottom_ids,...]
			#print('%d Scores after sorting %.4f %.4f'%(miniDATA.shape[0],miniDATA[0,SCORE],miniDATA[-1,SCORE]))
			point_ids=miniDATA[:,GOOD_ID]
			unique_points_sorted,actual_points_ids=np.unique(point_ids,return_index=True)
			new_miniDATA=miniDATA[actual_points_ids,...]
			sorted_ids=np.argsort(new_miniDATA[:,SCORE])
			sorted_ids=sorted_ids[::-1]
			new_miniDATA=new_miniDATA[sorted_ids,...]

			point_ids_neg=miniData_neg[:,GOOD_ID]
			unique_points_sorted_neg,actual_points_ids_neg=np.unique(point_ids_neg,return_index=True)
			miniData_neg=miniData_neg[actual_points_ids_neg,:]
			
			#print('%d Scores unique %.4f %.4f'%(new_miniDATA.shape[0],new_miniDATA[0,SCORE],new_miniDATA[-1,SCORE]))
			# this 'actual_points' are from 0 to goodPoints
			#print(new_miniDATA[:20,SAMPLE_ID],new_miniDATA[:20,SAMPLE_ID].astype(np.int32))
			#print('Point id Min %d Max: %d'%(np.min(new_miniDATA[:,GOOD_ID]),np.max(new_miniDATA[:,GOOD_ID])))
			#print('Point id Min %d Max: %d'%(np.min(new_miniDATA[:,SAMPLE_ID]),np.max(new_miniDATA[:,SAMPLE_ID])))

			# usefull_point_ids=np.intersect1d(extract_ids,filtered_height_ids)
		
			usefull_point_ids=new_miniDATA[:,SAMPLE_ID].astype(np.int32)
			usefull_orientations=new_miniDATA[:,ORI_ID]
			#print(new_miniDATA[:10,ORI_ID])
			sorted_useful=np.arange(usefull_point_ids.size)
			common[usefull_point_ids,j]=1
			common_orientations[usefull_point_ids,j]=usefull_orientations
			sorted_common[usefull_point_ids,j]=sorted_useful
			#print(usefull_point_ids[:10])
			#print(new_miniDATA[:10,SAMPLE_ID])
			useful_counts[j,0]=usefull_point_ids.size
			if usefull_point_ids.size<n_training_examples:
				print('\n%s-%s %d'%(aff,obj,usefull_point_ids.size))

			#get the negative/non response points
			usefull_point_ids_neg=miniData_neg[:,SAMPLE_ID].astype(np.int32)
			negative_common[usefull_point_ids_neg,j]=1
			#print('--------------------------------')
			bar.next()
			#print('\n%s-%s Total: %d Useful: %d Min:%.4f'%(aff,obj,sampled_ids.size,usefull_point_ids.size,minV))
			#bPoint=new_miniDATA[0,:3]
			#print(aPoint,bPoint)
			# #ax.scatter(input_cloud[usefull_point_ids[0],0],input_cloud[usefull_point_ids[0],1],input_cloud[usefull_point_ids[0],2]s=30,c='b')
			if plot:
				aPoint=input_cloud[usefull_point_ids[0],:]
				voxel_ids=getVoxel(aPoint,max_rad,kdt)
				voxel=input_cloud_dense[voxel_ids,:]
				sample=sample_cloud(voxel,pointsPerCloud)-aPoint
				ax.set_title(interaction_labels[j])
				v=ax.scatter(sample[:,0],sample[:,1],sample[:,2],s=1,c='b')
				vc=ax.scatter(0,0,0,s=30,c='r')
				plt.pause(3)
				plt.draw()
				v.remove()
				vc.remove()
			#ax.clear()
			# #get the top locations for verification
			# top_top_ids=np.argsort(data[top_ids,2])
			# top_top_bounds=actual_bounds[top_ids[top_top_ids[:100]],0]
			# top_extract_ids=sampled_ids[good_ids[top_top_bounds]]
			# good_points_top=input_cloud[top_extract_ids,:]
			# filtered_good_ids_top=np.nonzero(good_points_top[:,2]>min_z)[0]
			# good_points_top_filtered=good_points_top[filtered_good_ids_top,:]		
			# top_display=ax.scatter(good_points_top_filtered[:,0],good_points_top_filtered[:,1],good_points_top_filtered[:,2],s=10,c='r')
			# plt.pause(0.5)
			# plt.draw()
			# top_display.remove()
			# if aff=='Fill' and obj=="bowl":
			# 	print('SAVING!!!!')
			# 	actual_data_array=np.zeros(good_points_filtered.shape[0], dtype={'names':('x', 'y', 'z'),
  	#                         'formats':('f4', 'f4', 'f4')})
			# 	actual_data_array['x']=good_points_filtered[:,0]
			# 	actual_data_array['y']=good_points_filtered[:,1]
			# 	actual_data_array['z']=good_points_filtered[:,2]
			# 	new_cloud = pypcd.PointCloud.from_array(actual_data_array)
			# 	nname=data_paths[scene_id]+file_id+'_FilteredGood.pcd'
			# 	new_cloud.save_pcd(nname,compression='ascii')
		#sampled_points=input_cloud[sampled_ids,:]
		bar.finish()
		np.save('common'+scenes[scene_id]+'.npy',common)
		np.save('negative_common'+scenes[scene_id]+'.npy',negative_common)
		np.save('sorted_common'+scenes[scene_id]+'.npy',sorted_common)
		np.save('common_orientations'+scenes[scene_id]+'.npy',common_orientations)
		with open('common_names'+scenes[scene_id]+'.csv', "w") as text_file:
			for i in range(len(res_files)):
				text_file.write("%s\n" % (interaction_labels[i]))
			text_file.write("%s\n" % ('Non-affordance'))
	else:
		common=np.load('common'+scenes[scene_id]+'.npy')
		sorted_common=np.load('sorted_common'+scenes[scene_id]+'.npy')
		negative_common=np.load('negative_common'+scenes[scene_id]+'.npy')
		interaction_labels=np.genfromtxt('common_names'+scenes[scene_id]+'.csv',dtype='str',delimiter=',')
		common_orientations=np.load('common_orientations'+scenes[scene_id]+'.npy')
	chk_sum=np.sum(common,axis=1)
	#keep_ids goes from 0 to input_cloud_size
	keep_ids=np.nonzero(chk_sum)[0]
	#check non responses
	chk_sum=np.sum(negative_common,axis=1)
	all_negative=np.nonzero(chk_sum==len(res_files))[0]
	print('Negative examples %d'%all_negative.size)
	#common=common[keep_ids,:]
	per_affordance_points=np.sum(common[keep_ids,:],axis=0)
	sorted_ids=np.argsort(per_affordance_points)

	#sorted_ids=sorted_ids[::-1]
	#print('Min: %s %d Max: %s %d'%(interaction_labels[sorted_ids[0]],per_affordance_points[sorted_ids[0]],interaction_labels[sorted_ids[-1]],per_affordance_points[sorted_ids[-1]]))
	#start collecting data from lowest-response affordance
	
	training_examples_per_affordance=np.zeros((len(res_files),1),dtype=np.int32)
	#save the sampled_id for each point for each affordance so you can recover later the orientation and score, etc
	data_to_recover=np.zeros((common.shape[0],len(res_files)),dtype=np.uint8)
	extracted_voxels=0
	data=np.empty((10000,pointsPerCloud,3),dtype=np.float32)
	data_points=np.empty((10000,3),dtype=np.float32)
	labels=np.zeros((10000,len(res_files)+1),dtype=np.uint8)
	if plot:
		fig = plt.figure(figsize=(6, 6))
		plt.ion()
		ax = fig.add_subplot(111, projection='3d')
	already_sampled={}
	dataSet_orientations=np.zeros((10000,len(res_files)),dtype=np.int32)-1
	bar = Bar('Generating dataset',max=len(res_files))
	for i in range(len(res_files)):
		#get points for this affordance
		affordance_id=sorted_ids[i]
		if plot:
			print('Affordace %d %s'%(affordance_id,interaction_labels[affordance_id]))
		#these are based on input_cloud size
		ids=np.nonzero(common[:,affordance_id])[0]
		#get the order from sorted_matrix
		aff_sorted_ids=sorted_common[ids,affordance_id]
		#this should be increasing, i think no longer makes sense because of what I did with top_ids before
		aff_sorted_ids_ids=np.argsort(aff_sorted_ids)
		actually_sampled=training_examples_per_affordance[affordance_id,0]
		j=0
		while actually_sampled<n_training_examples and j<ids.size:
			#check point has not been sampled before
			if str(ids[aff_sorted_ids_ids[j]]) not in already_sampled:
				#get the first point
				test_point=input_cloud[ids[aff_sorted_ids_ids[j]],:]
				voxel_ids=getVoxel(test_point,max_rad,kdt)
				#voxel=cloud[voxel_ids,:]
				actual_voxel_size=voxel_ids.size
				if actual_voxel_size<pointsPerCloud:
					print('Bad point? Few points')
				else:
					#recover all the affordances in this point
					voxel=input_cloud_dense[voxel_ids,:]
					sample=sample_cloud(voxel,pointsPerCloud)-test_point
					if plot:
						if j==0:
							ax.scatter(sample[:,0],sample[:,1],sample[:,2],s=3)
							ax.scatter(0,0,0,s=40,c='r')
							ax.set_title(interaction_labels[affordance_id])
							plt.pause(1)
							plt.draw()
							ax.clear()
					data[extracted_voxels,...]=sample
					data_points[extracted_voxels,...]=test_point
					all_responses=np.nonzero(common[ids[aff_sorted_ids_ids[j]],:])
					labels[extracted_voxels,all_responses]=1
					dataSet_orientations[extracted_voxels,all_responses]=common_orientations[ids[aff_sorted_ids_ids[j]],all_responses]
					training_examples_per_affordance[all_responses,0]+=1
					actually_sampled+=1
					extracted_voxels+=1
					data_to_recover[ids[aff_sorted_ids_ids[j]],affordance_id]=1
					already_sampled[str(ids[aff_sorted_ids_ids[j]])]=ids[aff_sorted_ids_ids[j]]
			j+=1
		bar.next()
	bar.finish()
	print(training_examples_per_affordance.T)
	print('Before negatives %d'%extracted_voxels)
	#add 'negative' data
	mean_examples=np.mean(training_examples_per_affordance)
	negatives_to_add=int(mean_examples//1)
	if all_negative.size>0:
		if negatives_to_add>all_negative.size:
			negatives_to_add=all_negative.size
		for i in range(negatives_to_add):
			test_point=input_cloud[all_negative[i],...]
			voxel_ids=getVoxel(test_point,max_rad,kdt)
			actual_voxel_size=voxel_ids.size
			if actual_voxel_size<pointsPerCloud:
				print('Bad point? Few points')
				toGenerate=pointsPerCloud-actual_voxel_size
				someNoise=genereateNoisyData(np.array([[0,0,0]]),max_rad,toGenerate,1)
				voxel=input_cloud_dense[voxel_ids,:]
				sample=np.concatenate((someNoise,voxel),axis=0)
			else:
				voxel=input_cloud_dense[voxel_ids,:]
				sample=sample_cloud(voxel,pointsPerCloud)-test_point
			data[extracted_voxels,...]=sample
			data_points[extracted_voxels,...]=test_point
			labels[extracted_voxels,len(res_files)]=1
			extracted_voxels+=1
	else:
		for i in range(negatives_to_add):
			someNoise=genereateNoisyData(np.array([[0,0,0]]),max_rad,pointsPerCloud,1)
			sample=someNoise
			data[extracted_voxels,...]=sample
			data_points[extracted_voxels,...]=np.array([[0,0,0]])
			labels[extracted_voxels,len(res_files)]=1
			extracted_voxels+=1

	print('After negatives %d'%extracted_voxels)
	data=data[:extracted_voxels,...]
	data_points=data_points[:extracted_voxels,...]
	labels=labels[:extracted_voxels,...]
	orientations=dataSet_orientations[:extracted_voxels,...]
	name='MultilabelDataSet_'+scenes[scene_id]+'_points.npy'
	np.save(name,data_points)
	name='MultilabelDataSet_'+scenes[scene_id]+'.h5'
	# if os.path.exists(name):
	# 	os.system('rm %s' % (name))
	save_h5(name,data,labels,'float32','uint8')
	name='MultilabelDataSet_'+scenes[scene_id]+'_Orientations.npy'
	np.save(name,orientations)

def inspectData(data_name='MultilabelDataSet_kitchen5.h5',aff_names='common_nameskitchen5.csv',aff_id=0):
	data,labels=load_h5(data_name)
	interaction_labels=np.genfromtxt(aff_names,dtype='str',delimiter=',')
	max_rad=0.806884
	ok_ids=np.nonzero(labels[:,aff_id])[0]
	fig = plt.figure(figsize=(6, 6))
	plt.ion()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(0,0,0,s=30,c='r')
	ax.set_title(interaction_labels[aff_id])
	ax.set_xlim(-max_rad,max_rad)
	ax.set_ylim(-max_rad,max_rad)
	ax.set_zlim(-max_rad,max_rad)
	for j in range(ok_ids.size):
		aCloud=data[ok_ids[j],...]
		cloudExample=ax.scatter(aCloud[:,0],aCloud[:,1],aCloud[:,2],s=3,c='b')	
		print(labels[ok_ids[j],...])
		plt.pause(0.5)
		plt.draw()
		cloudExample.remove()



if __name__ == '__main__':
	#cnnStats()
	# for i in range(3):
 	getMultiLabelData(scene_id=0)
	#inspectData(data_name='MultilabelDataSet_real-kitchen1.h5',aff_names='common_namesreal-kitchen1.csv', aff_id=84)