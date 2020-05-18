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
	individual_path='/home/er13827/deepNetwork/skynetDATA/Eduardo/Individual/'
	agglo_paths=['/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/','/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/']
else:
	individual_path='/media/skynet/Eduardo/Inividual/'
	agglo_paths=['/home/hal/Eduardo/voronoi/build/','/home/skynet/Eduardo/GPU_Tensor/voronoi/build/']

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

def recoverTopIds(main_file):
	path_to_scene=individual_path+main_file
	someFile=glob.glob(path_to_scene)
	if len(someFile)>0:
		it_id1=someFile[0].split('_')[-1].split('.')[0]
		sampled_it1_file=individual_path+it_id1+'_samplePointsIds.pcd'
		pc=pypcd.PointCloud.from_path(sampled_it1_file)
		sampled_ids_it1=pc.pc_data['id']
		sampled_it1=np.arange(sampled_ids_it1.shape[0])
		# print('Scene 1 Min Max Sampled')
		# print(np.min(sampled_it1),np.max(sampled_it1))
		del sampled_ids_it1
		good_it1_file=individual_path+it_id1+'_goodPointsIds.pcd'
		#print(good_it1_file)
		pc=pypcd.PointCloud.from_path(good_it1_file)
		good_it1_ids=pc.pc_data['id']
		real_data_it1_file=individual_path+it_id1+'_goodPointsX.pcd'
		big_Data1,_,_=load_pcd_data_binary(real_data_it1_file)
		data_file=individual_path+it_id1+'_goodPoints.pcd'
		_,real_c_data,_=load_pcd_data_binary(data_file)
		red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
		green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
		blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
		real_c_data=np.concatenate((red,green,blue),axis=1)
		perPointDetections=np.sum(real_c_data,axis=1)
		bounds1=np.cumsum(perPointDetections)
		actual_bounds1=np.zeros((big_Data1.shape[0],1),dtype=np.int32)
		start_i=0
		for i in range(bounds1.shape[0]):
			end_i=bounds1[i]
			actual_bounds1[start_i:end_i,0]=i
			start_i=end_i
		del bounds1
		sorted_ids=np.argsort(big_Data1[:,2])
		sorted_ids=sorted_ids[::-1]
		top_size=sorted_ids.size
		top_ids=sorted_ids[:top_size]
		#print('Score: %f'%big_Data1[top_ids[0],2])
		#these top comes from 0 to goodPoints
		top_ids=actual_bounds1[top_ids]

		#good points and sampled points ids are in same range
		good_it1_ids=sampled_it1[good_it1_ids]
		return good_it1_ids[top_ids],sampled_it1
	else:
		return np.array([-1]),np.array([-1])


def cnnStats(agglo_descriptor_ids=(900,13)):
	#top_scores=
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	samples=512
	ids_target=np.nonzero(new_c>=samples)[0]

	path=os.path.abspath('/home/er13827/space/testing/tmp13.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	print(file_descriptor)
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')

	scenes=['real-kitchen1','kitchen5']
	scenes_extension=['pcd','ply']
	#top1, top5, top10, top15,top,30,top50,all
	data_points=np.empty((7,ids_target.size,2),dtype=np.float32)
	data_points_extra=np.empty((7,ids_target.size,3),dtype=np.float32)
	bar = Bar('Reading results',max=ids_target.size)
	data_name='basic_stats_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.h5'
	data_name_2='basic_stats_affordance2_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.h5'
	data_name_3='basic_stats_extra2_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.h5'
	data_name_4='basic_stats_extra_affordance2_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.h5'
	#top1, top5, top10, top15,top,30,top50,all
	per_affordance_data_points=np.zeros((7,4,2),dtype=np.float32)
	per_affordance_data_points_extra=np.zeros((7,4,3),dtype=np.float32)
	per_affordance_interactions=np.zeros((7,4,1),dtype=np.int32)

	last_read=''
	c=-1
	plot_labels=[x[0]+'-'+x[-1] for x in labels[ids_target,:]]
	plot_labels2=['Filling','Hanging','Placing','Sitting']
	#cnn results scene 1 -> real-kitchen1
	cnn_file_1='All_affordances_'+scenes[0]+'_*_'+str(agglo_descriptor_ids[0])+'_*.pcd'
	path_to_scene=agglo_paths[0]+cnn_file_1
	someFile=glob.glob(path_to_scene)
	print(someFile[0])
	cnn_id1=someFile[0].split('_')[-1].split('.')[0]
	sampled_cnn1_file=agglo_paths[0]+cnn_id1+'_samplePointsIds.pcd'
	pc=pypcd.PointCloud.from_path(sampled_cnn1_file)
	sampled_ids_cnn1=pc.pc_data['id']
	sampled_cnn1=np.arange(sampled_ids_cnn1.shape[0])
	print('Scene 1 Min Max Sampled')
	print(np.min(sampled_cnn1),np.max(sampled_cnn1))
	del sampled_ids_cnn1
	good_cnn1_file=agglo_paths[0]+cnn_id1+'_goodPointsIds.pcd'
	pc=pypcd.PointCloud.from_path(good_cnn1_file)
	good_cnn_ids1=pc.pc_data['id']
	#good points and sampled points ids are in same range
	good_cnn_ids1=sampled_cnn1[good_cnn_ids1]
	print('Scene 2 Min Max Good')
	print(np.min(good_cnn_ids1),np.max(good_cnn_ids1))
	#read 'actual' data
	data_file=agglo_paths[0]+cnn_id1+'_goodPointsX.pcd'
	#pc=pypcd.PointCloud.from_path(data_file)
	big_cnnData1,_,_=load_pcd_data_binary(data_file)
	data_file=agglo_paths[0]+cnn_id1+'_goodPoints.pcd'
	_,real_c_data,_=load_pcd_data_binary(data_file)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	real_c_data=np.concatenate((red,green,blue),axis=1)
	perPointDetectionsCNN=np.sum(real_c_data,axis=1)
	boundsCNN1=np.cumsum(perPointDetectionsCNN)
	actual_boundsCNN1=np.zeros((big_cnnData1.shape[0],1),dtype=np.int32)
	start_i=0
	for i in range(boundsCNN1.shape[0]):
		end_i=boundsCNN1[i]
		actual_boundsCNN1[start_i:end_i,0]=i
		start_i=end_i
	del boundsCNN1

	#cnn results scene 2 -> kitchen5
	cnn_file_2='All_affordances_'+scenes[1]+'_*_'+str(agglo_descriptor_ids[0])+'_*.pcd'
	path_to_scene=agglo_paths[1]+cnn_file_2
	someFile=glob.glob(path_to_scene)
	print(someFile[0])
	cnn_id2=someFile[0].split('_')[-1].split('.')[0]
	sampled_cnn2_file=agglo_paths[1]+cnn_id2+'_samplePointsIds.pcd'
	pc=pypcd.PointCloud.from_path(sampled_cnn2_file)
	sampled_ids_cnn2=pc.pc_data['id']
	sampled_cnn2=np.arange(sampled_ids_cnn2.shape[0])
	print('Scene 2 Min Max Sampled')
	print(np.min(sampled_cnn2),np.max(sampled_cnn2))
	del sampled_ids_cnn2
	good_cnn2_file=agglo_paths[1]+cnn_id2+'_goodPointsIds.pcd'
	pc=pypcd.PointCloud.from_path(good_cnn2_file)
	good_cnn_ids2=pc.pc_data['id']
	#good points and sampled points ids are in same range
	good_cnn_ids2=sampled_cnn2[good_cnn_ids2]
	print('Scene 2 Min Max Good')
	print(np.min(good_cnn_ids2),np.max(good_cnn_ids2))
	#read 'actual' data
	data_file=agglo_paths[1]+cnn_id2+'_goodPointsX.pcd'
	big_cnnData2,_,_=load_pcd_data_binary(data_file)
	data_file=agglo_paths[1]+cnn_id2+'_goodPoints.pcd'
	_,real_c_data,_=load_pcd_data_binary(data_file)
	red=np.array((real_c_data>>16)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	green=np.array((real_c_data>>8)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	blue=np.array((real_c_data)& 0x0000ff,dtype=np.uint8).reshape(-1,1)
	real_c_data=np.concatenate((red,green,blue),axis=1)
	perPointDetectionsCNN=np.sum(real_c_data,axis=1)
	boundsCNN2=np.cumsum(perPointDetectionsCNN)
	actual_boundsCNN2=np.zeros((big_cnnData2.shape[0],1),dtype=np.int32)
	start_i=0
	for i in range(boundsCNN2.shape[0]):
		end_i=boundsCNN2[i]
		actual_boundsCNN2[start_i:end_i,0]=i
		start_i=end_i
	del boundsCNN2

	if not os.path.exists(data_name) or not os.path.exists(data_name_2) or not os.path.exists(data_name_3) or not os.path.exists(data_name_4):
		for i in range(ids_target.size):
			current_agglo_interaction=i
			#get agglo data for this affordance scene1
			if agglo_descriptor_ids[0]==913 or agglo_descriptor_ids[0]==13:
				current_agglo_interaction=ids_target[i]
				#print('HELLO?')
				if labels[ids_target[i],0]!=labels[current_agglo_interaction,0] or labels[ids_target[i],-1]!=labels[current_agglo_interaction,-1]:
					print(labels[ids_target[i],0],labels[current_agglo_interaction,0])
					print(labels[ids_target[i],-1],labels[current_agglo_interaction,-1])
					print('%d %d error?'%(current_agglo_interaction,ids_target[i]))
					sys.exit()
			ids=np.nonzero(big_cnnData1[:,0]==current_agglo_interaction)[0]
			cnn1_ids=np.unique(actual_boundsCNN1[ids,0])
			good_cnn1_ids=good_cnn_ids1[cnn1_ids]

			#get agglo data for this affordance scene2
			ids=np.nonzero(big_cnnData2[:,0]==current_agglo_interaction)[0]
			cnn2_ids=np.unique(actual_boundsCNN2[ids,0])
			good_cnn2_ids=good_cnn_ids2[cnn2_ids]

			#get target interaction
			interaction=ids_target[i]
			affordance=labels[interaction,1]
			obj=labels[interaction,2]
			scene1_ok=True
			scene2_ok=True
			#it scene 1
			it_file_1=affordance+'_'+obj+'_'+scenes[0]+'_*_13_*.pcd'
			sorted_ids_it1,sampled_it1=recoverTopIds(main_file=it_file_1)
			if sorted_ids_it1[0]<0:
				scene1_ok=False
			#print(sorted_ids_it1.size)
			#it scene 2
			it_file_2=affordance+'_'+obj+'_'+scenes[1]+'_*_13_*.pcd'
			sorted_ids_it2,sampled_it2=recoverTopIds(main_file=it_file_2)
			if sorted_ids_it2[0]<0:
				scene2_ok=False
			#print(sorted_ids_it2.size)
			top_n=np.array([1,5,10,15,30,50,100])
			#bar2=Bar('\tTop-k',max=top_n.size)
			for j in range(top_n.size):
				if scene1_ok:
					top_indices=top_n[j]*sorted_ids_it1.size//100
					good_it1_ids=np.unique(sorted_ids_it1[:top_indices])
				# if good_it1_ids.size==0:
				# 	print(sorted_ids_it1.size)
				if scene2_ok:
					top_indices=top_n[j]*sorted_ids_it2.size//100
					good_it2_ids=np.unique(sorted_ids_it2[:top_indices])
				# if j==1:
				# 	print(good_it1_ids.size,good_it2_ids.size)
				if scene1_ok:
					pN1=np.setdiff1d(sampled_cnn1,good_cnn1_ids)
					aN1=np.setdiff1d(sampled_it1,good_it1_ids)
					TP1=np.intersect1d(good_cnn1_ids,good_it1_ids)
					TN1=np.intersect1d(pN1,aN1)
					FN1=np.setdiff1d(pN1,aN1)
					FP1=np.setdiff1d(good_cnn1_ids,good_it1_ids)
					if good_it1_ids.size==0:
						TPR1=0
					else:
						TPR1=TP1.size/float(good_it1_ids.size)
					
					FPR1=FP1.size/float(aN1.size)
					
					
					TNR1=TN1.size/float(aN1.size)
					FNR1=1-TPR1

					if TP1.size+FP1.size>0:
						precision1=TP1.size/float(TP1.size+FP1.size)
					else:
						precision1=0
					if TP1.size+FN1.size>0:
						recall1=TPR1
					else:
						recall1=0
					accuracy1=(TP1.size+TN1.size)/float(good_it1_ids.size+aN1.size)
				if scene2_ok:

					pN2=np.setdiff1d(sampled_cnn2,good_cnn2_ids)
					aN2=np.setdiff1d(sampled_it2,good_it2_ids)
					TN2=np.intersect1d(pN2,aN2)
					FN2=np.setdiff1d(pN2,aN2)
					TP2=np.intersect1d(good_cnn2_ids,good_it2_ids)
					if good_it2_ids.size==0:
						TPR2=0
					else:
						TPR2=TP2.size/float(good_it2_ids.size)

					TNR2=TN1.size/float(aN2.size)
					FNR2=1-TPR2
					
					FP2=np.setdiff1d(good_cnn2_ids,good_it2_ids)
					FPR2=FP2.size/float(aN2.size)
					if TP2.size+FP2.size>0:
						precision2=TP2.size/float(TP2.size+FP2.size)
					else:
						precision2=0
					if TP2.size+FN2.size>0:
						recall2=TPR2
					else:
						recall2=0
					accuracy2=(TP2.size+TN2.size)/float(good_it2_ids.size+aN2.size)
				if scene1_ok and scene2_ok:
					avg_prec=np.mean([precision1,precision2])
					avg_recall=np.mean([recall2,recall1])
					avg_accuracy=np.mean([accuracy1,accuracy2])
					avg_TPR=avg_recall
					avg_FPR=np.mean([FPR1,FPR2])
				elif scene1_ok and not scene2_ok:
					avg_prec=precision1
					avg_recall=recall1
					avg_accuracy=accuracy1
					avg_TPR=avg_recall
					avg_FPR=FPR1
				elif scene2_ok and not scene1_ok:
					avg_prec=precision2
					avg_recall=recall2
					avg_accuracy=accuracy2
					avg_TPR=avg_recall
					avg_FPR=FPR2
				elif not scene1_ok and not scene2_ok:
					avg_prec=avg_recall=avg_accuracy=avg_FPR=avg_TPR=0

				#print('%s %s mean-prec:%.2f mean-recall:%.2f'%(affordance,obj, avg_prec,avg_recall))
				data_points[j,i,0]=avg_prec
				data_points[j,i,1]=avg_recall
				data_points_extra[j,i,0]=avg_accuracy
				data_points_extra[j,i,1]=avg_TPR
				data_points_extra[j,i,2]=avg_FPR

				if last_read==affordance:
					per_affordance_data_points[j,c,1]+=avg_recall
					per_affordance_data_points[j,c,0]+=avg_prec
					per_affordance_data_points_extra[j,c,0]+=avg_accuracy
					per_affordance_data_points_extra[j,c,1]+=avg_TPR
					per_affordance_data_points_extra[j,c,2]+=avg_FPR
					per_affordance_interactions[j,c,0]+=1
				else:
					c+=1
					per_affordance_data_points[j,c,1]+=avg_recall
					per_affordance_data_points[j,c,0]+=avg_prec
					per_affordance_data_points_extra[j,c,0]+=avg_accuracy
					per_affordance_data_points_extra[j,c,1]+=avg_TPR
					per_affordance_data_points_extra[j,c,2]+=avg_FPR
					per_affordance_interactions[j,c,0]+=1
					last_read=affordance

				#bar2.next()
			#bar2.finish()
			bar.next()
		save_as_h5(data_name,data_points,'float32')
		#np.save(data_name,data_points)
		per_affordance_data_points=per_affordance_data_points/per_affordance_interactions
		per_affordance_data_points_extra=per_affordance_data_points_extra/per_affordance_interactions
		save_as_h5(data_name_2,per_affordance_data_points,'float32')
		#np.save(data_name_2,per_affordance_data_points)
		save_as_h5(data_name_3,data_points_extra,'float32')
		#np.save(data_name_3,data_points_extra)
		save_as_h5(data_name_4,per_affordance_data_points_extra,'float32')
		#np.save(data_name_4,per_affordance_data_points_extra)
		bar.finish()
	else:
		#data_points=np.load(data_name)
		data_points=load_from_h5(data_name)
		per_affordance_data_points=load_from_h5(data_name_2)
		data_points_extra=load_from_h5(data_name_3)
		per_affordance_data_points_extra=load_from_h5(data_name_4)
	#find max
	all_precisions=data_points[:,:,0]
	all_recalls=data_points[:,:,1]
	all_accuracies=data_points_extra[:,:,0]
	all_mean_data=np.zeros((3,all_accuracies.shape[0]))
	all_mean_data[0,...]=np.mean(all_precisions,axis=1)
	all_mean_data[1,...]=np.mean(all_recalls,axis=1)
	all_mean_data[2,...]=np.mean(all_accuracies,axis=1)
	#print(all_mean_data)
	name='Metrics_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.csv'
	with open(name, "w") as text_file:
			text_file.write("Top1,Top5,Top10,Top15,Top30,Top50,All\n")
			for i in range(all_mean_data.shape[0]):
				text_file.write("%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f\n" % (all_mean_data[i,0],all_mean_data[i,1],all_mean_data[i,2],all_mean_data[i,3],all_mean_data[i,4],all_mean_data[i,5],all_mean_data[i,6]))
				print("%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f" % (all_mean_data[i,0],all_mean_data[i,1],all_mean_data[i,2],all_mean_data[i,3],all_mean_data[i,4],all_mean_data[i,5],all_mean_data[i,6]))
	print(data_points.shape)
	#print(np.mean(all_precisions,axis=1))
	#print(np.mean(all_recalls,axis=1))
	#print(np.mean(all_accuracies,axis=1))

	#sys.exit()
	#print(np.mean(data_points,axis=0))
	#print(np.mean(data_points_extra,axis=0))
	# fig = plt.figure(figsize=(5, 5))
	# ax = fig.add_subplot(221)
	# ax2 = fig.add_subplot(222)
	# ax3 = fig.add_subplot(223)
	# ax4 = fig.add_subplot(224)
	# ax.scatter(data_points[:,1],data_points[:,0],c='g',label='perInteractionPR')
	# ax2.scatter(per_affordance_data_points[:,1],per_affordance_data_points[:,0],c='b',label='perAffordacePR')
	# ax3.plot(np.array([0,1]),np.array([0,1]),ls='--')
	# ax4.plot(np.array([0,1]),np.array([0,1]),ls='--')
	# ax3.scatter(data_points_extra[:,2],data_points_extra[:,1],c='r',label='perInteractionROC')
	# ax4.scatter(per_affordance_data_points_extra[:,2],per_affordance_data_points_extra[:,1],c='b',label='perAffordaceROC')
	# for i in range(ids_target.size):
	# 	if data_points[i,0]<0.95:
	# 		#print('%s %f'%(plot_labels[i],data_points[i,0]))
	# 		ax.text(data_points[i,1], data_points[i,0]+.02, plot_labels[i], horizontalalignment='right', size='medium', color='black', weight='semibold')
	# for i in range(per_affordance_data_points.shape[0]):
	# 	ax2.text(per_affordance_data_points[i,1], per_affordance_data_points[i,0]+.02, plot_labels2[i], horizontalalignment='center', size='medium', color='black', weight='semibold')
	# print('=========================')
	# for i in range(ids_target.size):
	# 	if data_points_extra[i,2]>0.3 or data_points_extra[i,1]<0.7:
	# 		#print('%s %f'%(plot_labels[i],data_points_extra[i,1]))
	# 		ax3.text(data_points_extra[i,2]+0.02, data_points_extra[i,1], plot_labels[i], horizontalalignment='left', size='medium', color='black', weight='semibold')

	# for i in range(per_affordance_data_points_extra.shape[0]):
	# 	ax4.text(per_affordance_data_points_extra[i,2], per_affordance_data_points_extra[i,1]+.02, plot_labels2[i], horizontalalignment='center', size='medium', color='black', weight='semibold')
	# #ax.set_xticks(ind)
	# #ax.set_xticklabels(plot_labels,rotation=45,ha="right")
	# #ax.plot(ind,counts[sorted_ids,0],c='b',label='2048')
	# #ax.plot(ind,counts[sorted_ids,1],c='g',label='1024')
	# #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	# ax.set_xlim(0,1)
	# #ax3.set_xlim(0,1)
	# ax2.set_xlim(0,1)
	# ax3.set_xlim(0,1)
	# ax4.set_xlim(0,1)

	# ax.set_xlabel('Recall')
	# ax.set_ylabel('Precision')
	# ax2.set_xlabel('Recall')
	# ax2.set_ylabel('Precision')
	# ax3.set_xlabel('FPR')
	# ax3.set_ylabel('TPR')
	# ax4.set_xlabel('FPR')
	# ax4.set_ylabel('TPR')

	# ax.set_ylim(0,1)
	# #ax3.set_ylim(0,1)
	# ax2.set_ylim(0,1)
	# ax3.set_ylim(0,1)
	# ax4.set_ylim(0,1)
	# ax.grid()
	# ax2.grid()
	# ax3.grid()
	# ax4.grid()
	# ax.set_axisbelow(True)
	# ax2.set_axisbelow(True)
	# ax3.set_axisbelow(True)
	# ax4.set_axisbelow(True)
	# plt.tight_layout()
	# data_name='stats_'+str(agglo_descriptor_ids[0])+'_'+str(agglo_descriptor_ids[1])+'.eps'
	# plt.savefig(data_name,bbox_inches='tight',format='eps', dpi=300)
	# plt.show()

if __name__ == '__main__':
	cnnStats()