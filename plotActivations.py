import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import glob
import array 
import sys
from compareIT import load_pcd_data_binary
from sklearn.neighbors import KDTree
import socket
import h5py
import os
from progress.bar import Bar
from recoverSetsAgglo import (save_as_h5,load_from_h5)
from compareIT import (load_h5,load_pcd_data_binary)
import seaborn as sns
import pypcd

host=socket.gethostname()

if host=="it057384":
	data_paths=['/home/er13827/deepNetwork/halDATA/Eduardo/PointNet/','/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad/','/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/']
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

def plotCells(data_points,data_scores,name):
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111,projection="3d")
	plt.ion()
	ax.scatter(data_points[:,0],data_points[:,1],data_points[:,2],s=20,c=data_scores,cmap='jet',alpha=1)
	plt.axis('off')
	plt.grid(False)
	plt.savefig(name, bbox_inches='tight',format='eps', dpi=300)
	plt.pause(2)
	plt.draw()
	plt.close()



def plotTensorActivations():
	i=1
	max_rad=0.806884
	responses=np.load('all_projected_activations_clipped.npy')
	print('Building grid')
	x_=np.arange(-max_rad,max_rad,0.01)
	y_=np.arange(-max_rad,max_rad,0.01)
	z_=np.arange(-max_rad,max_rad,0.01)
	x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
	x=x.reshape(1,-1)
	y=y.reshape(1,-1)
	z=z.reshape(1,-1)
	grid=np.concatenate((x,y,z),axis=0).T
	names_labels=np.expand_dims(np.genfromtxt('common_namesreal-kitchen1.csv',dtype='str'),axis=1)
	print('Done')
	for i in range(responses.shape[1]):
		activations=np.nonzero(responses[:,i])[0]
		active_grid=grid[activations,:]
		active_grid_scores=responses[activations,i]
		name=names_labels[i,0]+'.eps'
		plotCells(active_grid,active_grid_scores,name)

def plotDescriptorStats():
	target_files='point_count*.dat'
	someFiles=sorted(glob.glob(target_files))
	print(someFiles)
	target_files='New*_Approx_descriptor_8.pcd'
	someFiles2=sorted(glob.glob(target_files))
	print(someFiles2)
	target_files='New*_Approx_descriptor_8_points.pcd'
	someFiles3=sorted(glob.glob(target_files))
	print(someFiles3)
	all_counts=np.zeros((len(someFiles),84))
	d_sizes=np.zeros((len(someFiles2),2))
	for i in range(len(someFiles)):
		file=someFiles[i]
		size_ = array.array('i', [0, 0])
		with open(file, "rb") as binary_file:
			binary_file.readinto(size_)
			a=np.asarray(size_)
			list_=[]
			[list_.append(0) for x in range(a[0]*a[1])]
			#print(a[0],a[1])
			real_counts_arr=array.array('f',list_)
			binary_file.readinto(real_counts_arr)
			real_data=np.expand_dims(np.asarray(real_counts_arr),axis=1).reshape(a[0],a[1])
			real_data=real_data.T
			all_counts[i,:]=real_data[:,0].astype(int)
		approx_d,_,_=load_pcd_data_binary(someFiles2[i])
		all_points,_,_=load_pcd_data_binary(someFiles3[i])
		d_sizes[i,0]=approx_d.shape[0]
		d_sizes[i,1]=all_points.shape[0]
	fig = plt.figure(figsize=(15, 10))
	ax = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	#ax.plot(all_counts[0,:],c='r')
	wd=0.3
	X = np.arange(84)
	ax.bar(X -wd, all_counts[0,:],  width = 0.25, align='center')
	ax.bar(X , all_counts[1,:], width = 0.25, align='center')
	ax.bar(X + wd, all_counts[2,:], width = 0.25, align='center')
	ax.autoscale(tight=True)
	plt.tight_layout()
	ax.set_xticks(X)

	ax2.plot(np.mean(all_counts[:-1,:],axis=1))
	#ax2.plot(d_sizes[:-1,0])
	#ax2.plot(d_sizes[:-1,1])
	x=np.array([1,0.7,0.5])
	ax2.set_xticks(x)
	for i in range(d_sizes.shape[0]-1):
		increase=(d_sizes[i+1,1]-d_sizes[i,1])/float(d_sizes[i+1,1])
		print(increase)
	plt.show()

def plotSceneActivations():
	cell_size=0.01
	#995 1cm cells without clipping
	#994 1cm cells with clipping
	#993 0.7cm cells with clipping
	#992 0.5cm cells with clipping
	descriptor=994
	name=data_paths[0]+'MultilabelDataSet_splitTest2.h5'
	input_clouds,input_labels=load_h5(name)
	input_ids_name=data_paths[2]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/inputIds.npy'
	inputIds=np.load(input_ids_name)
	new_name=data_paths[2]+'AFF_All_BATCH_16_EXAMPLES_2_DATA_miniDataset3/dump/recoveredActivations.npy'
	all_activations=np.load(new_name)

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
	# object_sizes=np.zeros((n_affordances,1),dtype=np.float32)
	# cell_ids=np.load('pop_cell_ids_clipped.npy')
	# print(cell_ids.shape)
	# for i in range(cell_ids.shape[1]):
	# 	pop_cells=np.nonzero(cell_ids[:,i])[0]
	# 	print('Affordance %d Cells %d'%(i,pop_cells.size))
	#for every activation get the closest grid point 
	plot=True
	if plot:
		fig = plt.figure(figsize=(7, 7))
		#fig2= plt.figure(figsize=(10, 5))
		#plt.ion()
		#ax = fig.add_subplot(131,projection="3d")
		#ax = fig.add_subplot(121,projection="3d")
		ax = fig.add_subplot(111,projection="3d")
		#ax2 = fig2.add_subplot(111)
		#ax3 = fig.add_subplot(122,projection="3d")
		#just to skip file checking below
		name='all_projected_activations_clipped.npy'
	else:
		name='all_projected_activations_clipped.npy'
		# ax.view_init(azim=135)
		# ax2.view_init(azim=135)
		# ax3.view_init(azim=135)

	print(all_activations.shape)
	print(input_clouds.shape)

	kdt=KDTree(grid,metric='euclidean')
	#sys.exit()
	if not os.path.exists('All_saliencies_clipped.h5'):
		responses=np.zeros((grid.size,n_affordances),dtype=np.int32)
		bar = Bar('Projecting activations into grid',max=all_activations.shape[0])
		for i in range(all_activations.shape[0]):
			rotations=orientations[i,...]
			thisActivations=all_activations[i,...]
			real_ids=np.unique(thisActivations)
			real_ids=inputIds[i,real_ids]
			activations_3d=input_clouds[i,real_ids,:]
			affordances_actually_here=np.nonzero(input_labels[i,:n_affordances])[0]
			for j in range(affordances_actually_here.size):
				affordance_id=affordances_actually_here[j]
				#rotate back the actications
				anAngle=-rotations[affordance_id]*(2*np.pi)/8
				aCloud=rotate_point_cloud_by_angle(activations_3d,anAngle)
				#for these activations search the closes grid point
				_,ind=kdt.query(aCloud,k=1)
				#ids from 0 to grid size
				ind=np.unique(ind[:,0])
				#print(ind.size,affordance_id)
				responses[ind,affordance_id]+=1
			bar.next()
		bar.finish()
		save_as_h5('All_saliencies_clipped.h5',responses,'int32')
	else:
		print('Reading saliencies')
		responses=load_from_h5('All_saliencies_clipped.h5')
		print('done')

	test_points=np.load('MultilabelDataSet_splitTest2_points.npy')
	test_ids=np.load('MultilabelDataSet_splitTest2.npy')

	for i in range(all_activations.shape[0]):
			rotations=orientations[i,...]
			thisActivations=all_activations[i,...]
			#print(thisActivations.flatten().shape)
			counts,bins=np.histogram(thisActivations.flatten(),bins=np.arange(1,1024))
			sorted_counts_ids=np.argsort(counts)
			counts=counts[sorted_counts_ids[::-1]]
			bins=bins[sorted_counts_ids[::-1]]
			#print(counts[:20])
			minV=np.percentile(counts,50)
			#print(counts.min(),counts.max(),minV)
			#counts=counts/counts.max()
			minV=np.percentile(counts,50)
			#print(counts.min(),counts.max(),minV)
			in_percentile=np.nonzero(counts>=minV)[0]
			#print(minV,in_percentile.size)
			

			#print(np.max(scores[1:]),np.min(scores[1:]))
			#all_scores=np.zeros(input_clouds.shape[1])
			#all_scores[:scores.size]=scores
			real_ids=bins[in_percentile]
			real_ids=inputIds[i,real_ids]
			remaining_ids=np.setdiff1d(np.arange(4096),real_ids)
			activations_3d=input_clouds[i,real_ids,:]
			affordances_actually_here=np.nonzero(input_labels[i,:n_affordances])[0]
			ax.set_xlim(-max_rad,max_rad)
			ax.set_ylim(-max_rad,max_rad)
			ax.set_zlim(-max_rad,max_rad)
			ax.set_xticks([]) 
			ax.set_yticks([]) 
			ax.set_zticks([])
			ax.grid(False)
			#ax.xaxis.pane.set_edgecolor('white')
			#ax.yaxis.pane.set_edgecolor('white')
			#ax.zaxis.pane.set_edgecolor('white')
			ax.xaxis.pane.fill = False
			ax.yaxis.pane.fill = False
			ax.zaxis.pane.fill = False
			scores=counts[in_percentile];
			cm = plt.cm.get_cmap('hot_r')
			if affordances_actually_here.size>0:
				if 83 in set(affordances_actually_here) and test_ids[i]>6151:
				 	#ax.view_init(elev=10,azim=45)
					ax.set_title(str(i))
					#print(minV,scores.max())
					ax.scatter(input_clouds[i,remaining_ids,0],input_clouds[i,remaining_ids,1],input_clouds[i,remaining_ids,2],c=np.zeros(remaining_ids.size)+0.1,vmin=0,vmax=1,s=30,cmap='jet')
					splot=ax.scatter(activations_3d[:,0],activations_3d[:,1],activations_3d[:,2],c=scores, norm=colors.LogNorm(vmin=scores.min(), vmax=scores.max()), s=50, cmap=cm,depthshade=False)
					# plt.colorbar(splot)
					ax.scatter(0,0,0,s=20,c='r')
					#plt.show()
					plt.draw()
					plt.pause(0.1)
					ax.clear()
					#save cloud
					activations_3d_tosave=activations_3d+test_points[i,...]
					print('SAVING!!!!')
					actual_data_array=np.zeros(activations_3d_tosave.shape[0], dtype={'names':('x', 'y', 'z','w1'),
	  	                        'formats':('f4', 'f4', 'f4','f4')})
					actual_data_array['x']=activations_3d_tosave[:,0]
					actual_data_array['y']=activations_3d_tosave[:,1]
					actual_data_array['z']=activations_3d_tosave[:,2]
					actual_data_array['w1']=scores
					new_cloud = pypcd.PointCloud.from_array(actual_data_array)
					nname='activations_'+str(i)+'_MultiSplit2.pcd'
					new_cloud.save_pcd(nname,compression='ascii')

	#Once done with all activations get most of them
	bar = Bar('Plotting',max=n_affordances)
	for i in range(n_affordances):
		this_many_examples=np.count_nonzero(input_labels[:,i])
		target_responses=this_many_examples//2
		print('Examples %d -> %d'%(this_many_examples,target_responses))
		this_affordance_responses=np.nonzero(responses[:,i])[0]
		#minV=np.percentile(responses[this_affordance_responses,i],5)
		most_responses=np.nonzero(responses[:,i]>10)[0]
		some_to_plot=np.arange(this_affordance_responses.size)
		np.random.shuffle(some_to_plot)
		some_to_plot=some_to_plot[:2048]
		# ax.hist(responses[this_affordance_responses,i],bins=100)
		# plt.draw()
		# plt.pause(5)
		# ax.clear()
		if plot:
			print('Active grid %d Top activations %d'%(this_affordance_responses.size,most_responses.size))
# 			#data_scores=this_affordance_responses
 			ax.scatter(grid[this_affordance_responses[some_to_plot],0],grid[this_affordance_responses[some_to_plot],1],grid[this_affordance_responses[some_to_plot],2],s=1,c='b')
 			ax.scatter(grid[most_responses,0],grid[most_responses,1],grid[most_responses,2],s=2,c='r')
 		# 	name='Saliency_'+str(i)+'.eps'
			# print(name)
			plt.draw()
			#plt.savefig(name, bbox_inches='tight',format='eps', dpi=80)
			plt.pause(10)
#			plt.draw()
			ax.clear()
		bar.next()
	bar.finish()


	# 		if plot:
	# 			#setPlotLims(ax,ax2,ax3,max_rad)
	# 			# ax.scatter(activations_3d[:,0],activations_3d[:,1],activations_3d[:,2],s=1,c='b')
	# 			# ax.set_title(str(j)+' '+str(rotations[affordance_id]))
	# 			# ax.scatter(0,0,0,s=25,c='r')
	# 			somePoint=np.array([0,-0.8,0])
	# 			# ax.plot([0,somePoint[0]],[0,somePoint[1]],[0,somePoint[2]],linewidth=2, markersize=12,color='g')
	# 			rotatedPoint=rotate_point_cloud_by_angle(somePoint,anAngle)
	# 			ax2.scatter(aCloud[:,0],aCloud[:,1],aCloud[:,2],s=1,c='b')
	# 			ax2.scatter(0,0,0,s=25,c='r')
	# 			ax2.plot([0,rotatedPoint[0,0]],[0,rotatedPoint[0,1]],[0,rotatedPoint[0,2]],linewidth=2, markersize=12,color='g')
	# 			ax2.set_title(str(j)+' '+str(anAngle))
	# 			ax3.scatter(active_grid[:,0],active_grid[:,1],active_grid[:,2],s=1,c='b')
	# 			#ax3.plot([0,somePoint[0]],[0,somePoint[1]],[0,somePoint[2]],linewidth=2, markersize=12,color='g')
	# 			nn_cloud=grid[pop_cells[ind],:]
	# 			ax3.scatter(nn_cloud[:,0],nn_cloud[:,1],nn_cloud[:,2],s=20,c='g')
	# 			if rotations[affordance_id]!=0 and rotations[affordance_id]!=4:
	# 				plt.pause(10)
	# 			else:
	# 				plt.pause(1)
	# 			plt.draw()
	# 			#ax.clear()
	# 			ax2.clear()
	# 			ax3.clear()
	
def plotDescriptorDim(descriptors=(14,15,16,992,993,994)):

	descriptors_paths=['/home/er13827/space/testing/','/home/er13827/space/pointnet2/utils/']
	data=np.zeros((len(descriptors),2),dtype=np.float32)
	affordanceS_to_ignore=set(['Hang-mug','Place-cell-phone','Place-credit-card','Place-headphone-stand','Place-keyboard','Place-magazine','Place-tablet','Ride-biker2'])
	affordanceS_to_ignore_ids=set([13,28,35,47,49,54,81,90])
	for i in range(len(descriptors)):
		print('Descriptor id %d'%(descriptors[i]))
		descriptor_name=descriptors_paths[0]+'New'+str(descriptors[i])+'_Approx_descriptor_8.pcd'
		descriptor_members=descriptors_paths[0]+'New'+str(descriptors[i])+'_Approx_descriptor_8_extra.pcd'
		if not os.path.exists(descriptor_name):
			#try second path
			descriptor_name=descriptors_paths[1]+'New'+str(descriptors[i])+'_Approx_descriptor_8.pcd'
			descriptor_members=descriptors_paths[1]+'New'+str(descriptors[i])+'_Approx_descriptor_8_extra.pcd'
			if not os.path.exists(descriptor_name):
				print('Did not find descriptor')
				sys.exit()
		points,_,_=load_pcd_data_binary(descriptor_name)
		ids,_,_=load_pcd_data_binary(descriptor_members)
		affordance_ids=ids[:,0]
		n_affordances=np.unique(affordance_ids)
		affordance_counts=np.zeros((n_affordances.size,1))
		# for j in range(n_affordances.size):
		# 	if descriptors[i]<900 and j in affordanceS_to_ignore_ids:
		# 		#print('skip')
		# 		continue
		# 	this_many=np.nonzero(affordance_ids==j)[0]
		# 	affordance_counts[j,0]=this_many.size
		data[i,0]=points.shape[0]
		data[i,1]=512*84*8

	fig = plt.figure(figsize=(7, 3))
	#plt.ion()
	sns.set()
	sns.set_style("white")
	colors = ["#e74c3c","#9b59b6", "#3498db", "#34495e", "#2ecc71"]
	ax = fig.add_subplot(111)
	#ax2=fig.add_subplot(122,sharey=ax)
	bar_width = 0.35
	index = np.arange(len(descriptors)/2)
	index = np.arange(4)
	opacity = 0.4
	#ax.bar(index,data[:3,1],bar_width,label='keypoints',color=colors[4])
	ax.bar(index[0],data[0,1],bar_width,label='keypoints',color=colors[2])
	ax.bar(index[1:]-bar_width, data[:3,0], bar_width,label='iT Agglomeration',color=colors[3])
	ax.set_xlabel('Cell size [cm]')
	ax.set_ylabel('# of points')
	ax.set_title('iT Agglomeration')
	ax.set_xticks(index)
	#ax.set_yscale('log')
	ax.set_xticklabels(('','0.5','.75', '1'))
	ax.yaxis.grid(True)
	ax.set_ylim(bottom=1e2 )
	#ax.grid()
	#ax.legend()
	#ax.bar(index,data[3:,1],bar_width,label='keypoints',color=colors[4])
	ax.bar(index[1:], data[3:,0], bar_width,label='Saliency',color=colors[4])
	# ax2.set_xlabel('Cell size [cm]')
	# #ax2.set_ylabel('# of points')
	# ax2.set_title('Saliency')
	# ax2.set_xticks(index + bar_width / 2)
	# #ax2.set_yscale('log')
	# ax2.set_xticklabels(('0.5','.75', '1'))
	# ax2.yaxis.grid(True)
	#plt.setp(ax2.get_yticklabels(), visible=False)
	#ax2.legend()
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center',ncol=3)
	ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
	# ax2.ticklabel_format(style='sci', axis='y')
	#plt.legend(loc=3,ncol=2, borderaxespad=0.)

	fig.tight_layout()
	
	plt.savefig('cell_size.eps', bbox_inches='tight',format='eps', dpi=80)
	plt.show()

def plotTimes():
	#load older times
	#they come as cellSizes x phases
	#where phases are prepro, nn-search, scoring
	agglo_times=np.load('older_times.npy')
	print(np.sum(agglo_times[1:4,:],axis=1))
	fig = plt.figure(figsize=(7, 5))
	#plt.ion()
	sns.set()
	sns.set_style("white")
	#sns.set_context("poster")
	colors = ["#e74c3c","#9b59b6", "#3498db", "#34495e", "#2ecc71"]
	ax = fig.add_subplot(121)
	ax2=fig.add_subplot(122,sharey=ax)
	#only take 3 middle sizes -> 0.5, 0.74 and 1 cm
	sizes = ['0.5', '0.7', '1']
	ind = [x for x, _ in enumerate(sizes)]
	ax.bar(ind, agglo_times[1:4,0]+agglo_times[1:4,1],color=colors[1],label='Preprocess')
	ax.bar(ind, agglo_times[1:4,2],bottom=agglo_times[1:4,0]+agglo_times[1:4,1],color=colors[3],label='NN-search')
	ax.bar(ind, agglo_times[1:4,4],bottom=agglo_times[1:4,0]+agglo_times[1:4,1]+agglo_times[1:4,2],color=colors[4],label='Scoring')
	
	ax.legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=1)
	new_ticks_pos=np.array(ind)+0.5;
	ax.set_xticks(new_ticks_pos)
	ax.set_xticklabels(sizes)
	#ax.yaxis.set_label_position("right")
	#ax.yaxis.tick_right()
	ax.set_ylabel("time [ms]")
	ax.set_xlabel("cell size [cm]")
	#plt.yticks(rotation=90)
	plt.tight_layout(pad=1)
	#fig.tight_layout()
	plt.show()

def rect_prism(ax,center,side_lenght,some_alpha,some_color):
	x_range=np.array([center[0]-side_lenght,center[0]+side_lenght])
	y_range=np.array([center[1]-side_lenght,center[1]+side_lenght])
	z_range=np.array([center[2]-side_lenght,center[2]+side_lenght])
	# TODO: refactor this to use an iterotor
	xx, yy = np.meshgrid(x_range, y_range)
	ax.plot_wireframe(xx, yy, z_range[0], color=some_color,linewidth=1)
	ax.plot_surface(xx, yy, z_range[0], color=some_color, alpha=some_alpha)
	ax.plot_wireframe(xx, yy, z_range[1], color=some_color,linewidth=1)
	ax.plot_surface(xx, yy, z_range[1], color=some_color, alpha=some_alpha)


	yy, zz = np.meshgrid(y_range, z_range)
	ax.plot_wireframe(x_range[0], yy, zz, color=some_color,linewidth=1)
	ax.plot_surface(x_range[0], yy, zz, color=some_color, alpha=some_alpha)
	ax.plot_wireframe(x_range[1], yy, zz, color=some_color,linewidth=1)
	ax.plot_surface(x_range[1], yy, zz, color=some_color, alpha=some_alpha)

	xx, zz = np.meshgrid(x_range, z_range)
	ax.plot_wireframe(xx, y_range[0], zz, color=some_color,linewidth=1)
	ax.plot_surface(xx, y_range[0], zz, color=some_color, alpha=some_alpha)
	ax.plot_wireframe(xx, y_range[1], zz, color=some_color,linewidth=1)
	ax.plot_surface(xx, y_range[1], zz, color=some_color, alpha=some_alpha)


def plotCubes():
	#read wine-bottle, sit-human and stool
	path='/home/er13827/deepNetwork/skynetDATA/Eduardo/InividualAgglo/'
	descriptor_names=['New923_Approx_descriptor_8_points.pcd','New813_Approx_descriptor_8_points.pcd','New893_Approx_descriptor_8_points.pcd']
	members_names=['New923_Approx_descriptor_8_extra.pcd','New813_Approx_descriptor_8_extra.pcd','New893_Approx_descriptor_8_extra.pcd']
	#read descriptors:
	samples=512
	max_rad=0.806884
	descriptors=np.zeros((3,samples,3),dtype=np.float32)
	for i in range(len(descriptor_names)):
		points,_,_=load_pcd_data_binary(path+descriptor_names[i])
		ids,_,_=load_pcd_data_binary(path+members_names[i])
		targets=np.nonzero(ids[:,1]==0)[0]
		print(targets.size)
		descriptors[i,:targets.size,:]=points[targets,:]
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111,projection='3d')
	c=0.05
	ax.set_xlim(-max_rad,max_rad)
	ax.set_ylim(-max_rad,max_rad)
	ax.set_zlim(-max_rad,max_rad)
	sample=np.arange(samples)
	np.random.shuffle(sample)
	sample=sample[:300]
	# ax.scatter(descriptors[0,sample,0],descriptors[0,sample,1],descriptors[0,sample,2])
	# ax.scatter(descriptors[1,sample,0],descriptors[1,sample,1],descriptors[1,sample,2])
	# ax.scatter(descriptors[2,sample,0],descriptors[2,sample,1],descriptors[2,sample,2])
	ax.view_init(azim=-135)
	ax.grid(False)
	# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.axis('off')
	for i in range(len(descriptor_names)):
		for j in range(sample.size):
			rect_prism(ax,descriptors[i,sample[j],:],c,c,"b")
	plt.savefig('back_projection_grid1.pdf', bbox_inches='tight',format='pdf',transparent=True,dpi=50)
	plt.show()


if __name__ == '__main__':
	#plotSceneActivations()
	#plotDescriptorDim()
	#plotTimes()
	plotCubes()