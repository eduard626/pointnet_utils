import numpy as np
import pypcd
import os
import matplotlib.pyplot as plt
import socket

host=socket.gethostname()
if host=='it057384':
	MODEL_DIR2048='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/'
	MODEL_DIR1024='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered_1024/'
	NEW_DATA_DIR='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/New_data/'
	INPUT_DATA_DIR='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet/'
else:
	MODEL_DIR='/media/hal/DATA/Eduardo/PointNet2/Results_rad_centered/'
	NEW_DATA_DIR='/media/hal/DATA/Eduardo/PointNet2/New_data/'
	INPUT_DATA_DIR='/media/hal/DATA/Eduardo/PointNet/'


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

def autolabel(rects,values,ax):
    """
    Attach a text label above each bar displaying its height
    """
    i=0
    for rect in rects:
        height = rect.get_height()
        value=values[i]
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(value),
                ha='left', va='bottom',rotation=45)
        i+=1

def plotBars(accuracies,counts,labels,ids_target):
	#first subplot with a third of the data
	thisMany=ids_target.size//3

	fig = plt.figure(figsize=(15, 8))
	ax = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	ind = np.arange(thisMany)
	width = 0.3
	#SUBPLOT 1
	#p1 = ax.barh(ind, counts[:,0], width, color='b', bottom=0)
	p1 = ax.bar(ind, accuracies[:thisMany,0], width, color='b')
	#p2 = ax.barh(ind + width, counts[:,1], width,color='g', bottom=0)
	p2 = ax.bar(ind + width, accuracies[:thisMany,1], width,color='g')
	ax.set_xticks(ind + width / 2)
	plot_labels=[x[0]+'-'+x[-1] for x in labels[ids_target,:]]
	ax.set_xticklabels(plot_labels[:thisMany],rotation=45,ha="right")
	ax.legend((p1[0], p2[0]), ('2048', '1024'))
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	autolabel(p1,counts[:thisMany,0],ax)
	autolabel(p2,counts[:thisMany,1],ax)
	#SUBPLOT 2
	p1 = ax2.bar(ind, accuracies[thisMany:2*thisMany,0], width, color='b')
	#p2 = ax.barh(ind + width, counts[:,1], width,color='g', bottom=0)
	p2 = ax2.bar(ind + width, accuracies[thisMany:2*thisMany,1], width,color='g')
	ax2.set_xticks(ind + width / 2)
	ax2.set_xticklabels(plot_labels[thisMany:2*thisMany],rotation=45,ha="right")
	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	autolabel(p1,counts[thisMany:2*thisMany,0],ax2)
	autolabel(p2,counts[thisMany:2*thisMany,1],ax2)
	#ax2.legend((p1[0], p2[0]), ('1024', '2048'))
	#SUBPLOT 3
	p1 = ax3.bar(ind, accuracies[-thisMany:,0], width, color='b')
	#p2 = ax.barh(ind + width, counts[:,1], width,color='g', bottom=0)
	p2 = ax3.bar(ind + width, accuracies[-thisMany:,1], width,color='g')
	ax3.set_xticks(ind + width / 2)
	ax3.set_xticklabels(plot_labels[-thisMany:],rotation=45,ha="right")
	ax3.spines['right'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	autolabel(p1,counts[-thisMany:,0],ax3)
	autolabel(p2,counts[-thisMany:,1],ax3)
	#ax3.legend((p1[0], p2[0]), ('1024', '2048'))
	plt.tight_layout()
	#plt.pause(5)
	plt.show()

def plotLines(accuracies,counts,labels,ids_target):
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(111)
	ind=np.arange(ids_target.size)
	sorted_ids=np.argsort(accuracies[:,0])
	sorted_ids2=np.argsort(accuracies[:,1])
	#print(sorted_ids)
	plot_labels=[x[0]+'-'+x[-1] for x in labels[ids_target[sorted_ids],:]]
	ax.scatter(ind,accuracies[sorted_ids,0],c='b',s=7,label='2048')
	ax.scatter(ind,accuracies[sorted_ids2,1],c='g',s=7,label='1024')
	ax.set_xticks(ind)
	ax.set_xticklabels(plot_labels,rotation=45,ha="right")
	#ax.plot(ind,counts[sorted_ids,0],c='b',label='2048')
	#ax.plot(ind,counts[sorted_ids,1],c='g',label='1024')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	#plt.gca().xaxis.grid(True)
	plt.gca().yaxis.grid(True)
	plt.tight_layout()
	plt.show()


def compareActivations():
	data_path='../data/activations/'

	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	path=os.path.abspath('/home/er13827/space/testing/tmp.csv')
	pos=path.rfind('/')
	descriptor_id=13
	file_descriptor=path[:pos]+'/tmp'+str(descriptor_id)+'.csv'
	labels=np.genfromtxt(file_descriptor,dtype='str',skip_header=1,delimiter=',')
	print('Affordances in descriptor %d'%labels.shape[0])

	samples=512
	points=4096
	max_rad=0.806884
	traininig_examples=512
	ids_target=np.nonzero(new_c>=samples)[0]
	counts=np.zeros((ids_target.size,2),dtype=np.int32)
	
	#bigConfusion=np.empty((ids_target,size*2,))
	accuracies=np.zeros((ids_target.size,2),dtype=np.float32)
	for i in range(ids_target.size):
		interaction=ids_target[i]
		activations_2048=data_path+'ibs_sample_cnn_sample2048_'+labels[interaction,1]+'_'+labels[interaction,2]+'.pcd'
		data2,_,_=load_pcd_data_binary(activations_2048)
		activations_1024=data_path+'ibs_sample_cnn_sample_'+labels[interaction,1]+'_'+labels[interaction,2]+'.pcd'
		data1,_,_=load_pcd_data_binary(activations_1024)
		counts[i,0]=data2.shape[0]
		counts[i,1]=data1.shape[0]
		confusion_file=MODEL_DIR2048+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_confusion_matrix.csv'
		cm=np.genfromtxt(confusion_file,delimiter=',',dtype='int')
		accuracies[i,0]=(cm[0,0]+cm[1,1])/float(1024)
		confusion_file=MODEL_DIR1024+'AFF_'+str(interaction)+'_BATCH_16_EXAMPLES_'+str(traininig_examples)+'_DATA_binary_OC/dump/A_'+str(interaction)+'_DATA_binary_'+str(traininig_examples)+'_confusion_matrix.csv'
		cm=np.genfromtxt(confusion_file,delimiter=',',dtype='int')
		accuracies[i,1]=(cm[0,0]+cm[1,1])/float(1024)
	#plotBars(accuracies,counts,labels,ids_target)
	plotLines(accuracies,counts,labels,ids_target)
		
	

if __name__ == "__main__":
	compareActivations()
	