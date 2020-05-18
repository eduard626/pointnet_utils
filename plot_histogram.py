import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py
import os

data_paths=['/home/er13827/deepNetwork/skynetHome/Eduardo/GPU_Tensor/voronoi/build/','/home/er13827/deepNetwork/halHome/Eduardo/voronoi/build/',
'/media/WD2/testing/','/home/er13827/space/testing/','/home/er13827/space/Affordances/Hal/voronoi/build/']


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


files_ids=[]
files_paths=[]
for i in data_paths:
	base_name=i+'All_affordances_*_13_*.pcd'
	print(base_name)
	found=glob.glob(base_name)
	if len(found)>0:
		for file in found:
			id_=file.split('_')[-1].split('.')[0]
			if id_ not in files_ids:
				files_ids.append(id_)
				files_paths.append(i)
				if 'living' in file:	print 'living'
				elif 'kitchen' in file:	print 'kitchen'
				elif 'office' in file:	print 'office'
print('Found %d'%(len(files_ids)))
if not os.path.exists('data_counts_all.h5'):
	# count them
	counter=np.zeros(92,dtype=np.int32)
	for i in range(len(files_ids)):
		# build the file name 
		path=files_paths[i]
		file_id=files_ids[i]
		f=path+file_id+'_goodPointsX.pcd'
		real_f=glob.glob(f)[0]
		cloud,_,_=load_pcd_data_binary(real_f)
		print(i,len(files_ids))
		counts=np.bincount(cloud[:,0].astype(int),minlength=counter.size)
		counter+=counts

	print(counter)
	name='data_counts_all.h5'
	save_as_h5(name,counter,'int32')
else:
	data=load_from_h5('data_counts_all.h5')
	#read 13
	n1=np.genfromtxt('/home/er13827/space/testing/tmp13.csv',delimiter=',',skip_header=1,dtype='str')
	n2=np.genfromtxt('tmp992.csv',delimiter=',',skip_header=1,dtype='str')
	l1=np.asarray([x[1]+'-'+x[2] for x in n1])
	l2=np.asarray([x[1]+'-'+x[2] for x in n2])
	common,c1,c2=np.intersect1d(l1,l2,return_indices=True)
	#print(common.shape)
	#print(data.shape)
	lab=''
	ids_affordances=[]
	for i in range(n1[c1].shape[0]):
		if lab!=n1[c1[i],0]:
			ids_affordances.append(i)
			lab=n1[c1[i],0]
	bounds=np.expand_dims(np.asarray(ids_affordances),axis=1)
	bounds=np.concatenate((bounds,np.roll(bounds,-1,axis=0)-1),axis=1)
	# print(bounds)
	for i in range(n1[c1].shape[0]):
		if data[c1[i]]<10000:
			#get bound interval
			j=np.nonzero((bounds[:,0]<=c1[i]))[0][-1]
			print(n1[c1[i]],data[c1[i]],c1[i],j)
			data[c1[i]]=np.mean(data[c1[bounds[j,:]]])
	# print(data[c1])
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	sns.set()
	sns.set_style("white")
	sns.set_context("poster")
	ax.bar(np.arange(c1.size),data[c1],align='center',alpha=0.5)
	ax.set_ylim([0,10000])
	plt.tight_layout()
	plt.xlabel('Affordances',fontsize=18)
	plt.ylabel('Instances',fontsize=18)
	plt.show()
# y_pos = np.arange(len(objects))

# performance = [10,8,6,4,2,1]
 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')
 
# plt.show()