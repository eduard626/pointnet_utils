import os
import sys
import numpy as np
import h5py
import pypcd
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

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

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

#this should be used to load the voxel surounding a good test-point
def load_pcd_data(filename,cols=(0,1,2),dataType=float):
	tmp_cloud=np.loadtxt(filename,skiprows=11,usecols=cols,dtype=dataType)
	return tmp_cloud

def load_pcd_data_binary(filename):
	pc=pypcd.PointCloud.from_path(filename)
	xyz = np.empty((pc.points, 3), dtype=np.float)
	rgb=np.empty((pc.points, 1), dtype=np.int)
	xyz[:, 0] = pc.pc_data['x']
	xyz[:, 1] = pc.pc_data['y']
	xyz[:, 2] = pc.pc_data['z']
	try:
		rgb = pc.pc_data['rgb']
	except Exception as e:
		error_msg=e

	return xyz,rgb

def sample_cloud(data,n_samples):
	idx = np.arange(data.shape[0])
	np.random.shuffle(idx)
	return data[idx[0:n_samples], :]

def load_ply_data(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# In principle I'll call this code for different directories containing data
# coming from one affordance in particular
# Need to compare that directory/affordance name agains descriptor 13,
# that has 92 affordances. Label is the id of affordance in that array of 92
if __name__ == '__main__':
	if len(sys.argv)<2:
		print "Need csv_descriptor"
		sys.exit()
	if ".csv" not in sys.argv[1]:
		print "CSV with metrics"
		sys.exit()
	#path to results
	ind_res='~/deepnetwork/skynetDATA/Eduardo/Individual'

	if "tmp13." in sys.argv[1]:
		max_rad=0.89
	else:
		max_rad=0.5
	n_points=2048
	path=os.path.abspath(sys.argv[1])
	label_data=np.genfromtxt(sys.argv[1],dtype='str',skip_header=1,delimiter=',')
	all_data=np.zeros((label_data.shape[0],n_points,3),dtype=np.float32)
	all_labels=np.zeros((label_data.shape[0],1),dtype=np.uint8)
	for j in range(label_data.shape[0]):
		affordance_=label_data[j,1]
		object_=label_data[j,2]
		directory_=label_data[j,0]
		print('%d Working on %s %s'%(j,affordance_,object_))
		pos=path.rfind('/')
		#get the info file
		info_file=path[:pos]+'/'+directory_+'/ibs_full_'+affordance_+'_'+object_+'.txt'
		#print(info_file)
		#try to read info file
		try:
			with open(info_file) as fp:
				for i, line in enumerate(fp):
					if i==0:
						scene_name=line.split(':')[-1]
						scene_name=scene_name.replace("\n","")
					if i==8:
						scene_point_data=line.split(':')[-1]
						scene_point_data=scene_point_data.replace("\n","")
						scene_point_data=scene_point_data.split(',')
						#print(scene_point_data)
					if i>8:
						break
		except:
			print('Problem reading file %s'%info_file)
			# read the scene file
		if ".pcd" in scene_name:
			scene_file=path[:pos]+'/'+directory_+'/'+scene_name
			print('Reading scene %s'%scene_file)
			scene_cloud=load_pcd_data(scene_file)
		else:
			if affordance_=='Ride' or affordance_=='Sit':
				scene_file=path[:pos]+'/'+directory_+'/'+scene_name+".ply"
				scene_cloud=load_ply_data(scene_file)
			else:
				scene_file=path[:pos]+'/'+directory_+'/'+scene_name+".pcd"
				print('Reading scene %s'%scene_file)
				scene_cloud=load_pcd_data(scene_file)
		
		scenePoint=np.array([scene_point_data[0],scene_point_data[1],scene_point_data[2]],dtype='float32')
		#extract a 1m voxel around scenePoint
		print('extract voxel around point from cloud of size %dx%d'%(scene_cloud.shape[0],scene_cloud.shape[1]))
		print(scenePoint)
		from sklearn.neighbors import KDTree
		kdt = KDTree(scene_cloud, metric='euclidean')
		ind = kdt.query_radius(scenePoint.reshape(1,-1),r=max_rad)
		#print(ind.shape)
		point_ids=np.expand_dims(ind,axis=0)[0,0].reshape(1,-1)
		#print(scene_cloud[point_ids[0,:],:].shape)
		cloud_sample=sample_cloud(scene_cloud[point_ids[0,:],:],n_points)
		all_data[j,...]=cloud_sample
		all_labels[j]=j
		#print(all_data[i,...])
	save_h5('theFile.h5',all_data,all_labels,'float32','uint8')