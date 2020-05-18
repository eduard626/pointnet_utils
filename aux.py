from getTensorTraining import *

data_path='/media/hal/DATA/Eduardo/PointNet/'
TEST_EXAMPLES=512
data_points=1024*2
max_rad=0.806884
n_points=2048*2
n_samples=128
n_orientations=8
TRAIN_EXAMPLES=1


def getMultiDatasetSingle():
	#read names
	file='/home/er13827/space/testing/tmp13.csv'
	big_labels=np.genfromtxt(file,dtype='str',delimiter=',',skip_header=1)
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	actual_ids=np.arange(new_c.shape[0])
	ids_target=np.nonzero(new_c>=TEST_EXAMPLES)[0]
	print(ids_target.shape)
	aff_ids=ids_target.astype(int)
	print(aff_ids)
	print(actual_ids)
	print(big_labels[:,1])
	affordances_class_ids=np.empty((4,2),dtype=np.int)
	last_read=''
	counter=0
	for i in range(ids_target.size):
		affordance=big_labels[ids_target[i],1].split('-')[0]
		if last_read!=affordance:
			affordances_class_ids[counter,0]=i
			last_read=affordance
			counter+=1

	affordances_class_ids[:,1]=np.roll(affordances_class_ids[:,0]-1,-1)
	print(affordances_class_ids)
	sys.exit()
	#generate 4-affordance dataset
	new_training_data=np.empty(((affordances_class_ids.shape[0]+1)*TRAIN_EXAMPLES,data_points,3),dtype=np.float32)
	new_testing_data=np.empty(((affordances_class_ids.shape[0]+1)*TEST_EXAMPLES,data_points,3),dtype=np.float32)
	new_training_labels=np.empty(((affordances_class_ids.shape[0]+1)*TRAIN_EXAMPLES,1),dtype=np.uint8)
	new_testing_labels=np.empty(((affordances_class_ids.shape[0]+1)*TEST_EXAMPLES,1),dtype=np.uint8)
	data_start_training=0
	data_start_testing=0
	negatives_train=np.empty((1*TRAIN_EXAMPLES,data_points,3),dtype=np.float32)
	negatives_test=np.empty((1*TEST_EXAMPLES,data_points,3),dtype=np.float32)
	#sample negatives from a random dataset
	negatives_from=np.random.randint(affordances_class_ids.shape[0])
	file_name=''
	for i in range(affordances_class_ids.shape[0]):
		stard_id=affordances_class_ids[i,0]
		end_id=affordances_class_ids[i,1]
		if end_id<0:
			random_aff=stard_id
		else:
			random_aff=np.random.randint(stard_id,end_id+1)
		file_name=file_name+big_labels[random_aff,1][0]
		#read data for this affordance
		training_data_file=data_path+'binaryO_AffordancesDataset_train'+str(random_aff)+'_'+str(TRAIN_EXAMPLES)+'.h5'
		#something weird happens with Sitting, need to read data from elsewhere
		testing_data_file=data_path+'binary_AffordancesDataset_test'+str(random_aff)+'_'+str(TRAIN_EXAMPLES)+'.h5'
		print('%s %s'%(training_data_file,big_labels[random_aff,1]))
		training_data,training_labels=load_h5(training_data_file)
		idx = np.arange(training_data.shape[1])
		np.random.shuffle(idx)
		training_data=training_data[:,idx[:data_points],:]

		training_ids=np.nonzero(training_labels==0)[0]
		
		print('%s %s'%(testing_data_file,big_labels[random_aff,1]))
		testing_data,testing_labels=load_h5(testing_data_file)
		idx = np.arange(testing_data.shape[1])
		np.random.shuffle(idx)
		testing_data=testing_data[:,idx[:data_points],:]

		testing_ids=np.nonzero(testing_labels==0)[0]
		data_end_training=data_start_training+TRAIN_EXAMPLES
		new_training_data[data_start_training:data_end_training,...]=training_data[training_ids,...]
		data_end_testing=data_start_testing+TEST_EXAMPLES
		new_testing_data[data_start_testing:data_end_testing,...]=testing_data[testing_ids,...]
		new_training_labels[data_start_training:data_end_training,...]=i*np.ones((TRAIN_EXAMPLES,1),dtype=np.uint8)
		new_testing_labels[data_start_testing:data_end_testing,...]=i*np.ones((TEST_EXAMPLES,1),dtype=np.uint8)
		if i==negatives_from:
			negative_ids=np.nonzero(training_labels)[0]
			negatives_train=training_data[negative_ids,...]
			negative_ids=np.nonzero(testing_labels)[0]
			negatives_test=testing_data[testing_ids,...]
		print(new_testing_labels[data_start_testing:data_start_testing+10].T)
		print(new_training_labels.T)

		data_start_training=data_end_training
		data_start_testing=data_end_testing
		
	new_training_data[data_start_training:,...]=negatives_train
	new_testing_data[data_start_testing:,...]=negatives_test
	new_testing_labels[data_end_testing:,...]=(affordances_class_ids.shape[0])*np.ones((TEST_EXAMPLES,1),dtype=np.uint8)
	new_training_labels[data_end_training,...]=(affordances_class_ids.shape[0])*np.ones((TRAIN_EXAMPLES,1),dtype=np.uint8)
	print(new_testing_labels[data_start_testing:data_start_testing+10].T)
	print(new_training_labels.T)
	idx=np.arange(new_training_labels.shape[0])
	np.random.shuffle(idx)
	new_training_labels=new_training_labels[idx,...]
	new_training_data=new_training_data[idx,...]
	new_data_file=data_path+'miniO_AffordancesDataset_train_'+file_name+'_'+str(TRAIN_EXAMPLES)+'.h5'
	if os.path.exists(new_data_file):
		os.system('rm %s' % (new_data_file))
	save_h5(new_data_file,new_training_data,new_training_labels,'float32','uint8')
	print(new_data_file)
	idx=np.arange(new_testing_labels.shape[0])
	np.random.shuffle(idx)
	new_testing_labels=new_testing_labels[idx,...]
	new_testing_data=new_testing_data[idx,...]
	new_data_file=data_path+'miniO_AffordancesDataset_test_'+file_name+'_'+str(TRAIN_EXAMPLES)+'.h5'
	if os.path.exists(new_data_file):
		os.system('rm %s' % (new_data_file))
	save_h5(new_data_file,new_testing_data,new_testing_labels,'float32','uint8')
	print(new_data_file)

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

def rotate_point_cloud_by_angle_local(batch_data, rotation_angle):
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

def getCenteredTestingSetAugmented(training_examples=512):
	new_c=np.genfromtxt('filtered_counts.csv',delimiter=',',dtype='int')
	with open('file_lists.csv', 'r') as f:
		reader=csv.reader(f)
		new_n=list(reader)
	labels_aff=np.genfromtxt('/home/er13827/space/testing/tmp13.csv',dtype='str',skip_header=1,delimiter=',')
	# i_names,i_idx=np.unique(labels[:,0],return_index=True)
	# i_idx_sorted=np.argsort(i_idx)
	#print(i_names[i_idx_sorted])
	#print(i_idx)
	print(labels_aff.shape)
	samples=training_examples
	points=4096
	ids_target=np.nonzero(new_c>=samples)[0]
	new_data=np.empty((samples*2,points,3),dtype=np.float32)
	new_data_training=np.empty((training_examples*2,points,3),dtype=np.float32)
	new_labels=np.empty((samples*2,1),dtype=np.uint8)
	new_labels_training=np.empty((training_examples*2,1),dtype=np.uint8)
	all_ids=np.arange(ids_target.size)
	#print(all_ids)
	#bar = Bar('Creating actual dataset', max=ids_target.shape[0])
	for i in range(ids_target.size):
		np.random.shuffle(all_ids)
		interaction=ids_target[i]
		if i==0:
			interaction=90
		else:
			sys.exit()
		name_data='/home/er13827/space/pointnet2/data/new_data_centered_augmented/binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(training_examples)+'.h5'
		#if not os.path.exists(name_data):
		#read dataset from data dir
		thisAffordance=labels_aff[interaction,0]
		#get data from a different interaction
		gotIt=False
		random_aff=0
		j=0
		while not gotIt:
			random_aff=ids_target[all_ids[j]]
			#print('Interaction %d Target %d'%(random_aff,all_ids[j]))
			#print('C:%s T:%s'%(labels[random_aff,0],thisAffordance))
			if labels_aff[random_aff,0]!=thisAffordance:
				print('Sampling from %s for %s dataset'%(labels_aff[random_aff,0],thisAffordance))
				gotIt=True
			else:
				j+=1
		name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
		data,labels=load_h5(name)
		print('read %s'%name)
		interaction2=random_aff
		name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction2)+".h5"
		data2,labels2=load_h5(name)
		print('read %s'%name)
		#half negative examples will be noise and half are example from a different affordance
		half=samples/2
		random_noise_data=genereateNoisyData(np.array([[0,0,0]]),max_rad,points,half)
		print('Create noise %d'%random_noise_data.shape[0])
		#first half of samples is from random affordance
		new_data[:half,...]=data2[:half,...]
		#all these labels are 1
		new_labels[:samples,...]=np.ones((samples,1),dtype=np.uint8)
		#second half ofsamples is from random affordance
		new_data[half:2*half,...]=random_noise_data
		# take first 256 and rotate to get more data
		n_splits=2
		n_rot=1
		lim=samples//n_splits
		toAugment=data[:lim,...]
		for i in range(n_rot):
			angle=(i+1)*(np.pi*2)/8
			print('Rotate %f'%angle)
			#print(toAugment[0,:3,:])
			someDataRot=np.zeros((lim,points,3),dtype=np.float32)
			for k in range(lim):
				#print(toAugment[k,0,:])
				someDataRot[k,...]=rotate_point_cloud_by_angle_local(toAugment[k,...], angle)
				#someData=np.expand_dims(someData,axis=0)
				#print(someData.shape)
			toAugment=np.concatenate((toAugment,someDataRot),axis=0)
		new_data[samples:,...]=toAugment
		#read the original training cloud
		name_t='/home/er13827/space/pointnet2/data/affordances/binaryOc_AffordancesDataset_train'+str(interaction)+'_1.h5'
		data_t,labels_t=load_h5(name_t)
		#make the last 'positive' instance of the test equal to the original training
		new_data[-1,...]=data_t[0,...]
		new_labels[samples:,...]=np.zeros((samples,1),dtype=np.uint8)
			#grab the second part and make it training data
		toAugment=data[lim:,...]
		for i in range(n_rot):
			angle=(i+1)*(np.pi*2)/8
			someDataRot=np.zeros((lim,points,3),dtype=np.float32)
			for k in range(lim):
				someDataRot[k,...]=rotate_point_cloud_by_angle_local(toAugment[k,...], angle)
			toAugment=np.concatenate((toAugment,someDataRot),axis=0)

		new_data_training[:training_examples,...]=toAugment
		new_labels_training[:training_examples,...]=np.zeros((training_examples,1),dtype=np.uint8)
		new_data_training[training_examples:training_examples+half,...]=data2[half:,...]
		random_noise_data=genereateNoisyData(np.array([[0,0,0]]),max_rad,points,half)
		new_data_training[-half:]=random_noise_data
		new_labels_training[training_examples:,...]=np.ones((training_examples,1),dtype=np.uint8)
			
		# else:
		# 	print('Reading and shuffling %d'%interaction)
		# 	new_data,new_labels=load_h5(name_data)
		# 	os.system('rm %s' % (name_data))

		#shuffle things around
		new_ids=np.arange(samples*2)
		np.random.shuffle(new_ids)
		new_labels=new_labels[new_ids,...]
		new_data=new_data[new_ids,...]
		save_h5(name_data,new_data,new_labels,'float32','uint8')
		name='/home/er13827/space/pointnet2/data/new_data_centered_augmented/binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(training_examples)+'_shuffledIds'
		np.save(name,new_ids)
		#save training
		new_ids=np.arange(samples*2)
		np.random.shuffle(new_ids)
		new_labels_training=new_labels_training[new_ids,...]
		new_data_training=new_data_training[new_ids,...]
		name_data='/home/er13827/space/pointnet2/data/new_data_centered_augmented/binaryOc_AffordancesDataset_train'+str(interaction)+'_'+str(training_examples)+'.h5'
		save_h5(name_data,new_data_training,new_labels_training,'float32','uint8')
		name='/home/er13827/space/pointnet2/data/new_data_centered_augmented/binaryOc_AffordancesDataset_train'+str(interaction)+'_'+str(training_examples)+'_shuffledIds'
		np.save(name,new_ids)

def getCenteredTestingSet():
	new_c=np.genfromtxt('filtered_counts2.csv',delimiter=',',dtype='int')
	with open('file_lists2.csv', 'r') as f:
		reader=csv.reader(f)
		new_n=list(reader)
	labels_aff=np.genfromtxt('/home/er13827/space/testing/tmp13.csv',dtype='str',skip_header=1,delimiter=',')
	# i_names,i_idx=np.unique(labels[:,0],return_index=True)
	# i_idx_sorted=np.argsort(i_idx)
	#print(i_names[i_idx_sorted])
	#print(i_idx)
	print(labels_aff.shape)
	samples=32
	points=4096
	ids_target=np.nonzero(new_c>=samples)[0]
	new_data=np.empty((samples*2,points,3),dtype=np.float32)
	new_labels=np.empty((samples*2,1),dtype=np.uint8)
	all_ids=np.arange(ids_target.size)
	#print(all_ids)
	#bar = Bar('Creating actual dataset', max=ids_target.shape[0])
	for i in range(ids_target.size):
		np.random.shuffle(all_ids)
		interaction=ids_target[i]
		name_data='/home/er13827/space/pointnet2/data/new_data_centered/binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(TRAIN_EXAMPLES)+'.h5'
		if not os.path.exists(name_data):
			#read dataset from data dir
			thisAffordance=labels_aff[interaction,0]
			#get data from a different interaction
			gotIt=False
			random_aff=0
			j=0
			while not gotIt:
				random_aff=ids_target[all_ids[j]]
				#print('Interaction %d Target %d'%(random_aff,all_ids[j]))
				#print('C:%s T:%s'%(labels[random_aff,0],thisAffordance))
				if labels_aff[random_aff,0]!=thisAffordance:
					print('Sampling from %s for %s dataset'%(labels_aff[random_aff,0],thisAffordance))
					gotIt=True
				else:
					j+=1
			name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
			data,labels=load_h5(name)
			if data.shape[0]>samples:
				samples=data.shape[0]
				half=samples//2
				samples=half*2
				print(samples)
				new_data=np.empty((samples*2,points,3),dtype=np.float32)
				new_labels=np.empty((samples*2,1),dtype=np.uint8)
			print('read %s'%name)
			interaction2=random_aff
			name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction2)+".h5"
			data2,labels2=load_h5(name)
			print('read %s'%name)
			#half negative examples will be noise and half are example from a different affordance
			random_noise_data=genereateNoisyData(np.array([[0,0,0]]),max_rad,points,half)
			print('Create noise %d'%random_noise_data.shape[0])
			
			new_data[:half,...]=data2[:half,...]
			new_labels[:2*half,...]=np.ones((2*half,1),dtype=np.uint8)
			new_data[half:2*half,...]=random_noise_data
			new_data[2*half:,...]=data[:2*half,...]
			new_labels[2*half:,...]=labels[:2*half,...]
			name_t='/home/er13827/space/pointnet2/data/affordances/binaryOc_AffordancesDataset_train'+str(interaction)+'_1.h5'
			data_t,labels_t=load_h5(name_t)
			#make the last 'positive' instance of the test equal to the original training
			new_data[-1,...]=data_t[0,...]
			#shuffle things around
		else:
			continue
			print('Reading and shuffling %d'%interaction)
			new_data,new_labels=load_h5(name_data)
			os.system('rm %s' % (name_data))

		new_ids=np.arange(samples*2)
		np.random.shuffle(new_ids)
		new_labels=new_labels[new_ids,...]
		new_data=new_data[new_ids,...]
		save_h5(name_data,new_data,new_labels,'float32','uint8')
		name='/home/er13827/space/pointnet2/data/new_data_centered/binaryOc_AffordancesDataset_test'+str(interaction)+'_'+str(TRAIN_EXAMPLES)+'_shuffledIds.h5'
		save_h5(name,new_data,new_ids,'float32','int16')
			#bar.next()
	#bar.finish()

if __name__ == "__main__":
	#getMultiDatasetSingle()
	#getCenteredTestingSet()
	getCenteredTestingSetAugmented(training_examples=62)