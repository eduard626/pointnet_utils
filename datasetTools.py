import numpy as np
from prep_affordanceData import (save_h5,load_h5)

def extractSubset(dataSet,new_size=0.5):
	tmp_data,tmp_labels=load_h5(dataSet)
	print(tmp_labels.shape[0])
	all_ids=np.arange(tmp_labels.shape[0])
	newSize=int(all_ids.size*new_size)
	print('New %d'%newSize)
	np.random.shuffle(all_ids)
	newData=tmp_data[all_ids[:newSize],...]
	newLabels=tmp_labels[all_ids[:newSize],...]
	print('New Data size %d %d'%(newData.shape[0],newLabels.shape[0]))
	return newData,newLabels

def split_data(list_of_files,train_size=0.8):
	for i in range(len(list_of_files)):
		tmp_data,tmp_labels=load_h5(list_of_files[i])
		if i>0:
			
			data=np.concatenate((data,tmp_data),axis=0)
			labels=np.concatenate((labels,tmp_labels),axis=0)
		else:
			data=tmp_data
			labels=tmp_labels
		print(tmp_data.shape)

	print('All data %d'%(data.shape[0]))
	all_ids=np.arange(data.shape[0])
	np.random.shuffle(all_ids)
	train_ids_size=int(all_ids.size*train_size)
	print(train_ids_size)
	train_ids=all_ids[:train_ids_size]
	new_train_data=data[train_ids,...]
	new_train_labels=labels[train_ids,...]
	test_ids=all_ids[train_ids_size:]
	new_test_data=data[test_ids,...]
	new_test_labels=labels[test_ids,...]

	print('Train data %d'%new_train_labels.shape[0])
	print('Test data %d'%new_test_labels.shape[0])
	save_h5('MultilabelDataSet_splitTrain4.h5',new_train_data,new_train_labels,'float32','uint8')
	save_h5('MultilabelDataSet_splitTest4.h5',new_test_data,new_test_labels,'float32','uint8')
	np.save('MultilabelDataSet_splitTest4.npy',test_ids)

def extractSingleLabeledData(data_file):
	data,label=load_h5(data_file)
	print(label.shape)
	train_examples=512
	test_examples=128
	examples=train_examples+test_examples
	print(examples*label.shape[1],data.shape[1],3)
	new_data_train=np.zeros((train_examples*label.shape[1],data.shape[1],3),dtype=np.float32)
	new_labels_train=np.zeros((train_examples*label.shape[1],1),dtype=np.int32)

	new_data_test=np.zeros((test_examples*label.shape[1],data.shape[1],3),dtype=np.float32)
	new_labels_test=np.zeros((test_examples*label.shape[1],1),dtype=np.int32)

	#for every affordance
	st=0
	st2=0
	for i in range(label.shape[1]):
		#get the pointclouds of this affordance
		target_indices=np.nonzero(label[:,i])[0]
		#print('Aff %d %d'%(i,target_indices.size))
		to_sample_from=np.arange(target_indices.size)
		np.random.shuffle(to_sample_from)
		if to_sample_from.size <(train_examples+test_examples):
			real_train_examples=int(to_sample_from.size*.8//1)
			#print(real_train_examples)
			real_test_examples=to_sample_from.size - real_train_examples
			print('Less data from %d,%d'%(real_train_examples,real_test_examples))
		else:
			real_train_examples=train_examples
			real_test_examples=test_examples

		ed=st+real_train_examples
		ed2=st2+real_test_examples

		real_sample=target_indices[to_sample_from[:real_train_examples]]
		real_sample_test=target_indices[to_sample_from[real_train_examples:real_train_examples+real_test_examples]]

		new_data_train[st:ed,...]=data[real_sample,...]
		new_labels_train[st:ed,...]=i

		new_data_test[st2:ed2,...]=data[real_sample_test,...]
		new_labels_test[st2:ed2,...]=i

		st=ed
		st2=ed2
	# get the real data in case some affordances had less examples than the target
	new_data_train=new_data_train[:ed,...]
	new_labels_train=new_labels_train[:ed,...]
	new_data_test=new_data_test[:ed2,...]
	new_labels_test=new_labels_test[:ed2,...]

	#shuffle things
	ids=np.arange(new_labels_train.shape[0])
	np.random.shuffle(ids)
	new_data_train=new_data_train[ids,...]
	new_labels_train=new_labels_train[ids,...]

	ids=np.arange(new_labels_test.shape[0])
	np.random.shuffle(ids)
	new_data_test=new_data_test[ids,...]
	new_labels_test=new_labels_test[ids,...]


	print('New binary train data %d'%new_labels_train.shape[0])
	print('New binary test data %d'%new_labels_test.shape[0])
	name='SinglelabelDataSet_train_'+data_file.split('.')[0].split('_')[-1]+'.h5'
	print(name)
	save_h5(name,new_data_train,new_labels_train,'float32','uint8')
	name='SinglelabelDataSet_test_'+data_file.split('.')[0].split('_')[-1]+'.h5'
	print(name)
	save_h5(name,new_data_test,new_labels_test,'float32','uint8')

if __name__ == '__main__':
	files=['MultilabelDataSet_real-kitchen1.h5','MultilabelDataSet_living-room6.h5']
	#split_data(list_of_files=files)
	extractSingleLabeledData(files[0])
	#data,labels=extractSubset('MultilabelDataSet_real-kitchen1.h5')
	#save_h5('MultilabelDataSet_real-kitchen1Test.h5',data,labels,'float32','uint8')


##SplitTest1 -> kitchen5+living-room6
##SplitTest2 -> kitchen5+real-kitchen1
##SplitTest3 -> kitchen5+real-kitchen+living-room6
##SplitTest4 -> real-kitchen1 + living-room6
##splitTest5 -> real-kitchen1 + real-kitchen2