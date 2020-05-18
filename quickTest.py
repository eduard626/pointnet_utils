from __future__ import print_function
from getTensorTraining import *
import csv
import time
if __name__ == "__main__":
	# if len(sys.argv)<2:
	# 	print('Need results main file: Affordance_object_scene_..._id....pcd')
	# 	sys.exit()
	# if ".pcd" not in sys.argv[1]:
	# 	print ('Need pcd file')
	# 	sys.exit()
	#from itertools import *
	getSingleTraining(sys.argv[1])
	sys.exit()
	#computeResultStats(13)
	# new_counts,names=readCounts()
	# with open('file_lists2.csv', 'w') as f:
	# 	writer=csv.writer(f)
	# 	writer.writerows(names)
	# np.savetxt('filtered_counts2.csv', new_counts.astype(int), delimiter=",",fmt='%.0f')
	# sys.exit()
	aff_target=-1
	if len(sys.argv)>1:
		aff_target=int(sys.argv[1])
		print('Single affordace %d'%aff_target)
	# sys.exit()
	new_c=np.genfromtxt('filtered_counts2.csv',delimiter=',',dtype='int')
	with open('file_lists2.csv', 'r') as f:
		reader=csv.reader(f)
		new_n=list(reader)

	#samples=512//8
	samples=32
	points=4096
	thisMany=np.nonzero(new_c>=samples)[0]
	print('Actually using %d affordances'%(thisMany.size))
	print(thisMany)
	samples2=512
	thisMany2=np.nonzero(new_c>=samples2)[0]
	#print(thisMany2)
	dif=np.setdiff1d(thisMany,thisMany2)
	print(dif)
	#sys.exit()
	thisMany=dif;
	#bar = Bar('Generating dataset',max=thisMany.size)
	if aff_target<0:
		for i in range(thisMany.size):
			interaction=thisMany[i]
			#print('Affordance %d: Sample %d clouds with %d points '%(interaction,samples,points))
			#start = time.time()
			print('Affordance %d'%(interaction))
			name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
			if os.path.exists(name):
				continue
			if new_c[interaction]>samples:
				samples=new_c[interaction]
			dataPoints,dataClouds=sampleFromFile(interaction,new_n[interaction],samples)
			#end = time.time()
			#print(end - start)
			if dataPoints.shape[0]<samples:
				print('Something wrong with affordance %d'%(interaction))
				continue
			someLabels=np.zeros((samples,1),dtype=np.uint8)
			name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
			if not os.path.exists(name):
				save_h5(name,dataClouds,someLabels,'float32','uint8')
			name='/home/er13827/space/pointnet2/data/new_data_centered/dataPoints_'+str(interaction)+".h5"
			if not os.path.exists(name):
				save_h5(name,dataPoints,someLabels,'float32','uint8')
			del dataPoints
			del dataClouds
	else:
		interaction=thisMany[aff_target]
		#print('Affordance %d: Sample %d clouds with %d points '%(interaction,samples,points))
		#start = time.time()
		print('Getting Affordance %d only'%(interaction))
		name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
		if os.path.exists(name):
			print('File exists %s'%name)
			sys.exit()
		dataPoints,dataClouds=sampleFromFile(interaction,new_n[interaction],samples)
		#end = time.time()
		#print(end - start)
		if dataPoints.shape[0]!=samples:
			print('Something wrong with affordance %d'%(interaction))
			sys.exit()
		someLabels=np.zeros((samples,1),dtype=np.uint8)
		name='/home/er13827/space/pointnet2/data/new_data_centered/dataClouds_'+str(interaction)+".h5"
		if not os.path.exists(name):
			save_h5(name,dataClouds,someLabels,'float32','uint8')
		name='/home/er13827/space/pointnet2/data/new_data_centered/dataPoints_'+str(interaction)+".h5"
		if not os.path.exists(name):
			save_h5(name,dataPoints,someLabels,'float32','uint8')
		del dataPoints
		del dataClouds
