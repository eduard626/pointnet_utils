import h5py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#f=h5py.File('/home/er13827/deepNetwork/halDATA/Eduardo/PointNet/miniO_AffordancesDataset_train_FHPS_1.h5')
f=h5py.File('/home/er13827/space/pointnet2/data/affordances/binaryOc_AffordancesDataset_train91_1.h5')
lables=['Filling','Hanging','Placing','Sitting','Non']
d = f['data'][:]
print(d.shape)
l = f['label'][:]
print(l)
print(l[:20].T)
unique_classes=np.unique(l)
print(unique_classes)
fig = plt.figure()
plt.ion()
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
for i in range(unique_classes.size):
	aff_class_ids=np.nonzero(l==unique_classes[i])[0]
	anExample_id=np.random.randint(aff_class_ids.shape[0])
	anExample=d[aff_class_ids[anExample_id],...]
	print('%s'%lables[i])
	print(anExample.shape)
	aff_class_ids=np.nonzero(l==(unique_classes.size-1))[0]
	aNegativeExample_id=np.random.randint(aff_class_ids.shape[0])
	aNegativeExample=d[aff_class_ids[aNegativeExample_id],...]
	print('%s'%lables[unique_classes.size-1])
	print(aNegativeExample.shape)
	ax.scatter(anExample[:,0],anExample[:,1],anExample[:,2],s=1,c='b')
	ax2.scatter(aNegativeExample[:,0],aNegativeExample[:,1],aNegativeExample[:,2],s=3,c='r')
	plt.pause(10)
	plt.draw()
	ax.clear()
	ax2.clear()
#i=2
#ax.scatter(d[i,:,0],d[i,:,1],d[i,:,2],s=3)
#print(l[i,...])
#plt.show()