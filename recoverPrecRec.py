import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.ticker as ticker

def recoverPrecisionRecall(path_to_logs='/home/er13827/deepNetwork/halDATA/Eduardo/PointNet2/Results_rad_centered/'):
	matches = []
	affordance_ids=[]
	for root, dirnames, filenames in os.walk(path_to_logs):
		for filename in fnmatch.filter(filenames, 'log_train.txt'):
			matches.append(os.path.join(root, filename))
			affordance=root.split('/')[-2].split('_')[1]
			affordance_ids.append(affordance)
	anArray=np.asarray(affordance_ids,dtype=np.int32)
	id_sorted=np.argsort(anArray)
	#print(anArray[id_sorted])
	precs_=np.zeros((len(matches),2))
	recs_=np.zeros((len(matches),2))
	for i in range(id_sorted.size):
		a_file=matches[id_sorted[i]]
		with open(a_file, "r") as file:
			lastlines = (list(file)[-10:])
			#remove \n
			content=[line.strip() for line in lastlines]
			#print(content[0])
			aPrec=content[0].split(' ')[-1]
			precs_[i,0]=float(aPrec)
			#print(content[7])
			aPrec=content[7].split(' ')[-1]
			precs_[i,1]=float(aPrec)
			#print(content[1])
			aRec=content[1].split(' ')[-1]
			recs_[i,0]=float(aRec)
			#print(content[8])
			aRec=content[8].split(' ')[-1]
			recs_[i,1]=float(aRec)

	labels=np.genfromtxt('/home/er13827/space/testing/tmp13.csv',dtype='str',skip_header=1,delimiter=',')
	plot_labels=[label[1]+'-'+label[2] for label in labels]
	#print(plot_labels)
	#print(id_sorted[:42])
	#print(anArray[id_sorted[:42]])
	some_ids=anArray[id_sorted[:42]]
	plot_labels1=[]
	for i in range(some_ids.size):
		plot_labels1.append(plot_labels[some_ids[i]])
	plot_labels2=[]
	some_ids=anArray[id_sorted[42:]]
	for i in range(some_ids.size):
		plot_labels2.append(plot_labels[some_ids[i]])
	#print(plot_labels1)
	fig = plt.figure(figsize=(15, 5))
	plt.ion()
	sns.set()
	ax = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	#ax.plot(all_counts[0,:],c='r')
	wd=0.4
	X = np.arange(42)
	ax.bar(X , precs_[:42,1], width = wd, color='b',align='center',label='precision')
	ax.bar(X + wd, recs_[:42,1], width = wd, color='g',align='center',label='recall')
	ax.autoscale(tight=True)
	ax.set_xticks(X+wd/2)
	ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
	ax.set_xticklabels(plot_labels1,rotation=45,ha='right')
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	X2 = np.arange(43)
	ax2.bar(X2 , precs_[42:,1], width = wd, color='b',align='center')
	ax2.bar(X2 + wd, recs_[42:,1], width = wd, color='g',align='center')
	ax2.autoscale(tight=True)
	ax2.set_xticks(X2+wd/2)	
	ax2.yaxis.set_major_locator(ticker.MaxNLocator(2))
	ax2.set_xticklabels(plot_labels2,rotation=45,ha='right')
	name='individual_pre_rec.eps'
	plt.tight_layout()
	#plt.savefig(name,bbox_inches='tight',format='eps', dpi=300)
	plt.draw()
	plt.pause(2)
	plt.show()

if __name__ == '__main__':
	recoverPrecisionRecall()