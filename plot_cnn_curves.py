import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def plot_curves(file):
	data=np.genfromtxt(file,delimiter=',',dtype='float',skip_header=1,usecols = (1,2) )
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	#ax.set_xlim(0,1.05)
	ax.set_ylim(0,1.05)
	ax.set_xlabel('Iteration',fontsize=18)
	ax.set_ylabel('Precision',fontsize=18)
	sns.set()
	sns.set_style("white")
	colors = ["#e74c3c","#9b59b6", "#3498db", "#34495e", "#2ecc71"]
	sns.set_context("poster")
	ax.plot(data[:,0],data[:,1])
	plt.tight_layout()
	plt.tick_params(labelsize=18)
	ax.legend(loc=0, ncol=4)

	plt.show()


if __name__ == '__main__':
	if len(sys.argv)<2:
		print "Need csv data"
		sys.exit()
	if '.csv' not in argv[1]:
		print "Need csv data"
		sys.exit()
	plot_curves(argv[1])