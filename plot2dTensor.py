import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot
from shapely.geometry.polygon import Polygon
from shapely import affinity
from scipy.spatial import Voronoi, voronoi_plot_2d
import seaborn as sns
import h5py
import sys

def plotFillingDummy():
	#cup of 
	#cup=Polygon([(0,0),(0.01,0),(0.01,-0.09),(0.07,-0.09),(0.07,0),(0.08,0),(0.08,-0.1),(0,-0.1)])
	#bowl
	cup=Polygon([(-0.07,0),(-0.06,0),(0.01,-0.09),(0.07,-0.09),(0.14,0),(0.15,0),(0.08,-0.1),(0,-0.1)])
	cup = affinity.translate(cup, xoff=-0.04,yoff=-0.02)
	x_cup,y_cup = cup.exterior.xy
	#make 'denser' pointclouds by interpolating every 1cm
	n_points=int(cup.length//0.01)
	dense_cup=np.zeros((n_points,2))
	for i in range(1,n_points):
		distance=i*0.01
		some_point=cup.boundary.interpolate(distance)
		dense_cup[i,0]=some_point.x
		dense_cup[i,1]=some_point.y
	#print(dense_cup)
	#x_cup=[x f]
	#vertices for sink
	sink_1=Polygon([(0,0),(0.025,0),(0.025,0.10),(0,0.10)])
	sink_1 = affinity.translate(sink_1, xoff=-0.0125)
	x,y = sink_1.exterior.xy
	n_points=int(sink_1.length//0.01)
	dense_sink1=np.zeros((n_points,2))
	for i in range(1,n_points):
		distance=i*0.01
		some_point=sink_1.boundary.interpolate(distance)
		dense_sink1[i,0]=some_point.x
		dense_sink1[i,1]=some_point.y
	#print(dense_sink1)

	# sink of 35x20 cm
	sink_2=Polygon([(0,0),(0.05,0),(0.05,-0.15),(0.35,-0.15),(0.35,0),(0.4,0),(0.4,-0.2),(0,-0.2)])
	#translate 17.5 cm to the left and 10 cm down
	sink_2 = affinity.translate(sink_2, xoff=-0.2,yoff=-0.1)
	x2,y2 = sink_2.exterior.xy
	n_points=int(sink_2.length//0.01)
	dense_sink2=np.zeros((n_points,2))
	for i in range(1,n_points):
		distance=i*0.01
		some_point=sink_2.boundary.interpolate(distance)
		dense_sink2[i,0]=some_point.x
		dense_sink2[i,1]=some_point.y
	#print(dense_sink2)

	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111)
	ax.set_xlim(-0.3,.3)
	ax.set_ylim(-0.3,0.3)
	ax.plot(x,y,color='g',linestyle=':',linewidth=4)
	ax.plot(x2,y2,color='g',linestyle=':',linewidth=4)
	ax.plot(x_cup,y_cup,color='b',linestyle=':',linewidth=4)
	#compute voronoi
	all_vertices=np.concatenate((dense_cup,dense_sink1,dense_sink2),axis=0)
	print(all_vertices.shape)
	vor = Voronoi(all_vertices)
	vor_vertices=vor.vertices
	voronoiCells=True
	print(len(vor.ridge_vertices))
	print(vor.ridge_points.shape)
	if voronoiCells:
		for i in range(len(vor.ridge_vertices)):
			point_ids=vor.ridge_vertices[i]
			if point_ids[0]<0 or point_ids[1]<0:
				continue
			if vor.ridge_points[i,0]<dense_cup.shape[0] and vor.ridge_points[i,1]>dense_cup.shape[0]:
				ax.plot(vor_vertices[point_ids,0],vor_vertices[point_ids,1],color='r',linewidth=5,linestyle='-')
			elif vor.ridge_points[i,0]>dense_cup.shape[0] and vor.ridge_points[i,1]<dense_cup.shape[0]:
				ax.plot(vor_vertices[point_ids,0],vor_vertices[point_ids,1],color='r',linewidth=5,linestyle='-')
			# else:
			#  	ax.plot(vor_vertices[point_ids,0],vor_vertices[point_ids,1],color='k',linewidth=1)

	#ax.scatter(0,0,0,c='y')
	
	#print(all_vertices.shape)
	plt.axis('off')
	plt.grid(False)
	#voronoi_plot_2d(vor)
	#plt.savefig(name, bbox_inches='tight',format='eps', dpi=300)
	#plt.pause(2)
	#plt.draw()
	#plt.close()
	plt.show()

def PrecRecPlots(file='/home/er13827/Metrics2.csv'):
	data=np.genfromtxt(file,delimiter=',',dtype='float',skip_header=1,usecols = (2,3,4,5,6,7,8) )
	labels=np.genfromtxt(file,delimiter=',',dtype='str',skip_header=0,usecols = (2,3,4,5,6,7,8),max_rows=1 )
	# print(labels)
	# print(data.shape)
	# print(data)
	fig = plt.figure(figsize=(20, 10))
	ax = fig.add_subplot(111)
	ax.set_xlim(0,1.05)
	ax.set_ylim(0,1.05)
	ax.set_xlabel('Recall',fontsize=18)
	ax.set_ylabel('Precision',fontsize=18)
	sns.set()
	sns.set_style("white")
	colors = ["#e74c3c","#9b59b6", "#3498db", "#34495e", "#2ecc71"]
	sns.set_context("poster")
	styles=[':','--','-',':','--','-']
	for i in range(data.shape[1]):
		# print(i)
		recall=data[1::2,i]
		precision=data[::2,i]
		sorted_ids=np.argsort(recall)
		precision=np.expand_dims(precision[sorted_ids],axis=1)
		recall=np.expand_dims(recall[sorted_ids],axis=1)
		# print(precision.shape)
		# print(recall.shape)
		padded_prec=np.vstack((precision[0],precision,0))
		padded_recall=np.vstack((0,recall,recall[-1]))
		#padding at the end and beginning
		if i<3:
			ax.plot(padded_recall,padded_prec,label=labels[i],linestyle=styles[i],color=colors[3],linewidth=3)
		elif i<6:
			ax.plot(padded_recall,padded_prec,label=labels[i],linestyle=styles[i],color=colors[4],linewidth=3)
		else:
			ax.plot(padded_recall,padded_prec,label=labels[i],linestyle=styles[2],color=colors[2],linewidth=3)
		#print('Rec',padded_recall.shape)
		#print('Prec',padded_prec.shape)
		print('%s %f'%(labels[i],np.trapz(y=padded_prec[:,0],x=padded_recall[:,0])))
	#ax.autoscale(tight=True)
	plt.tight_layout()
	plt.tick_params(labelsize=18)
	ax.legend(loc=0, ncol=4)

	plt.show()

def plotPrec():
	f=h5py.File('basic_stats_992_13.h.old')
	which=-1
	data=f['data'][:]
	np.savetxt("newer_992_data.csv", data[-1,...], delimiter=",")
	precisions1=data[which,:,0]
	print(precisions1)
	recalls1=data[which,:,1]
	f1s1=2*precisions1*recalls1/(precisions1+recalls1)
	# csv_file_name='prec_rec992.csv'
	# header='Precision,Recall,F1\n'
	# with open(csv_file_name,"w") as csv_file:
	# 	for i in range(precisions.size):
	# 		csv_file.write('%4f,%.4f,%.4f\n'%(precisions[i],recalls[i],f1s[i]))
	f=h5py.File('basic_stats_994_13.h.old')
	data2=f['data'][:]
	np.savetxt("newer_994_data.csv", data2[-1,...], delimiter=",")
	precisions2=data2[which,:,0]
	recalls2=data2[which,:,1]
	f1s2=2*precisions2*recalls2/(precisions2+recalls2)
	affs=np.genfromtxt('tmp992.csv',dtype='str',skip_header=1,delimiter=',')
	labels=[x[1]+'-'+x[2] for x in affs]
	#print(labels)
	print(data.shape)
	fig = plt.figure(figsize=(20,8 ))
	ax = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	# ax.set_xlim(0,1.05)
	# ax.set_ylim(0,1.05)
	ax.set_xlabel('Affordances',fontsize=18)
	ax.set_ylabel('Precision (%)',fontsize=18)
	ax2.set_xlabel('Affordances',fontsize=18)
	ax2.set_ylabel('Precision (%)',fontsize=18)
	sns.set()
	sns.set_style("white")
	colors = ["#e74c3c","#9b59b6", "#3498db", "#34495e", "#2ecc71"]
	#sns.set_context("poster")
	wd=0.4
	zeros_2=np.nonzero(data2[which,:,0])
	zeros_1=np.nonzero(data[which,:,0])
	comparable_ids=np.intersect1d(zeros_1,zeros_2)
	print('Comparable %d'%comparable_ids.size)
	#sorted_ids=np.arange(data1.shape[1])
	sorted_ids=np.argsort(data[which,comparable_ids,0])
	#sorted_ids=sorted_ids[::-1]
	half=sorted_ids.size//2
	print(sorted_ids[half:].shape)
	X = np.arange(half)
	ax.grid(zorder=0)
	ax.bar(X, 100*data[which,comparable_ids[sorted_ids[:half]],0],  width = wd, align='center',label='saliency',color=colors[4],zorder=3)
	# ax.bar(X, data[which,sorted_ids[:half],1],  width = wd, align='center',label='recall',color=colors[3],zorder=3)
	ax.bar(X+wd, 100*data2[which,comparable_ids[sorted_ids[:half]],0],  width = wd, align='center',label='iT agglomeration',color=colors[3],zorder=3)
	thisLabels=labels=[x[1]+'-'+x[2] for x in affs[comparable_ids[sorted_ids[:half]]]]
	ax.set_xticklabels(thisLabels,rotation=45,ha='right',fontsize=18)
	ax.set_xticks(X+wd/2)
	ax.legend(loc=0, ncol=2,fontsize=18)
	X = np.arange(sorted_ids.size-half)
	ax.set_ylim(0,100)
	ax2.bar(X, 100*data[which,comparable_ids[sorted_ids[half:]],0],  width = wd, align='center',label='saliency',color=colors[4],zorder=3)
	# ax2.bar(X, data[which,sorted_ids[half:],1],  width = wd, align='center',label='recall',color=colors[3],zorder=3)
	ax2.bar(X+wd, 100*data2[which,comparable_ids[sorted_ids[half:]],0],  width = wd, align='center',label='iT agglomeration',color=colors[3],zorder=3)
	thisLabels=labels=[x[1]+'-'+x[2] for x in affs[comparable_ids[sorted_ids[half:]]]]
	ax2.set_ylim(0,100)
	ax2.set_xticklabels(thisLabels,rotation=45,ha='right',fontsize=18)
	ax2.set_xticks(X+wd/2)
	ax2.grid(zorder=0)
	ax.yaxis.set_major_locator(MaxNLocator(4))
	ax2.yaxis.set_major_locator(MaxNLocator(4))
	#ax.autoscale(tight=True)
	plt.tight_layout()
	#plt.tick_params(labelsize=18)
	#plt.savefig('prec_all.eps',bbox_inches='tight',format='eps', dpi=100)
	plt.show()

def confusion_matrix(matrix_file):
	#sns.set(context='poster')
	values=np.genfromtxt(matrix_file,delimiter=',',dtype='float')
	fig, ax = plt.subplots(figsize=(10,10))
	im = ax.imshow(values.T)
	name=matrix_file.split('/')[-1].split('.')[0]
	#ax.set_title(name)
	labels=['Filling','Hanging','Placing','Sitting','Background']
	#ticks_=np.arange(0,values.shape[0],10)
	ticks_=np.arange(0,5,1)
	ax.set_xticks(ticks_)
	ax.set_xlabel('Actual',fontsize=22)
	ax.set_ylabel('Predcited',fontsize=22)
	ax.set_xticklabels(labels,rotation=45,ha='left',fontsize=18)
	ax.set_yticklabels(labels,fontsize=18)
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_yticks(ticks_)
	plt.tight_layout()
	plt.show()



if __name__ == '__main__':
	#PrecRecPlots()
	plotFillingDummy()
	#plotPrec()
	#confusion_matrix(sys.argv[1])