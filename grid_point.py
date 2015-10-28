import numpy as np 
import matplotlib.pyplot as plt
import asciitable 
import pdb

#######################################################################
def defineGrid(e,n,box_size=50.,res = 1.):
    #get the resolution 
    dxy = res           # in meters
    boxSize = box_size  # in meters
    #border
    max_e = e + boxSize*.5
    min_e = e - boxSize*.5
    max_n = n + boxSize*.5
    min_n = n - boxSize*.5

    acceptable_number = np.arange(1000) * 16.

    nx =  int( round( (max_e - min_e)/dxy) //2 * 2 ) 
    ny =  int( round( (max_n - min_n)/dxy) //2 * 2 ) 
   
    nx = acceptable_number[np.abs(acceptable_number-nx).argmin()]
    ny = acceptable_number[np.abs(acceptable_number-ny).argmin()]

    #reajust resolution to conserve the box size as define
    dxy = boxSize/(nx)

    grid_e = np.arange(nx)*dxy + min_e 
    grid_n = np.arange(ny)*dxy + min_n
    
    xv, yv = np.meshgrid(grid_e, grid_n)

    return dxy, xv.T, yv.T


root = './'

reader = asciitable.NoHeader()
reader.data.splitter.delimiter = ' '
reader.data.start_line = 0
reader.data.splitter.process_line = None
data = reader.read(root + 'distance.txt')

name_distance = np.array(data['col1']) 
distance      = np.array(data['col2'])

dxy, grid_e, grid_n =  defineGrid(1,1,box_size=5.,res = .01)

loc = np.zeros_like(grid_e)

#first point
A = (0.,0.)
loc[np.where( ( np.abs(grid_e-A[0]) < .5*dxy ) & ( np.abs(grid_n-A[1]) < .5*dxy) ) ] = 1

#second point
B = (distance[np.where(name_distance=='AB')[0]],0.)
loc[np.where( ( np.abs(grid_e-B[0]) < .5*dxy ) & ( np.abs(grid_n-B[1]) < .5*dxy) ) ] = 2

#thrird point
distanceA = []
distanceB = []
for (x,y) in zip(grid_e.flatten(),grid_n.flatten()):
    distanceA.append(np.sqrt( (x-A[0])**2 + (y-A[1])**2 ))
    distanceB.append(np.sqrt( (x-B[0])**2 + (y-B[1])**2 ))
distanceA = np.array(distanceA).reshape(loc.shape)
distanceB = np.array(distanceB).reshape(loc.shape)

distance_CA = distance[np.where(name_distance=='AC')[0]]
distance_CB = distance[np.where(name_distance=='BC')[0]]

sum_dist = np.abs(distanceA-distance_CA) + np.abs(distanceB-distance_CB)

idx = np.where(sum_dist < sum_dist.min() + .1*dxy )

for (x,y) in zip(idx[0],idx[1]):
    if grid_n[x,y] < A[1] :
        continue
    loc[x,y] = 3
C = (grid_e[x,y],grid_n[x,y])


#other point
other_point = ['D','E','F','G']
for i_pt, pt_X in enumerate(other_point):
    distanceA = []
    distanceB = []
    distanceC = []
    for (x,y) in zip(grid_e.flatten(),grid_n.flatten()):
        distanceA.append(np.sqrt( (x-A[0])**2 + (y-A[1])**2 ))
        distanceB.append(np.sqrt( (x-B[0])**2 + (y-B[1])**2 ))
        distanceC.append(np.sqrt( (x-C[0])**2 + (y-C[1])**2 ))
    distanceA = np.array(distanceA).reshape(loc.shape)
    distanceB = np.array(distanceB).reshape(loc.shape)
    distanceC = np.array(distanceC).reshape(loc.shape)

    distance_XA = distance[np.where( (name_distance=='A'+pt_X) | (name_distance==pt_X+'A') )[0]]
    distance_XB = distance[np.where( (name_distance=='B'+pt_X) | (name_distance==pt_X+'B') )[0]]
    distance_XC = distance[np.where( (name_distance=='C'+pt_X) | (name_distance==pt_X+'C') )[0]]

    sum_dist = np.abs(distanceA-distance_XA) + np.abs(distanceB-distance_XB)  + np.abs(distanceC-distance_XC)
    idx = np.where(sum_dist < sum_dist.min() + .1*dxy )

    x,y = idx[0],idx[1]
    loc[x,y] = i_pt+4 

all_point = ['A','B','C'] + other_point
f = open(root + 'point_location.txt','w')
idx = np.where(loc > 0)
loc_arr = []; name_arr = []
for i_pt, (x,y) in enumerate(zip(idx[0],idx[1])):
    loc_arr.append([grid_e[x,y],grid_n[x,y]])
    name_arr.append(all_point[int(loc[x,y])-1])
    line = '{:>3s}  {:>4.1f}  {:>4.1f}\n'.format(all_point[int(loc[x,y])-1],grid_e[x,y],grid_n[x,y])
    f.write(line)
f.close()

fig = plt.figure()
ax = plt.subplot(111)
extent = (grid_e.min()-.5*dxy, grid_e.max()+.5*dxy, grid_n.min()-.5*dxy, grid_n.max()+.5*dxy )
#ax.imshow(np.ma.masked_where(loc==0,loc).T,origin='lower',interpolation='nearest',extent=extent)
for ii, pt in enumerate(loc_arr):
    ax.scatter(pt[0],pt[1],c='k',s=100)
    ax.text(pt[0]+.1,pt[1]+.1,name_arr[ii])
ax.set_xlim(extent[0],extent[1])
ax.set_ylim(extent[2],extent[3])
fig.savefig('map.png')
plt.close(fig)
