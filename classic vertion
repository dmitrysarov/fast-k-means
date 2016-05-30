import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dist(x,y):
    # square of euclidian distance
    return np.sum((x-y)**2)

def centers_initiation(points,number_of_centers):
    # initialization of clusters centers
    dist_per_point = np.empty((0, 0), int)
    dist_for_point = 0
    index_of_deleted_point=0
    for point in points:
        for other_point in np.delete(points,index_of_deleted_point,axis=0):
            dist_for_point += dist(point,other_point)
        dist_per_point = np.append(dist_per_point,dist_for_point)
        dist_for_point = 0
        index_of_deleted_point+=1
    ordered_points_by_min = np.array([key for key,value in sorted(enumerate(dist_per_point),key=lambda p: p[1])])
    return points[ordered_points_by_min[0:number_of_centers]]


def my_kmean(points, number_of_centers):
    centers = centers_initiation(points, number_of_centers)
    list_of_clusters_point = [np.empty((0,points.shape[1])) for i in xrange(number_of_centers)]
    t = 0
    class_shift = 1
    while(class_shift != 0):
        t += 1
        distances_from_centers = np.empty((0, number_of_centers), int)
        centers_old = np.copy(centers)
        for point in points:
            distances_from_center = np.array([])
            for center in centers:
                distances_from_center = np.concatenate((distances_from_center, [dist(point,center)]))
            distances_from_centers = np.concatenate((distances_from_centers, [distances_from_center]), axis=0)
        nearest_center_number = np.argmin(distances_from_centers, axis=1)
        for i in xrange(number_of_centers):
            indecis_of_cluster_point = [x for x,y in enumerate(nearest_center_number) if nearest_center_number[x] == i]
            centers[i] = np.mean(points[np.array(indecis_of_cluster_point)])
        class_shift = dist(centers_old,centers)
    for i in xrange(number_of_centers):
        indecis_of_cluster_point = [x for x,y in enumerate(nearest_center_number) if nearest_center_number[x] == i]
        list_of_clusters_point[i] = np.append(list_of_clusters_point[i],points[np.array(indecis_of_cluster_point)],axis=0)
    return list_of_clusters_point


points = np.random.normal(0, 1,[10,3])
points = np.append(points, np.random.normal(5, 1,[10,3]),axis=0)
a = my_kmean(points,2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = ['r','g','b','y']
t=0
for i in a:
    ax.scatter(i[:,0],i[:, 1],i[:,2],marker = '.', c = color[t],s=100)
    t+=1
plt.show()
