import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import random


def dist(x, y):
    # euclidean distance
    if len(x.shape) == 1:
        d = np.sqrt(np.sum((x - y) ** 2))
    else:
        d = np.sqrt(np.sum((x - y) ** 2, axis=1))
    return d


def centers_initiation(points, number_of_centers):
    # initialization of clusters centers as most distant points. return cluster centers (point)
    dist_per_point = np.empty((0, 0), int)
    dist_for_point = 0
    index_of_deleted_point = 0
    for point in points:
        for other_point in np.delete(points, index_of_deleted_point, axis=0):
            dist_for_point += dist(point, other_point)
        dist_per_point = np.append(dist_per_point, dist_for_point)
        dist_for_point = 0
        index_of_deleted_point += 1
    ordered_points_by_min = np.array(
            [key for key, value in sorted(enumerate(dist_per_point), key=lambda p: p[1], reverse=True)])
    return points[ordered_points_by_min[0:number_of_centers]]


def get_cluster_number(points, centers):
    # clustering points. return numbers of clusters for each point
    distances_from_centers = np.zeros((0, centers.shape[0]), int)
    for point in points:
        distances_from_center = np.array([])
        for center in centers:
            distances_from_center = np.concatenate((distances_from_center, [dist(point, center)]))
        distances_from_centers = np.concatenate((distances_from_centers, [distances_from_center]), axis=0)
    nearest_center_number = np.argmin(distances_from_centers, axis=1)
    return nearest_center_number


def kmean(points, centers):
    #classical k-mean. return numbers of clusters for each point
    clusters = get_cluster_number(points, centers)
    clusters_centers_shift = 1
    new_centers = np.zeros(centers.shape)
    counter = 0
    while np.sum(clusters_centers_shift) != 0:
        counter += 1
        for i in xrange(centers.shape[0]):
            new_centers[i] = np.mean(points[:][clusters == i], axis=0)
        clusters_centers_shift = dist(new_centers, centers)
        clusters = get_cluster_number(points, new_centers)
        centers = np.copy(new_centers)
    return clusters


# for yinyang

def global_filter(point_low_bound, point_upper_bound, point_max_noncenter_shift, point_cluster_center_shift):
    #global filtering. retun numbers of passed and filtered points
    numbers_of_nonfiltered_points_bool = point_low_bound - point_max_noncenter_shift >= point_upper_bound + point_cluster_center_shift
    numbers_of_nonfiltered_points = np.nonzero(numbers_of_nonfiltered_points_bool == False)
    numbers_of_filtered_poins = np.nonzero(numbers_of_nonfiltered_points_bool)
    return numbers_of_nonfiltered_points[0], numbers_of_filtered_poins[0]


def point_low_and_upper_bound_calc(points, centers):
    # calculation of low and upper bound for each point. return low and upper bound
    point_upper_bound = np.array([])
    point_low_bound = np.array([])
    for point in points:
        distances_from_centers = np.array([])
        for center in centers:
            distances_from_centers = np.concatenate((distances_from_centers, [dist(point, center)]))
        distances_from_centers = np.sort(distances_from_centers)
        point_upper_bound = np.append(point_upper_bound, distances_from_centers[0])
        point_low_bound = np.append(point_low_bound, distances_from_centers[1])
    return point_low_bound, point_upper_bound


def update_low_and_upper_bound(point_low_bound, point_upper_bound, point_cluster_center_shift,
                               point_max_noncenter_shift):
    # updating low and upper bound for each poind. return low and upper bound
    point_low_bound = point_low_bound - point_max_noncenter_shift
    point_upper_bound = point_upper_bound + point_cluster_center_shift
    return point_low_bound, point_upper_bound


def max_noncentr_shift(clusters_centers_shift, clusters):
    # calculation of second closest cluster center. return numbers of clusters for each point
    max_noncenter_shift = np.array(
            [np.max(np.delete(clusters_centers_shift, i)) for i in xrange(clusters_centers_shift.size)])
    return np.array([max_noncenter_shift[i] for i in clusters])


def glob_yingyang(points, centers):
    # glob_yingyang clustering using just global filtering. return numbers of clusters for each point
    clusters = get_cluster_number(points, centers)
    point_low_bound, point_upper_bound = point_low_and_upper_bound_calc(points, centers)
    clusters_centers_shift = 1
    counter = 0
    new_centers = np.zeros(centers.shape)
    while np.sum(clusters_centers_shift) != 0:
        counter += 1
        for i in xrange(centers.shape[0]):
            new_centers[i] = np.mean(points[:][clusters == i], axis=0)
        clusters_centers_shift = dist(new_centers, centers)
        point_cluster_center_shift = np.array([clusters_centers_shift[i] for i in clusters])
        point_max_noncenter_shift = max_noncentr_shift(clusters_centers_shift, clusters)

        point_after_filter, filtered_points = global_filter(point_low_bound, point_upper_bound,
                                                            point_max_noncenter_shift, point_cluster_center_shift)
        point_low_bound[filtered_points], point_upper_bound[filtered_points] = update_low_and_upper_bound(
                point_low_bound[filtered_points], point_upper_bound[filtered_points],
                point_cluster_center_shift[filtered_points], point_max_noncenter_shift[filtered_points])
        point_low_bound[point_after_filter], point_upper_bound[point_after_filter] = point_low_and_upper_bound_calc(
                points[point_after_filter], new_centers)
        clusters[point_after_filter] = get_cluster_number(points[point_after_filter], new_centers)
        # print 'filtered point ', 100*(points.shape[0]- point_after_filter.shape[0])/points.shape[0], '%'
        centers = np.copy(new_centers)
    return clusters

# For group and local filtering yingyang

def creat_groups(centers, number_of_groups):
#  for clusters centers defining groups as kmean clustering operation (5 iterations, as in article). returne cluster (group) number for each center 
    group_centers = centers[np.array(random.sample(range(centers.shape[0]), number_of_groups))]
    center_cluster = get_cluster_number(centers, group_centers)
#    clusters_centers_shift = 1
    new_group_centers = np.zeros([number_of_groups, 2])
    counter = 0
    while counter != 5:
        counter += 1
        for i in xrange(number_of_groups):
            new_group_centers[i] = np.mean(centers[:][center_cluster == i], axis=0)
#        clusters_centers_shift = dist(new_centers, centers)
        center_cluster = get_cluster_number(centers, new_group_centers)
        # group_centers = np.copy(new_group_centers)
    return center_cluster

def group_max_shift(old_centers,new_centers, center_cluster, number_of_groups):
#  return max  cluster shift per group. exclude center of cluster for witch point belongs
    centers_shift = dist(new_centers,old_centers)
    max_group_shift = np.zeros(number_of_groups)
    for i in xrange(number_of_groups):
        max_group_shift[i] = np.max(centers_shift[center_cluster == i])
    return max_group_shift

def points_low_bound_per_group(points, centers, center_cluster, number_of_groups):
    # return low bounds for each point per group of clusters
    low_bounds = np.zeros([points.shape[0], number_of_groups])
    for i in xrange(number_of_groups):
        for j in xrange(points.shape[0]):
            centers_in_group = centers[center_cluster==i]
            low_bounds[j,i]=np.min(dist(centers_in_group,np.repeat([points[j]],centers_in_group.shape[0],axis=0)))
    return low_bounds


      
def group_local_yingyang(points,centers,number_of_groups):
    clusters = get_cluster_number(points, centers)
    center_cluster = creat_groups(centers, number_of_groups)
    low_bounds = points_low_bound_per_group(points, centers, center_cluster, number_of_groups)
    new_centers = np.zeros(centers.shape)
    point_to_center_dist = dist(points,np.array([centers[i] for i in clusters]))
    for i in xrange(centers.shape[0]):
        new_centers[i] = np.mean(points[clusters == i],axis=0)
    grp_max_shft = group_max_shift(centers, new_centers, center_cluster, number_of_groups)
    center_shift = dist(new_centers,centers)
    point_center_shift = np.array([center_shift[i] for i in clusters])
    passed_groups = np.array([low_bounds[:,i] - grp_max_shft[i] < point_to_center_dist + point_center_shift for i in xrange(number_of_groups)]).transpose()
    passed_groups = np.array([np.nonzero(passed_groups[i]) for i in xrange(points.shape[0])])
    print passed_groups



    
# testing
def tree_dementional_space_example():
    np.random.seed(1024)
    points = np.random.multivariate_normal([0,0,0],[[1,0,0],[0,1,0],[0,0,1]],5000)
    time = np.zeros([2,50])
    for i in xrange(1):
        # centers = centers_initiation(points, 2)
        centers = points[np.array(random.sample(range(points.shape[0]), 2))] # for random center initiation
        print centers
        start_time = timeit.default_timer()
        a = kmean(points,centers)
        print '3d space kmean took ', timeit.default_timer() - start_time, 'sec'
        time[0] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        b = glob_yingyang(points,centers)
        print '3d space glob_yingyang took', timeit.default_timer() - start_time, 'sec'
        time[1] = timeit.default_timer() - start_time
    print np.mean(time,axis=1)
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='3d')
    # color = ['r','g','b','y']
    # for i in a:
    #     ax.scatter(points[a==i,0],points[a==i, 1],points[a==i,2],marker = '.', c = color[i],s=100)
    # fig2 = plt.figure()
    # bx = fig2.add_subplot(111, projection='3d')
    # color = ['r','g','b','y']
    # for i in a:
    #     bx.scatter(points[b==i,0],points[b==i, 1],points[b==i,2],marker = '.', c = color[i],s=100)
    # plt.show()

def two_dementional_space_example():
    np.random.seed(1024)
    points = np.random.normal(0, 1, [100, 2])
    # points = np.append(points, np.random.normal(2, 1, [5000, 2]), axis=0)
    # centers = centers_initiation(points, 2)
    centers = points[np.array(random.sample(range(points.shape[0]), 2))] # for random center initiation
    # p = [x for ]
    start_time = timeit.default_timer()
    a = kmean(points, centers)
    print '3d space  kmean took ', timeit.default_timer() - start_time, 'sec'

    plt.figure()
    for i in xrange(centers.shape[0]):
        plt.plot(points[a == i, 0], points[a == i, 1], '.')

    start_time = timeit.default_timer()
    b = glob_yingyang(points, centers)
    print '3d space glob_yingyang took', timeit.default_timer() - start_time, 'sec'

    plt.figure()
    for i in xrange(centers.shape[0]):
        plt.plot(points[b == i, 0], points[b == i, 1], '.')
    plt.show()
def main():
#    tree_dementional_space_example()
#      two_dementional_space_example()

    points = np.random.normal(0, 1, [10, 2])
    centers = points[np.array(random.sample(range(points.shape[0]), 4))]
    number_of_groups = 2
    group_local_yingyang(points,centers,number_of_groups)

if __name__ == "__main__":
    main()
