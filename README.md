The article addressed with problem of acceleration of clustering algorithm "k-means". Authors propose improvements for both stages of algorithm: 1) points assignment to cluster and 2) calculation of cluster center shifting. For the first stage the main idea is to use triangular inequality for point filtering, in such a way one can reduce the number of point to center distance calculation. Take into account that fact that only few points change its cluster, authors propose the fast way of cluster center renewal for the second stage. Below I provide a brief modeling result of the algorithm enhancement.
Global filtering
For 2 cluster 2D space advantage of just global filtering starting from 500 points task, e.g. 5000 points 2D space 2 clusters, classic k-mean took 3.8 sec k-mean with just global filtering 0.9 sec if clusters centers were randomly chosen, and 1.8 sec and 0.7 if clusters centers initiated as most distant points 
For 3D space 2 clusters and 5000 points, classic k-mean took 5.98 sec k-mean with just global filtering 1.43 for clusters centers randomly chosen, and 11.7 sec and 2 sec correspondently if clusters centers initiated as most distant points. Obviously such a difference in results explains by more shifts of cluster centers in “most distant” case (comparing with 2D space), but it should be noted that the result are different too:
  
Most distant centers
 
Random centers
Group and local filtering

