import numpy as np
import random


def KMeans(data, k=2, max_iters=10, theta=0.1):
    cluster_data = []
    cluster_center = []
    # initialization
    for i in random.sample(range(len(data)), k):
        cluster_center.append(data[i])
    cluster_center = np.array(cluster_center)
    # print('init cluster center:', cluster_center)

    for iter_idx in range(max_iters):
        # assign data to cluster
        cluster_data = []
        for i in range(k):
            cluster_data.append([])
        for data_point in data:
            data_array = np.array(data_point).reshape(1, 2)
            data2center = ((cluster_center - data_array) ** 2).sum(1)
            center_idx = data2center.argmin()
            cluster_data[center_idx].append(data_point)
        # print('cluster_data at iter{}:'.format(iter_idx), cluster_data)

        # update cluster center
        cluster_center_before = cluster_center
        cluster_center = []
        for i in range(k):
            new_center = np.array(cluster_data[i]).mean(0)
            cluster_center.append(new_center.tolist())
        cluster_center = np.array(cluster_center)
        # print('cluster_center at iter{}:'.format(iter_idx), cluster_center)

        # stop criteria
        if ((cluster_center - cluster_center_before) ** 2).sum() < theta:
            break

    return cluster_data


if __name__ == "__main__":
    data = [
        [1,2], [1,3], [2,1], [2,2], [10,10], [9,9]
    ]
    res = KMeans(data, k=5)
    for center_idx in range(len(res)):
        print(res[center_idx])