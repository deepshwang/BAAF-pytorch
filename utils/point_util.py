import torch
import pdb


def knn(db, q, k):
    """
    Args:
        q [Query point]: (B, N_q, C)
        db [candiates of neighbors]: (B, N_db, C)
        k [k-neighbors]: int

    Return:
        idx [k-neighboring indexes]: (B, N, k)
    """
    pairwise_distance = torch.cdist(q, db)
    idx = pairwise_distance.topk(k=k, dim=2)[1]

    return idx



def farthest_point_sample(xyz, npoint):
    """
    Args:
        xyz [point]: (B, N, 3)
        npoint [number of points]: int 
    
    Return:
        centroids [sampled pointcloud index]: (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index2Points(points, idx):
    """
    Args:
        points [point]: (B, N, C)
        idx [sample index data]: (B, S)
    
    Return:
        new_points [indexed points data]: (B, S, C)

	Example:
	    fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)

    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index2kNNPoints(points, knn_idx):
    """
    Args:
        points [point]: (B, N, C)
        knn_idx [k-Nearest Neighbor Index]: (B, N, k)

    Return:
    	knn_points [k-Nearest Neighboring points retrieved]: (B, N, k, C)

   	Example:
   		knn_idx = knn(points, k)
   		knn_points = index2kNNPoints(points, knn_idx)
    """

    _, _, C = points.shape
    B, N, k = knn_idx.shape

    knn_idx = knn_idx.reshape(B, -1) # (B, N*k)
    knn_points = index2Points(points, knn_idx) # (B, N*k, C)
    knn_points = knn_points.reshape(B, N, k, C)

    return knn_points


def nearest_interpolation(feature, interp_idx):
    """
    Args:
        feature [input feature]: (B, N, d)
        interp_idx [Nearest neighbor index]: (B, N_up, 1)

    Return:
        interpolated_features [Interpolated Features]: (B, N_up, d)
    """
    B, N_up, _ = interp_idx.shape
    interp_idx = torch.squeeze(interp_idx)
    interpolated_features = index2Points(feature, interp_idx)
    return interpolated_features
