import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt, fit=False):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
	# m = torch.nn.Sigmoid()
	# lossclass = torch.nn.BCELoss()
	# loss = lossclass(m(voxel_src), voxel_tgt)
	# implement some loss for binary voxel grids

	voxel_src.unsqueeze(1)
	voxel_tgt.type(dtype=torch.LongTensor)
	if fit: loss = torch.nn.functional.cross_entropy(voxel_src, voxel_tgt)
	else: loss = torch.nn.functional.binary_cross_entropy(voxel_src, voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	
	point_cloud_src_cpy, point_cloud_tgt_cpy = point_cloud_src, point_cloud_tgt
	src = knn_points(point_cloud_src, point_cloud_tgt)
	tgt = knn_points(point_cloud_tgt, point_cloud_src)
	loss_chamfer = torch.mean(torch.sum(src.dists[..., 0].sum(1) + tgt.dists[..., 0].sum(1)))
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian