import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt, fit=False):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
	m = torch.nn.Sigmoid()
	lossclass = torch.nn.BCELoss()
	loss = lossclass(m(voxel_src), voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian