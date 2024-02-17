import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import sys
import os
import random
import imageio
import numpy as np
from PIL import Image as im 
from PIL import ImageDraw
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms 
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import mcubes
import numpy as np
import pytorch3d
import torch


save_path = "results/"


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def load_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces

### 360-degree Renders ###
def render_360d(
    obj_path="data/cow.obj", 
    mesh = None,
    image_size=256, 
    color=[0.7, 0.7, 1], 
    device=None, 
    save_path=save_path, 
    fname="q1_1.gif", 
    angle_step=5, # create a view per how many degrees
    fps=15, # how many frames per second
    color1=None,
    color2=None,
    textured = False,
    clip = False,
    dist = 3,
    elev = 0
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size, device = device)

    # Get the vertices, faces, and textures.
    if mesh == None and textured == False:
        vertices, faces = load_mesh(obj_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        if color1 is not None and color2 is not None:
            z = vertices[:, :, -1]
            z_min = torch.min(z).item()
            z_max = torch.max(z).item()
            for i in range(len(z[0])):
                alpha = (z[0][i].item() - z_min) / (z_max - z_min)
                color = alpha * torch.tensor(color2) + (1 - alpha) * torch.tensor(color1)
                textures[0][i] = color
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
    mesh = mesh.to(device)

    res = []

    for angle in range(-180, 180, angle_step):
        # Prepare the camera:
        print(angle)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        if clip:
            rend = rend.clip(0, 1)
        # The .cpu moves the tensor to GPU (if needed).

        # convert datatype to avoid errors while saving gif
        rend = rend*255
        rend = rend.astype(np.uint8)
        # plt.imsave(savepath + str(angle) + ".jpg", rend)
        res.append(rend)

    imageio.mimsave(save_path + fname, res, fps=fps, loop=0)
    return res

def visualize_voxels(v, fname = "q1-0-src.gif", image_size = 1024, dist = 3, elev = 0):
    device = get_device()
    myMesh = pytorch3d.ops.cubify(v, thresh=0.5, device=device)

    vertices = myMesh.verts_list()[0]
    faces = myMesh.faces_list()[0]
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    myMesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    render_360d(mesh = myMesh, fname = fname, device=device, image_size=image_size, dist=dist, elev=elev)



if __name__ == "__main__":
    pass