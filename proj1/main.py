import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import sys
import os
import random
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, unproject_depth_image
import imageio
import numpy as np
from PIL import Image as im 
from PIL import ImageDraw
from tqdm.auto import tqdm
from starter.render_generic import *
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

save_path = "results/"

### Q 1.1 360-degree Renders ###
def render_360d(
    cow_path="data/cow.obj", 
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
        vertices, faces = load_cow_mesh(cow_path)
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

### Q 1.2 Re-creating the Dolly Zoom ###
def dolly_zoom(
    obj = "data/cow_on_plane.obj",
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
    mesh = None,
    fov_low = 5,
    fov_high = 120
):
    if device is None:
        device = get_device()
    if mesh is None:
        mesh = pytorch3d.io.load_objs_as_meshes([obj])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(fov_low, fov_high, num_frames)

    renders = []
    for fov in tqdm(fovs):
        distance = 5/(2*np.tan(0.5*np.radians(fov)) ) # TODO: change this.
        T = [[0, 0, distance]]  # TODO: Change this.
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = im.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration))
    return images

### Q 2.1 Constructing a Tetrahedron ###
### Q 2.2 Constructing a Cube ###
def construct_mesh(vertices=None,
                   faces=None,
                   fname="q2_1.gif",
                   device = None,
                   image_size=256,
                   angle_step=5,
                   fps=15):
    if device == None:
        device = get_device()
    if vertices==None and faces==None:
        vertices = torch.Tensor([[-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.4, 0, -0.5], [0, 0, 0.3]])
        faces = torch.Tensor([[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3]])
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor([[0, 0.4, 0.4]])  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    render_360d(mesh=mesh, fname=fname, image_size=image_size, angle_step=angle_step, fps=fps)

### Q 4 Camera Transformations ###
def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
    save_path=save_path, 
    fname="q4_0.jpg", 
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].cpu().numpy()
    plt.imsave(save_path + fname, rend)

    return rend

### Q 5 Rendering Generic 3D Representations ###
### Q 5.1 Rendering Point Clouds from RGB-D Images ###
def generate_pcd(device=None):
    data = load_rgbd_data()
    if device is None:
        device = get_device()

    image1 = torch.Tensor(data["rgb1"])
    mask1 = torch.Tensor(data["mask1"])
    depth1 = torch.Tensor(data["depth1"])
    camera1 = data["cameras1"]
    pts1, rgb1 = unproject_depth_image(image1, mask1, depth1, camera1)

    image2 = torch.Tensor(data["rgb2"])
    mask2 = torch.Tensor(data["mask2"])
    depth2 = torch.Tensor(data["depth2"])
    camera2 = data["cameras2"]
    pts2, rgb2 = unproject_depth_image(image2, mask2, depth2, camera2)

    verts1 = torch.Tensor(pts1).to(device).unsqueeze(0)
    feat1 = torch.Tensor(rgb1).to(device).unsqueeze(0)
    verts2 = torch.Tensor(pts2).to(device).unsqueeze(0)
    feat2 = torch.Tensor(rgb2).to(device).unsqueeze(0)
    verts3 = torch.Tensor(torch.cat((pts1, pts2), 0)).to(device).unsqueeze(0)
    feat3 = torch.Tensor(torch.cat((rgb1, rgb2), 0)).to(device).unsqueeze(0)

    pcd1 = pytorch3d.structures.Pointclouds(points=verts1, features=feat1)
    pcd2 = pytorch3d.structures.Pointclouds(points=verts2, features=feat2)
    pcd3 = pytorch3d.structures.Pointclouds(points=verts3, features=feat3)

    return pcd1, pcd2, pcd3

def visualize_pcd(
        pcd,
        image_size = 256,
        background_color=(1, 1, 1),
        save_path = save_path,
        fname = "q5_1_0.gif",
        device = None,
        fps = 15,
        angle_step = 5,
        dist = 7,
        elev = 0,
        upside_down = True
    ):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
    image_size=image_size, background_color=background_color
    )
    if upside_down:
        angle = torch.Tensor([0, 0, np.pi])
    else:
        angle = torch.Tensor([0, 0, 0])
    r = pytorch3d.transforms.euler_angles_to_matrix(angle, "XYZ")
    res = []
    for angle in range(-180, 180, angle_step):
        # Prepare the camera:
        print(angle)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R @ r, T=T, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(pcd, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to GPU (if needed).

        # convert datatype to avoid errors while saving gif
        rend = rend*255
        rend = rend.astype(np.uint8)
        # plt.imsave(savepath + str(angle) + ".jpg", rend)
        res.append(rend)

    imageio.mimsave(save_path + fname, res, fps=fps)

### Q 5.2 Parametric Functions ###
def torus(
        image_size = 256,
        background_color=(1, 1, 1),
        save_path = save_path,
        fname = "q5_2.gif",
        device = None,
        fps = 15,
        angle_step = 5,
        num_samples = 200,
        a = 1,
        b = 2
):
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = a * (torch.cos(Theta) + b) * torch.cos(Phi)
    y = a * (torch.cos(Theta) + b)* torch.sin(Phi)
    z = a * torch.sin(Theta) 

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    visualize_pcd(
        sphere_point_cloud,
        image_size = image_size,
        background_color = background_color,
        save_path = save_path,
        fname = fname,
        device = device,
        fps = fps,
        angle_step = angle_step,
        dist = 8,
        elev = 0
    )

def custom_pointcloud(image_size = 256,
        background_color=(1, 1, 1),
        save_path = save_path,
        fname = "q5_2_custom.gif",
        device = None,
        fps = 15,
        angle_step = 5,
        num_samples = 200,
        a = 1, 
        b = 2
        ):

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 4 * np.pi, num_samples)
    theta = torch.linspace(0, 4 * np.pi, num_samples)
    R, h = 3, 1
    t = torch.linspace(-3 * np.pi, 3 * np.pi, num_samples)

    # Densely sample phi and theta on a grid
    phi, t = torch.meshgrid(phi, t)

    # x = a * (torch.cos(Theta) + b) * torch.cos(Phi)
    # y = a * (torch.cos(Theta) + b)* torch.sin(Phi)
    # z = a * torch.sin(Theta) 
    x = h * t + (R * a * torch.sin(phi)) / np.sqrt(R ** 2 + h ** 2)
    y = R * torch.cos(t) - a * torch.cos(t) * torch.cos(phi) - (h * a * torch.sin(t) * torch.sin(phi)) / np.sqrt(R ** 2 + h ** 2)
    z = R * torch.sin(t) - a * torch.sin(t) * torch.cos(phi) + (h * a * torch.cos(t) * torch.sin(phi)) / np.sqrt(R ** 2 + h ** 2)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    visualize_pcd(
        sphere_point_cloud,
        image_size = image_size,
        background_color = background_color,
        save_path = save_path,
        fname = fname,
        device = device,
        fps = fps,
        angle_step = angle_step,
        dist = 25,
        elev = 0
    )

### Q 5.3 Implicit Surfaces ###
def render_torus(
        image_size=256, 
        voxel_size=64, 
        device=None,
        save_path = save_path,
        fname = "q5_3.gif",
        fps = 15,
        angle_step = 5,
        a = 1,
        b = 2,
        dist = 10,
        elev = 0
        ):
    if device is None:
        device = get_device()
    min_value = -5
    max_value = 5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    voxels = (b - np.sqrt(X ** 2 + Y ** 2)) ** 2 + Z ** 2 - a ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    plt.imsave("results/5_3.jpg", rend)

    render_360d(mesh=mesh, fname=fname, image_size=image_size, angle_step=angle_step, fps=fps, clip = True, dist = dist, elev = elev)

def render_tori(
        image_size=256, 
        voxel_size=64, 
        device=None,
        save_path = save_path,
        fname = "q5_3_custom.gif",
        fps = 15,
        angle_step = 5,
        a = 1,
        b = 2,
        dist = 10,
        elev = 0
        ):
    if device is None:
        device = get_device()
    min_value = -4
    max_value = 4
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    R = 3
    a = 0.5
    r = 0.01
    voxels = ((X ** 2 + Y ** 2 + Z ** 2 + R ** 2 - a ** 2) ** 2 - 4 * R ** 2 * (Y ** 2 + Z ** 2)) * ((X ** 2 + Y ** 2 + Z ** 2 + R ** 2 - a ** 2) ** 2 - 4 * R ** 2 * (X ** 2 + Z ** 2)) * ((X ** 2 + Y ** 2 + Z ** 2 + R ** 2 - a ** 2) ** 2 - 4 * R ** 2 * (Y ** 2 + X ** 2)) - r
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    plt.imsave("results/5_3_custom.jpg", rend)

    render_360d(mesh=mesh, fname=fname, image_size=image_size, angle_step=angle_step, fps=fps, clip = True, dist = dist, elev = elev)
   
### Q 6 Do Something Fun ###
def fun(
        obj = "custom/tower.obj",
        image_size = 512,
        fname = "q6_all.gif",
        device=None,
        save_path = save_path,
        fps = 15
    ):
    if device is None:
        device = get_device()

    verts, faces, aux = pytorch3d.io.load_obj(
        obj,
        device = device,
        load_textures = True,
        create_texture_atlas = True,
        texture_atlas_size = 4,
        texture_wrap = "repeat"
    )

    texture_map = plt.imread("data/cow_texture.png")

    textures_uv = pytorch3d.renderer.TexturesUV(
    maps=torch.tensor([texture_map]).to(device),
    faces_uvs=faces.textures_idx.unsqueeze(0),
    verts_uvs=aux.verts_uvs.unsqueeze(0),
    ).to(device)

    mesh = pytorch3d.structures.Meshes(
        verts=verts.unsqueeze(0)+torch.tensor([0, -5, 0]).float().to(device),
        faces=faces.verts_idx.unsqueeze(0).to(device),
        textures=textures_uv,
    )

    res1 = render_360d(
    #cow_path="custom/tower.obj",
    mesh = mesh,
    image_size=image_size, 
    color=[0.7, 0.7, 1], 
    save_path=save_path, 
    fname="q6_1.gif", 
    angle_step=5, # create a view per how many degrees
    fps=15, # how many frames per second
    color1=[0.3, 0.3, 0.7],
    color2=[0.4, 0.9, 0.3],
    textured = True,
    dist = 15,
    elev = 5,
    )

    res2 = render_360d(
    cow_path="custom/tower.obj",
    image_size=image_size, 
    color=[0.7, 0.7, 1], 
    save_path=save_path, 
    fname="q6_2.gif", 
    angle_step=5, # create a view per how many degrees
    fps=15, # how many frames per second
    color1=[0.3, 0.3, 0.7],
    color2=[0.4, 0.9, 0.3],
    dist = 15,
    elev = 5,
    )

    res3 = dolly_zoom(
    obj = "custom/tower.obj",
    image_size=image_size,
    num_frames=50,
    duration=3,
    device=None,
    output_file="results/q6_3.gif",
    mesh = mesh,
    fov_low = 1,
    fov_high = 200
    )

    res2.extend(res3)
    res1.extend(res2)
    imageio.mimsave(save_path + fname, res1, fps=fps)

### Q 7 Sampling Points on Meshes ###
def sample(
        obj = "data/cow.obj",
        save_path = save_path,
        fname = "q7.gif",
        image_size = 256,
        device = None,
        sample_size = 400
    ):

    if device is None:
        device = get_device()
    
    verts, faces, aux = pytorch3d.io.load_obj(
        obj,
        device = device,
        load_textures = True,
        create_texture_atlas = True,
        texture_atlas_size = 4,
        texture_wrap = "repeat"
    )
    probs_list = np.zeros(verts.shape[0])
    print(probs_list)

    verts_face = verts[faces.verts_idx]
    S = abs(0.5 * torch.bmm((verts_face[:, 1]-verts_face[:, 0]).unsqueeze(1), (verts_face[:, 1]-verts_face[:, -1]).unsqueeze(2)))
    
    for i in range(0, len(probs_list)):
        probs_list[i] = S[i][0][0]

    normalized_probs = (probs_list-np.min(probs_list))/(np.max(probs_list)-np.min(probs_list)) 

    print(normalized_probs)

    arr = np.arange(verts.shape[0])
    np.random.shuffle(arr)

    total = 0
    selected_idx = []
    for idx in arr:
        random_weight = random.random()
        if normalized_probs[idx] >= random_weight:
            total += 1
            selected_idx.append(idx)
            if total >= sample_size:
                break

    res = verts_face[:, 0][selected_idx] + verts_face[:, 1][selected_idx] + verts_face[:, 2][selected_idx]

    res = res.squeeze(0)
    textures = torch.ones_like(res)

    texture = ((res - res.min()) / (res.max() - res.min()))*torch.Tensor([0, 1, 0]) + ((res.max() - res) / (res.max() - res.min())) * torch.Tensor([0, 0, 1]) + ((res.max() - res) / (res.max() - res.min()))*torch.Tensor([0, 1, 0])
    r = pytorch3d.structures.Pointclouds(points=[res], features = [texture])

    return r

if __name__ == "__main__":
    # Q 1.1
    render_360d(image_size = 1024)

    # Q 1.2
    dolly_zoom(
        image_size=1024,
        num_frames=30,
        duration=3,
        output_file=save_path + "q1_2.gif"
    )

    # Q 2.1
    construct_mesh(image_size=1024, angle_step=3, fps=20)

    # Q 2.2
    vertices_cube = torch.Tensor([
        [-0.5, -1, 0],
        [-0.5, 1, 0],
        [-0.5, 1, 1],
        [-0.5, -1, 1],
        [0.5, -1, 0],
        [0.5, 1, 0],
        [0.5, 1, 1],
        [0.5, -1, 1]
        ])
    faces_cube = torch.Tensor([
        [0, 1, 2],
        [2, 3, 0],
        [0, 1, 5],
        [4, 5, 0],
        [1, 2, 5],
        [5, 6, 2],
        [2, 3, 6],
        [6, 7, 3],
        [3, 0, 7],
        [7, 4, 0],
        [4, 5, 6],
        [6, 7, 4]
    ])
    construct_mesh(vertices = vertices_cube, faces = faces_cube, fname = "q2_2.gif", image_size=1024, angle_step=3, fps=20)

    # Q 3 Re-texturing a mesh
    color1 = [0.2, 0.6, 0.2] # green
    color2 = [0.2, 0.4, 0.9] # blue
    render_360d(fname="q3.gif", image_size = 1024, color1=color1, color2=color2)
    
    # Q 4
    # top-left image
    angle1 = torch.Tensor([0, 0, np.pi/2])
    R1 = pytorch3d.transforms.euler_angles_to_matrix(angle1, "XYZ")
    render_cow(R_relative=R1, fname="q4_1.jpg")

    # top-right image
    T2 = torch.Tensor([0, 0, 2])
    render_cow(T_relative=T2, fname="q4_2.jpg")

    # bottom-left image
    T3 = [0.5, -0.5, -0.05]
    render_cow(T_relative=T3, fname="q4_3.jpg")

    # bottom-right image
    angle4 = torch.Tensor([0, -np.pi/2, 0])
    R4 = pytorch3d.transforms.euler_angles_to_matrix(angle4, "XYZ")
    T4 = [3, 0, 3]
    render_cow(R_relative=R4, T_relative=T4, fname="q4_4.jpg")
    
    # Q 5.1
    pcd1, pcd2, pcd3 = generate_pcd()
    visualize_pcd(pcd1, image_size = 1024, fname="q5_1_1.gif")
    visualize_pcd(pcd2, image_size = 1024, fname="q5_1_2.gif")
    visualize_pcd(pcd3, image_size = 1024, fname="q5_1_3.gif")

    # Q 5.2
    torus(image_size = 1024, num_samples=800, device="cpu")
    custom_pointcloud(image_size = 1024, num_samples = 600, angle_step=5)

    # Q 5.3
    render_torus(image_size = 1024)   
    render_tori(image_size = 1024) 

    # Q 6
    fun(image_size=1024)

    # Q 7
    sample_pc = sample(
        sample_size=10
    )
    visualize_pcd(
        sample_pc,
        image_size = 1024,
        background_color = [0, 0, 0],
        save_path = save_path,
        fname = "q7_10.gif",
        device = None,
        fps = 15,
        angle_step = 15,
        dist = 25,
        elev = 0,
        upside_down = False
    )

    sample_pc = sample(
    sample_size=100
    )
    visualize_pcd(
        sample_pc,
        image_size = 1024,
        background_color = [0, 0, 0],
        save_path = save_path,
        fname = "q7_100.gif",
        device = None,
        fps = 15,
        angle_step = 15,
        dist = 25,
        elev = 0,
        upside_down = False
    )

    sample_pc = sample(
    sample_size=1000
    )
    visualize_pcd(
        sample_pc,
        image_size = 1024,
        background_color = [0, 0, 0],
        save_path = save_path,
        fname = "q7_1000.gif",
        device = None,
        fps = 15,
        angle_step = 15,
        dist = 25,
        elev = 0,
        upside_down = False
    )

    sample_pc = sample(
    sample_size=10000
    )
    visualize_pcd(
        sample_pc,
        image_size = 1024,
        background_color = [0, 0, 0],
        save_path = save_path,
        fname = "q7_10000.gif",
        device = None,
        fps = 15,
        angle_step = 15,
        dist = 25,
        elev = 0,
        upside_down = False
    )

    render_360d(fname = "q7_original.gif", fps = 15)
    
    # make gifs loop forever
    gifs = os.listdir("results")
    print(gifs)
    for item in gifs:
        print(item)
        if ".gif" in item:
            g = Image.open("results/" + item)
            g.save("results_loop/" + item, save_all=True, loop=0)

    gifs = os.listdir("results")
    print(gifs)
    for item in gifs:
        print(item)
        if "q6" in item:
            g = Image.open("results/" + item)
            g.save("results_loop/" + item, save_all=True, loop=0)

    print("done!")
