import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import sys
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
import imageio
import numpy as np
from PIL import Image as im 
from PIL import ImageDraw
from tqdm.auto import tqdm

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
    color2=None
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    if mesh == None:
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
        else:
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
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to GPU (if needed).

        # convert datatype to avoid errors while saving gif
        rend = rend*255
        rend = rend.astype(np.uint8)
        # plt.imsave(savepath + str(angle) + ".jpg", rend)
        res.append(rend)

    imageio.mimsave(save_path + fname, res, fps=fps)

### Q 1.2 Re-creating the Dolly Zoom ###
def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

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


if __name__ == "__main__":
    # Q 1.1
    render_360d(image_size = 1024)

    # Q 1.2
    # dolly_zoom(
    #     image_size=1024,
    #     num_frames=30,
    #     duration=3,
    #     output_file=save_path + "q1_2.gif"
    # )

    # Q 2.1
    # construct_mesh(image_size=1024, angle_step=3, fps=20)

    # Q 2.2
    # vertices_cube = torch.Tensor([
    #     [-0.5, -1, 0],
    #     [-0.5, 1, 0],
    #     [-0.5, 1, 1],
    #     [-0.5, -1, 1],
    #     [0.5, -1, 0],
    #     [0.5, 1, 0],
    #     [0.5, 1, 1],
    #     [0.5, -1, 1]
    #     ])
    # faces_cube = torch.Tensor([
    #     [0, 1, 2],
    #     [2, 3, 0],
    #     [0, 1, 5],
    #     [4, 5, 0],
    #     [1, 2, 5],
    #     [5, 6, 2],
    #     [2, 3, 6],
    #     [6, 7, 3],
    #     [3, 0, 7],
    #     [7, 4, 0],
    #     [4, 5, 6],
    #     [6, 7, 4]
    # ])
    # construct_mesh(vertices = vertices_cube, faces = faces_cube, fname = "q2_2.gif", image_size=1024, angle_step=3, fps=20)

    # Q 3 Re-texturing a mesh
    color1 = [0.2, 0.6, 0.2] # green
    color2 = [0.2, 0.4, 0.9] # blue
    render_360d(fname="q3.gif", image_size = 256, color1=color1, color2=color2)
    
