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
def render_cow_360d(
    cow_path="data/cow.obj", 
    image_size=256, 
    color=[0.7, 0.7, 1], 
    device=None, 
    save_path=save_path, 
    fname="q1_1.gif", 
    angle_step=5, # create a view per how many degrees
    fps=15 # how many frames per second
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
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
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


if __name__ == "__main__":
    # Q 1.1
    render_cow_360d(image_size = 1024)

    # Q 1.2
    dolly_zoom(
        image_size=1024,
        num_frames=30,
        duration=3,
        output_file=save_path + "q1_2.gif"
    )

    # Q 2.1

