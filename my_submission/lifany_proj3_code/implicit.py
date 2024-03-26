import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg=None,
        center=[0,0,0], 
        radius=[1.0],
        c_opt = True,
        r_opt = False
    ):
        super().__init__()

        if cfg != None:
            center = cfg.center.val
            radius = cfg.radius.val

        self.radius = torch.nn.Parameter(
            torch.tensor(center).float(), requires_grad=c_opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=c_opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )
        

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}




# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )


    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.sdfs=[]
        
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.0,1.0,1.0]).to('cuda:0'), radius=torch.tensor(0.001).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.0,-1.0,1.0]).to('cuda:0'), radius=torch.tensor(0.001).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([1.0,-1.0,1.0]).to('cuda:0'), radius=torch.tensor(0.001).to('cuda:0')))

        self.sdfs.append(SphereSDF(center=torch.tensor([1.0,1.0,1.0]).to('cuda:0'), radius=torch.tensor(0.001).to('cuda:0')))


    
    def get_distance(self, points):
        # points = points.view(-1, 3)
        # return self.sdf(points)
        points = points.view(-1, 3)
        dists = []
        for sdf in self.sdfs:
            dists.append(sdf(points))
        dists = torch.cat(dists, dim=1)
        dist,_ = torch.min(dists,dim=1,keepdim=True)
        return dist

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        hidden_dims = [cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir]
        self.layers_xyz_init = torch.nn.Linear(embedding_dim_xyz, hidden_dims[0])

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(cfg.n_layers_xyz):
            if i == 0:
                self.layers_xyz.append(self.layers_xyz_init)
            elif i == 4:
                self.layers_xyz.append(torch.nn.Linear(embedding_dim_xyz+hidden_dims[0], hidden_dims[0]))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_dims[0], hidden_dims[0]))
        self.relu = torch.nn.ReLU()

        self.layer_sigma = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[0], 1),
                torch.nn.ReLU()
            )
        
        self.layer_feature = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[0], hidden_dims[0]),
                torch.nn.ReLU()
            )
        
        self.layers_dir = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim_dir+hidden_dims[0], hidden_dims[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[1], 3),
                torch.nn.Sigmoid()
            )
        
        torch.nn.init.xavier_normal_(self.layers_xyz_init.weight)

    def forward(self, ray_bundle):
        x, h = ray_bundle.sample_points, self.harmonic_embedding_xyz(x)
        for i, layer in enumerate(self.layers_xyz):
            if i == 0:
                x = h
            elif i == 4:
                x = torch.cat((h, x), dim=-1)
            x = layer(x)
            if i != 7:
                x = self.relu(x)

        sigma = self.layer_sigma(x)
        feature = self.layer_feature(x)

        harmonic_embedding_dir = self.harmonic_embedding_dir(ray_bundle.directions).unsqueeze(1)
        harmonic_embedding_dir = torch.tile(harmonic_embedding_dir, (1, feature.shape[1], 1))
        x = torch.cat((harmonic_embedding_dir, feature), dim=-1)
        rgb =  self.layers_dir(x)
        res = {'density': sigma, 'feature': rgb}

        return res


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.n_layers_distance, self.n_layers_color = cfg.n_layers_distance, cfg.n_layers_color
        hidden_dims = [cfg.n_hidden_neurons_distance, cfg.n_hidden_neurons_color]
        self.skip_ind = self.n_layers_distance//2
        self.layers_dist = torch.nn.ModuleList()

        for layeri in range(self.n_layers_distance):
            if layeri == 0:
                self.layers_dist.append(torch.nn.Linear(self.embedding_dim_xyz, hidden_dims[0]))
            elif layeri == self.skip_ind:
                self.layers_dist.append(torch.nn.Linear(self.embedding_dim_xyz+hidden_dims[0], hidden_dims[0]))
            else:
                self.layers_dist.append(torch.nn.Linear(hidden_dims[0], hidden_dims[0]))
        
        self.relu = torch.nn.ReLU()
        self.layer_sigma = torch.nn.Linear(hidden_dims[0], 1)
        
        self.rgb = torch.nn.ModuleList()
        for layeri in range(self.n_layers_color):
            if layeri == 0: 
                self.rgb.append(torch.nn.Linear(3+hidden_dims[0], hidden_dims[1]))
            else: 
                self.rgb.append(torch.nn.Linear(hidden_dims[1], hidden_dims[1]))
            self.rgb.append(torch.nn.ReLU())
        
        self.rgb.append(torch.nn.Linear(hidden_dims[1], 3))
        self.rgb.append(torch.nn.Sigmoid())

       
    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        x = points.view(-1, 3)
        h = self.harmonic_embedding_xyz(x)
        for i, layer in enumerate(self.layers_dist):
            if i == 0:
                x = h
            elif i == self.skip_ind:
                x = torch.cat((x, h), dim=-1)
            x = self.relu(layer(x))
        return self.layer_sigma(x)
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        x = points.view(-1, 3)
        xyz = points.view(-1, 3)
        h = self.harmonic_embedding_xyz(x)
        for i, layer in enumerate(self.layers_dist):
            if i == 0:
                x = h
            elif i == self.skip_ind:
                x = torch.cat((x, h), dim=-1)
            x = self.relu(layer(x))
        x = torch.cat((x, xyz), dim=-1)
        for i, layer in enumerate(self.rgb):
            x = layer(x)

        return x
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        x = points.view(-1, 3)
        xyz = points.view(-1, 3)
        h = self.harmonic_embedding_xyz(x)
        for i, layer in enumerate(self.layers_dist):
            if i == 0:
                x = h
            elif i == self.skip_ind:
                x = torch.cat((x, h), dim=-1)
            x = layer(x)
            x = self.relu(x)
        distance =  self.layer_sigma(x)
        x = torch.cat((x, xyz), dim=-1)
        for i, layer in enumerate(self.rgb):
            x = layer(x)
        points = x

        return distance, points
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}



