import logging
import torch
import redner
import pyredner
import time
from typing import List, Union, Tuple, Optional
import warnings

log = logging.getLogger(__name__)

use_correlated_random_number = False
def set_use_correlated_random_number(v: bool):
    """
        | There is a bias-variance trade off in the backward pass.
        | If the forward pass and the backward pass are correlated
        | the gradients are biased for L2 loss.
        | E[d/dx(f(x) - y)^2] = E[(f(x) - y) d/dx f(x)]
        |                     = E[f(x) - y] E[d/dx f(x)]
        | The last equation only holds when f(x) and d/dx f(x) are independent.
        | It is usually better to use the unbiased one, but we left it as an option here
    """
    global use_correlated_random_number
    use_correlated_random_number = v

def get_use_correlated_random_number():
    """
        See set_use_correlated_random_number
    """
    global use_correlated_random_number
    return use_correlated_random_number

print_timing = False
def set_print_timing(v: bool):
    """
        Set whether to print time measurements or not.
    """
    global print_timing
    print_timing = v

def get_print_timing():
    """
        Get whether we print time measurements or not.
    """
    global print_timing
    return print_timing

def serialize_texture(texture, args, device):
    if texture is None:
        args.append(0)
        return
    args.append(len(texture.mipmap))
    for mipmap in texture.mipmap:
        assert(torch.isfinite(mipmap).all())
        assert(mipmap.is_contiguous())
        if mipmap.device != device:
            warnings.warn('Converting texture from {} to {}, this can be inefficient.'.format(mipmap.device, device))
        args.append(mipmap.to(device))
    assert(torch.isfinite(texture.uv_scale).all())
    args.append(texture.uv_scale.to(device))

class Context: pass

class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of C++ redner.
    """

    @staticmethod
    def serialize_scene(scene: pyredner.Scene,
                        num_samples: Union[int, Tuple[int, int]],
                        max_bounces: int,
                        channels: List = [redner.channels.radiance],
                        sampler_type = redner.SamplerType.independent,
                        use_primary_edge_sampling: bool = True,
                        use_secondary_edge_sampling: bool = True,
                        sample_pixel_center: bool = False,
                        device: Optional[torch.device] = None):
        """
            Given a pyredner scene & rendering options, convert them to a linear list of argument,
            so that we can use it in PyTorch.

            Args
            ====
            scene: pyredner.Scene
            num_samples: int
                Number of samples per pixel for forward and backward passes.
                Can be an integer or a tuple of 2 integers.
                If a single integer is provided, use the same number of samples
                for both.
            max_bounces: int
                Number of bounces for global illumination,
                1 means direct lighting only.
            channels: List[redner.channels]
                | A list of channels that should present in the output image
                | following channels are supported\:
                | redner.channels.radiance,
                | redner.channels.alpha,
                | redner.channels.depth,
                | redner.channels.position,
                | redner.channels.geometry_normal,
                | redner.channels.shading_normal,
                | redner.channels.uv,
                | redner.channels.barycentric_coordinates,
                | redner.channels.diffuse_reflectance,
                | redner.channels.specular_reflectance,
                | redner.channels.vertex_color,
                | redner.channels.roughness,
                | redner.channels.generic_texture,
                | redner.channels.shape_id,
                | redner.channels.triangle_id,
                | redner.channels.material_id
                | all channels, except for shape id, triangle id, and material id, are differentiable
            sampler_type: redner.SamplerType
                | Which sampling pattern to use?
                | see `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`
                  for an explanation of the difference between different samplers.
                | Following samplers are supported:
                | redner.SamplerType.independent
                | redner.SamplerType.sobol
            use_primary_edge_sampling: bool

            use_secondary_edge_sampling: bool

            sample_pixel_center: bool
                Always sample at the pixel center when rendering.
                This trades noise with aliasing.
                If this option is activated, the rendering becomes non-differentiable
                (since there is no antialiasing integral),
                and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.

            device: Optional[torch.device]
                Which device should we store the data in.
                If set to None, use the device from pyredner.get_device().
        """
        if device is None:
            device = pyredner.get_device()
        if device.index is None and device.type == 'cuda':
            # Assign a default index so that we can avoid generating warnings
            device = torch.device('cuda:' + str(torch.cuda.current_device()))

        # Record if there is any parameter that requires gradient need discontinuity sampling.
        # For skipping edge sampling when it is not necessary.
        requires_visibility_grad = False
        cam = scene.camera
        num_shapes = len(scene.shapes)
        num_materials = len(scene.materials)
        num_lights = len(scene.area_lights)
        for light_id, light in enumerate(scene.area_lights):
            scene.shapes[light.shape_id].light_id = light_id

        if max_bounces == 0:
            use_secondary_edge_sampling = False

        args = []
        args.append(num_shapes)
        args.append(num_materials)
        args.append(num_lights)
        if not cam.position is None and not torch.isfinite(cam.position).all():
            log.warning("Camera position is not finite !")
        if not cam.look_at is None and not torch.isfinite(cam.look_at).all():
            log.warning("Camera look at is not finite !")
        if not cam.up is None and not torch.isfinite(cam.up).all():
            log.warning("Camera up is not finite !")        
        assert(torch.isfinite(cam.intrinsic_mat_inv).all())
        assert(torch.isfinite(cam.intrinsic_mat).all())
        if cam.position is not None and cam.position.requires_grad:
            requires_visibility_grad = True
        if cam.look_at is not None and cam.look_at.requires_grad:
            requires_visibility_grad = True
        if cam.up is not None and cam.up.requires_grad:
            requires_visibility_grad = True
        args.append(cam.position.cpu() if cam.position is not None else None)
        args.append(cam.look_at.cpu() if cam.look_at is not None else None)
        args.append(cam.up.cpu() if cam.up is not None else None)
        if cam.cam_to_world is not None:
            if cam.cam_to_world.requires_grad:
                requires_visibility_grad = True
            args.append(cam.cam_to_world.cpu().contiguous())
        else:
            args.append(None)
        if cam.world_to_cam is not None:
            if cam.world_to_cam.requires_grad:
                requires_visibility_grad = True
            args.append(cam.world_to_cam.cpu().contiguous())
        else:
            args.append(None)
        if cam.intrinsic_mat.requires_grad or cam.intrinsic_mat_inv.requires_grad:
            requires_visibility_grad = True
        args.append(cam.intrinsic_mat_inv.cpu().contiguous())
        args.append(cam.intrinsic_mat.cpu().contiguous())
        if cam.distortion_params is not None:
            if cam.distortion_params.requires_grad:
                requires_visibility_grad = True
            args.append(cam.distortion_params.cpu().contiguous())
        else:
            args.append(None)
        args.append(cam.clip_near)
        args.append(cam.resolution)
        viewport = cam.viewport
        if viewport is None:
            viewport = (0, 0, cam.resolution[0], cam.resolution[1])
        # Clamp the viewport if necessary
        viewport = (max(viewport[0], 0),
                    max(viewport[1], 0),
                    min(viewport[2], cam.resolution[0]),
                    min(viewport[3], cam.resolution[1]))
        args.append(viewport)
        args.append(cam.camera_type)
        for shape in scene.shapes:
            if not torch.isfinite(shape.vertices).all():
                log.warning("Rendering vertices which are not finite !")
            if (shape.uvs is not None):
                assert(torch.isfinite(shape.uvs).all())
            if (shape.normals is not None):
                assert(torch.isfinite(shape.normals).all())
            if (shape.vertices.requires_grad):
                requires_visibility_grad = True
            if shape.vertices.device != device:
                warnings.warn('Converting shape vertices from {} to {}, this can be inefficient.'.format(shape.vertices.device, device))
            if shape.indices.device != device:
                warnings.warn('Converting shape indices from {} to {}, this can be inefficient.'.format(shape.indices.device, device))
            args.append(shape.vertices.to(device))
            args.append(shape.indices.to(device))
            args.append(shape.uvs.to(device) if shape.uvs is not None else None)
            args.append(shape.normals.to(device) if shape.normals is not None else None)
            args.append(shape.uv_indices.to(device) if shape.uv_indices is not None else None)
            args.append(shape.normal_indices.to(device) if shape.normal_indices is not None else None)
            args.append(shape.colors.to(device) if shape.colors is not None else None)
            args.append(shape.material_id)
            args.append(shape.light_id)
        for material in scene.materials:
            serialize_texture(material.diffuse_reflectance, args, device)
            serialize_texture(material.specular_reflectance, args, device)
            serialize_texture(material.roughness, args, device)
            serialize_texture(material.generic_texture, args, device)
            serialize_texture(material.normal_map, args, device)
            args.append(material.compute_specular_lighting)
            args.append(material.two_sided)
            args.append(material.use_vertex_color)
        for light in scene.area_lights:
            args.append(light.shape_id)
            args.append(light.intensity.cpu())
            args.append(light.two_sided)
            args.append(light.directly_visible)
        if scene.envmap is not None:
            assert(torch.isfinite(scene.envmap.env_to_world).all())
            assert(torch.isfinite(scene.envmap.world_to_env).all())
            assert(torch.isfinite(scene.envmap.sample_cdf_ys).all())
            assert(torch.isfinite(scene.envmap.sample_cdf_xs).all())
            serialize_texture(scene.envmap.values, args, device)
            args.append(scene.envmap.env_to_world.cpu())
            args.append(scene.envmap.world_to_env.cpu())
            args.append(scene.envmap.sample_cdf_ys.to(device))
            args.append(scene.envmap.sample_cdf_xs.to(device))
            args.append(scene.envmap.pdf_norm)
            args.append(scene.envmap.directly_visible)
        else:
            args.append(None)

        args.append(num_samples)
        args.append(max_bounces)
        args.append(channels)
        args.append(sampler_type)
        if requires_visibility_grad:
            args.append(use_primary_edge_sampling)
            args.append(use_secondary_edge_sampling)
        else:
            # Don't need to do edge sampling if we don't require spatial derivatives
            args.append(False)
            args.append(False)
        args.append(sample_pixel_center)
        args.append(device)

        return args

    @staticmethod
    def unpack_args(seed,
                    args,
                    use_primary_edge_sampling = None,
                    use_secondary_edge_sampling = None):
        """
            Given a list of serialized scene arguments, unpack
            all information into a Context.
        """

        current_index = 0
        num_shapes = args[current_index]
        current_index += 1
        num_materials = args[current_index]
        current_index += 1
        num_lights = args[current_index]
        current_index += 1

        cam_position = args[current_index]
        current_index += 1
        cam_look_at = args[current_index]
        current_index += 1
        cam_up = args[current_index]
        current_index += 1
        cam_to_world = args[current_index]
        current_index += 1
        world_to_cam = args[current_index]
        current_index += 1
        intrinsic_mat_inv = args[current_index]
        current_index += 1
        intrinsic_mat = args[current_index]
        current_index += 1
        distortion_params = args[current_index]
        current_index += 1
        clip_near = args[current_index]
        current_index += 1
        resolution = args[current_index]
        current_index += 1
        viewport = args[current_index]
        current_index += 1
        camera_type = args[current_index]
        current_index += 1
        if cam_to_world is None:
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(cam_position.data_ptr()),
                                   redner.float_ptr(cam_look_at.data_ptr()),
                                   redner.float_ptr(cam_up.data_ptr()),
                                   redner.float_ptr(0), # cam_to_world
                                   redner.float_ptr(0), # world_to_cam
                                   redner.float_ptr(intrinsic_mat_inv.data_ptr()),
                                   redner.float_ptr(intrinsic_mat.data_ptr()),
                                   redner.float_ptr(distortion_params.data_ptr() if distortion_params is not None else 0),
                                   clip_near,
                                   camera_type,
                                   redner.Vector2i(viewport[1], viewport[0]),
                                   redner.Vector2i(viewport[3], viewport[2]))
        else:
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(0), # cam_position
                                   redner.float_ptr(0), # cam_look_at
                                   redner.float_ptr(0), # cam_up
                                   redner.float_ptr(cam_to_world.data_ptr()),
                                   redner.float_ptr(world_to_cam.data_ptr()),
                                   redner.float_ptr(intrinsic_mat_inv.data_ptr()),
                                   redner.float_ptr(intrinsic_mat.data_ptr()),
                                   redner.float_ptr(distortion_params.data_ptr() if distortion_params is not None else 0),
                                   clip_near,
                                   camera_type,
                                   redner.Vector2i(viewport[1], viewport[0]),
                                   redner.Vector2i(viewport[3], viewport[2]))
        shapes = []
        for i in range(num_shapes):
            vertices = args[current_index]
            current_index += 1
            indices = args[current_index]
            current_index += 1
            uvs = args[current_index]
            current_index += 1
            normals = args[current_index]
            current_index += 1
            uv_indices = args[current_index]
            current_index += 1
            normal_indices = args[current_index]
            current_index += 1
            colors = args[current_index]
            current_index += 1
            material_id = args[current_index]
            current_index += 1
            light_id = args[current_index]
            current_index += 1
            assert(vertices.is_contiguous())
            assert(indices.is_contiguous())
            if uvs is not None:
                assert(uvs.is_contiguous())
            if normals is not None:
                assert(normals.is_contiguous())
            if uv_indices is not None:
                assert(uv_indices.is_contiguous())
            if normal_indices is not None:
                assert(normal_indices.is_contiguous())
            shapes.append(redner.Shape(\
                redner.float_ptr(vertices.data_ptr()),
                redner.int_ptr(indices.data_ptr()),
                redner.float_ptr(uvs.data_ptr() if uvs is not None else 0),
                redner.float_ptr(normals.data_ptr() if normals is not None else 0),
                redner.int_ptr(uv_indices.data_ptr() if uv_indices is not None else 0),
                redner.int_ptr(normal_indices.data_ptr() if normal_indices is not None else 0),
                redner.float_ptr(colors.data_ptr() if colors is not None else 0),
                int(vertices.shape[0]),
                int(uvs.shape[0]) if uvs is not None else 0,
                int(normals.shape[0]) if normals is not None else 0,
                int(indices.shape[0]),
                material_id,
                light_id))

        materials = []
        for i in range(num_materials):
            num_levels = args[current_index]
            current_index += 1
            diffuse_reflectance = []
            for j in range(num_levels):
                diffuse_reflectance.append(args[current_index])
                current_index += 1
            diffuse_uv_scale = args[current_index]
            current_index += 1
            
            num_levels = args[current_index]
            current_index += 1
            specular_reflectance = []
            for j in range(num_levels):
                specular_reflectance.append(args[current_index])
                current_index += 1
            specular_uv_scale = args[current_index]
            current_index += 1
            
            num_levels = args[current_index]
            current_index += 1
            roughness = []
            for j in range(num_levels):
                roughness.append(args[current_index])
                current_index += 1
            roughness_uv_scale = args[current_index]
            current_index += 1

            num_levels = args[current_index]
            current_index += 1
            generic_texture = []
            if num_levels > 0:
                for j in range(num_levels):
                    generic_texture.append(args[current_index])
                    current_index += 1
                generic_uv_scale = args[current_index]
                current_index += 1
            else:
                generic_uv_scale = None

            num_levels = args[current_index]
            current_index += 1
            normal_map = []
            if num_levels > 0:
                for j in range(num_levels):
                    normal_map.append(args[current_index])
                    current_index += 1
                normal_map_uv_scale = args[current_index]
                current_index += 1
            else:
                normal_map_uv_scale = None

            compute_specular_lighting = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1
            use_vertex_color = args[current_index]
            current_index += 1

            if diffuse_reflectance[0].dim() == 1:
                # Constant texture
                diffuse_reflectance = redner.Texture3(\
                    [redner.float_ptr(diffuse_reflectance[0].data_ptr())],
                    [0], [0], 3,
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))
            else:
                assert(diffuse_reflectance[0].dim() == 3)
                diffuse_reflectance = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in diffuse_reflectance],
                    [x.shape[1] for x in diffuse_reflectance],
                    [x.shape[0] for x in diffuse_reflectance],
                    3,
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))

            if specular_reflectance[0].dim() == 1:
                # Constant texture
                specular_reflectance = redner.Texture3(\
                    [redner.float_ptr(specular_reflectance[0].data_ptr())],
                    [0], [0], 3,
                    redner.float_ptr(specular_uv_scale.data_ptr()))
            else:
                assert(specular_reflectance[0].dim() == 3)
                specular_reflectance = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in specular_reflectance],
                    [x.shape[1] for x in specular_reflectance],
                    [x.shape[0] for x in specular_reflectance],
                    3,
                    redner.float_ptr(specular_uv_scale.data_ptr()))

            if roughness[0].dim() == 1:
                # Constant texture
                roughness = redner.Texture1(\
                    [redner.float_ptr(roughness[0].data_ptr())],
                    [0], [0], 1,
                    redner.float_ptr(roughness_uv_scale.data_ptr()))
            else:
                assert(roughness[0].dim() == 3)
                roughness = redner.Texture1(\
                    [redner.float_ptr(x.data_ptr()) for x in roughness],
                    [x.shape[1] for x in roughness],
                    [x.shape[0] for x in roughness],
                    1,
                    redner.float_ptr(roughness_uv_scale.data_ptr()))

            if len(generic_texture) > 0:
                assert(generic_texture[0].dim() == 3)
                generic_texture = redner.TextureN(\
                    [redner.float_ptr(x.data_ptr()) for x in generic_texture],
                    [x.shape[1] for x in generic_texture],
                    [x.shape[0] for x in generic_texture],
                    generic_texture[0].shape[2],
                    redner.float_ptr(generic_uv_scale.data_ptr()))
            else:
                generic_texture = redner.TextureN(\
                    [], [], [], 0, redner.float_ptr(0))

            if len(normal_map) > 0:
                assert(normal_map[0].dim() == 3)
                normal_map = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in normal_map],
                    [x.shape[1] for x in normal_map],
                    [x.shape[0] for x in normal_map],
                    3,
                    redner.float_ptr(normal_map_uv_scale.data_ptr()))
            else:
                normal_map = redner.Texture3(\
                    [], [], [], 0, redner.float_ptr(0))
            materials.append(redner.Material(\
                diffuse_reflectance,
                specular_reflectance,
                roughness,
                generic_texture,
                normal_map,
                compute_specular_lighting,
                two_sided,
                use_vertex_color))

        area_lights = []
        for i in range(num_lights):
            shape_id = args[current_index]
            current_index += 1
            intensity = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1
            directly_visible = args[current_index]
            current_index += 1

            area_lights.append(redner.AreaLight(\
                shape_id,
                redner.float_ptr(intensity.data_ptr()),
                two_sided,
                directly_visible))

        envmap = None
        if args[current_index] is not None:
            num_levels = args[current_index]
            current_index += 1
            values = []
            for j in range(num_levels):
                values.append(args[current_index])
                current_index += 1
            envmap_uv_scale = args[current_index]
            current_index += 1
            env_to_world = args[current_index]
            current_index += 1
            world_to_env = args[current_index]
            current_index += 1
            sample_cdf_ys = args[current_index]
            current_index += 1
            sample_cdf_xs = args[current_index]
            current_index += 1
            pdf_norm = args[current_index]
            current_index += 1
            directly_visible = args[current_index]
            current_index += 1
            values = redner.Texture3(\
                [redner.float_ptr(x.data_ptr()) for x in values],
                [x.shape[1] for x in values], # width
                [x.shape[0] for x in values], # height
                3, # channels
                redner.float_ptr(envmap_uv_scale.data_ptr()))
            envmap = redner.EnvironmentMap(\
                values,
                redner.float_ptr(env_to_world.data_ptr()),
                redner.float_ptr(world_to_env.data_ptr()),
                redner.float_ptr(sample_cdf_ys.data_ptr()),
                redner.float_ptr(sample_cdf_xs.data_ptr()),
                pdf_norm,
                directly_visible)
        else:
            current_index += 1

        # Options
        num_samples = args[current_index]
        current_index += 1
        max_bounces = args[current_index]
        current_index += 1
        channels = args[current_index]
        current_index += 1
        sampler_type = args[current_index]
        current_index += 1
        use_primary_edge_sampling_ = args[current_index]
        current_index += 1
        use_secondary_edge_sampling_ = args[current_index]
        current_index += 1
        sample_pixel_center = args[current_index]
        current_index += 1
        device = args[current_index]
        current_index += 1

        if use_primary_edge_sampling is None:
            use_primary_edge_sampling = use_primary_edge_sampling_
        if use_secondary_edge_sampling is None:
            use_secondary_edge_sampling = use_secondary_edge_sampling_

        device_index = device.index
        if device.index is None:
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        start = time.time()
        scene = redner.Scene(camera,
                             shapes,
                             materials,
                             area_lights,
                             envmap,
                             device.type == 'cuda',
                             device_index,
                             use_primary_edge_sampling,
                             use_secondary_edge_sampling)
        time_elapsed = time.time() - start
        if get_print_timing():
            print('Scene construction, time: %.5f s' % time_elapsed)

        # check that num_samples is a tuple
        if isinstance(num_samples, int):
            num_samples = (num_samples, num_samples)

        options = redner.RenderOptions(seed,
                                       num_samples[0],
                                       max_bounces,
                                       channels,
                                       sampler_type,
                                       sample_pixel_center)

        ctx = Context()
        ctx.channels = channels
        ctx.options = options
        ctx.resolution = resolution
        ctx.viewport = viewport
        ctx.scene = scene
        ctx.camera = camera
        ctx.shapes = shapes
        ctx.materials = materials
        ctx.area_lights = area_lights
        ctx.envmap = envmap
        ctx.scene = scene
        ctx.options = options
        ctx.num_samples = num_samples
        ctx.device = device

        return ctx

    @staticmethod
    def forward(ctx,
                seed,
                *args):
        """
            Forward rendering pass: given a serialized scene and output an image.
        """

        args_ctx = RenderFunction.unpack_args(seed, args)
        area_lights = args_ctx.area_lights
        camera = args_ctx.camera
        channels = args_ctx.channels
        envmap = args_ctx.envmap
        materials = args_ctx.materials
        num_samples = args_ctx.num_samples
        options = args_ctx.options
        resolution = args_ctx.resolution
        viewport = args_ctx.viewport
        scene = args_ctx.scene
        shapes = args_ctx.shapes
        device = args_ctx.device

        num_channels = redner.compute_num_channels(channels,
                                                   scene.max_generic_texture_dimension)
        img_height = viewport[2] - viewport[0]
        img_width = viewport[3] - viewport[1]
        rendered_image = torch.zeros(img_height, img_width, num_channels, device = device)
        start = time.time()
        redner.render(scene,
                      options,
                      redner.float_ptr(rendered_image.data_ptr()),
                      redner.float_ptr(0), # d_rendered_image
                      None, # d_scene
                      redner.float_ptr(0), # translational_gradient_image
                      redner.float_ptr(0)) # debug_image
        time_elapsed = time.time() - start
        if get_print_timing():
            print('Forward pass, time: %.5f s' % time_elapsed)

        ctx.camera = camera
        ctx.shapes = shapes
        ctx.materials = materials
        ctx.area_lights = area_lights
        ctx.envmap = envmap
        ctx.scene = scene
        ctx.options = options
        ctx.num_samples = num_samples
        ctx.device = device
        ctx.args = args # Important to prevent GC from deallocating the tensors
        return rendered_image

    @staticmethod
    def create_gradient_buffers(ctx):
        scene = ctx.scene
        options = ctx.options
        camera = ctx.camera
        device = ctx.device

        buffers = Context()

        if camera.use_look_at:
            buffers.d_cam_position = torch.zeros(3, device = device)
            buffers.d_cam_look = torch.zeros(3, device = device)
            buffers.d_cam_up = torch.zeros(3, device = device)
            buffers.d_cam_to_world = None
            buffers.d_world_to_cam = None
        else:
            buffers.d_cam_position = None
            buffers.d_cam_look = None
            buffers.d_cam_up = None
            buffers.d_cam_to_world = torch.zeros(4, 4, device = device)
            buffers.d_world_to_cam = torch.zeros(4, 4, device = device)
        buffers.d_intrinsic_mat_inv = torch.zeros(3, 3, device = device)
        buffers.d_intrinsic_mat = torch.zeros(3, 3, device = device)
        buffers.d_distortion_params = None
        if camera.has_distortion_params():
            buffers.d_distortion_params = torch.zeros(8, device = device)
        if camera.use_look_at:
            buffers.d_camera = redner.DCamera(\
                redner.float_ptr(buffers.d_cam_position.data_ptr()),
                redner.float_ptr(buffers.d_cam_look.data_ptr()),
                redner.float_ptr(buffers.d_cam_up.data_ptr()),
                redner.float_ptr(0), # cam_to_world
                redner.float_ptr(0), # world_to_cam
                redner.float_ptr(buffers.d_intrinsic_mat_inv.data_ptr()),
                redner.float_ptr(buffers.d_intrinsic_mat.data_ptr()),
                redner.float_ptr(buffers.d_distortion_params.data_ptr() if buffers.d_distortion_params is not None else 0))
        else:
            buffers.d_camera = redner.DCamera(\
                redner.float_ptr(0), # pos
                redner.float_ptr(0), # look
                redner.float_ptr(0), # up
                redner.float_ptr(buffers.d_cam_to_world.data_ptr()),
                redner.float_ptr(buffers.d_world_to_cam.data_ptr()),
                redner.float_ptr(buffers.d_intrinsic_mat_inv.data_ptr()),
                redner.float_ptr(buffers.d_intrinsic_mat.data_ptr()),
                redner.float_ptr(buffers.d_distortion_params.data_ptr() if buffers.d_distortion_params is not None else 0))
        buffers.d_vertices_list = []
        buffers.d_uvs_list = []
        buffers.d_normals_list = []
        buffers.d_colors_list = []
        buffers.d_shapes = []
        for shape in ctx.shapes:
            num_vertices = shape.num_vertices
            num_uv_vertices = shape.num_uv_vertices
            num_normal_vertices = shape.num_normal_vertices
            d_vertices = torch.zeros(num_vertices, 3, device = device)
            d_uvs = torch.zeros(num_uv_vertices, 2,
                device = device) if shape.has_uvs() else None
            d_normals = torch.zeros(num_normal_vertices, 3,
                device = device) if shape.has_normals() else None
            d_colors = torch.zeros(num_vertices, 3,
                device = device) if shape.has_colors() else None
            buffers.d_vertices_list.append(d_vertices)
            buffers.d_uvs_list.append(d_uvs)
            buffers.d_normals_list.append(d_normals)
            buffers.d_colors_list.append(d_colors)
            buffers.d_shapes.append(redner.DShape(\
                redner.float_ptr(d_vertices.data_ptr()),
                redner.float_ptr(d_uvs.data_ptr() if d_uvs is not None else 0),
                redner.float_ptr(d_normals.data_ptr() if d_normals is not None else 0),
                redner.float_ptr(d_colors.data_ptr() if d_colors is not None else 0)))

        buffers.d_diffuse_list = []
        buffers.d_diffuse_uv_scale_list = []
        buffers.d_specular_list = []
        buffers.d_specular_uv_scale_list = []
        buffers.d_roughness_list = []
        buffers.d_roughness_uv_scale_list = []
        buffers.d_generic_list = []
        buffers.d_generic_uv_scale_list = []
        buffers.d_normal_map_list = []
        buffers.d_normal_map_uv_scale_list = []
        buffers.d_materials = []
        for material in ctx.materials:
            if material.get_diffuse_size(0)[0] == 0:
                d_diffuse = [torch.zeros(3, device = device)]
            else:
                d_diffuse = []
                for l in range(material.get_diffuse_levels()):
                    diffuse_size = material.get_diffuse_size(l)
                    d_diffuse.append(\
                        torch.zeros(diffuse_size[1],
                                    diffuse_size[0],
                                    3, device = device))

            if material.get_specular_size(0)[0] == 0:
                d_specular = [torch.zeros(3, device = device)]
            else:
                d_specular = []
                for l in range(material.get_specular_levels()):
                    specular_size = material.get_specular_size(l)
                    d_specular.append(\
                        torch.zeros(specular_size[1],
                                    specular_size[0],
                                    3, device = device))

            if material.get_roughness_size(0)[0] == 0:
                d_roughness = [torch.zeros(1, device = device)]
            else:
                d_roughness = []
                for l in range(material.get_roughness_levels()):
                    roughness_size = material.get_roughness_size(l)
                    d_roughness.append(\
                        torch.zeros(roughness_size[1],
                                    roughness_size[0],
                                    1, device = device))

            if material.get_generic_levels() == 0:
                d_generic = None
            else:
                d_generic = []
                for l in range(material.get_generic_levels()):
                    generic_size = material.get_generic_size(l)
                    d_generic.append(\
                        torch.zeros(generic_size[2],
                                    generic_size[1],
                                    generic_size[0], device = device))

            if material.get_normal_map_levels() == 0:
                d_normal_map = None
            else:
                d_normal_map = []
                for l in range(material.get_normal_map_levels()):
                    normal_map_size = material.get_normal_map_size(l)
                    d_normal_map.append(\
                        torch.zeros(normal_map_size[1],
                                    normal_map_size[0],
                                    3, device = device))

            buffers.d_diffuse_list.append(d_diffuse)
            buffers.d_specular_list.append(d_specular)
            buffers.d_roughness_list.append(d_roughness)
            buffers.d_generic_list.append(d_generic)
            buffers.d_normal_map_list.append(d_normal_map)
            d_diffuse_uv_scale = torch.zeros(2, device = device)
            d_specular_uv_scale = torch.zeros(2, device = device)
            d_roughness_uv_scale = torch.zeros(2, device = device)
            buffers.d_diffuse_uv_scale_list.append(d_diffuse_uv_scale)
            buffers.d_specular_uv_scale_list.append(d_specular_uv_scale)
            buffers.d_roughness_uv_scale_list.append(d_roughness_uv_scale)
            if d_generic is None:
                d_generic_uv_scale = None
            else:
                d_generic_uv_scale = torch.zeros(2, device = device)
            if d_normal_map is None:
                d_normal_map_uv_scale = None
            else:
                d_normal_map_uv_scale = torch.zeros(2, device = device)

            buffers.d_generic_uv_scale_list.append(d_generic_uv_scale)
            buffers.d_normal_map_uv_scale_list.append(d_normal_map_uv_scale)
            if d_diffuse[0].dim() == 1:
                d_diffuse_tex = redner.Texture3(\
                    [redner.float_ptr(d_diffuse[0].data_ptr())],
                    [0],
                    [0],
                    3,
                    redner.float_ptr(d_diffuse_uv_scale.data_ptr()))
            else:
                d_diffuse_tex = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_diffuse],
                    [x.shape[1] for x in d_diffuse],
                    [x.shape[0] for x in d_diffuse],
                    3,
                    redner.float_ptr(d_diffuse_uv_scale.data_ptr()))

            if d_specular[0].dim() == 1:
                d_specular_tex = redner.Texture3(\
                    [redner.float_ptr(d_specular[0].data_ptr())],
                    [0],
                    [0],
                    3,
                    redner.float_ptr(d_specular_uv_scale.data_ptr()))
            else:
                d_specular_tex = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_specular],
                    [x.shape[1] for x in d_specular],
                    [x.shape[0] for x in d_specular],
                    3,
                    redner.float_ptr(d_specular_uv_scale.data_ptr()))

            if d_roughness[0].dim() == 1:
                d_roughness_tex = redner.Texture1(\
                    [redner.float_ptr(d_roughness[0].data_ptr())],
                    [0],
                    [0],
                    1,
                    redner.float_ptr(d_roughness_uv_scale.data_ptr()))
            else:
                d_roughness_tex = redner.Texture1(\
                    [redner.float_ptr(x.data_ptr()) for x in d_roughness],
                    [x.shape[1] for x in d_roughness],
                    [x.shape[0] for x in d_roughness],
                    1,
                    redner.float_ptr(d_roughness_uv_scale.data_ptr()))

            if d_generic is None:
                d_generic_tex = redner.TextureN(\
                    [], [], [], 0, redner.float_ptr(0))
            else:
                d_generic_tex = redner.TextureN(\
                    [redner.float_ptr(x.data_ptr()) for x in d_generic],
                    [x.shape[1] for x in d_generic],
                    [x.shape[0] for x in d_generic],
                    d_generic[0].shape[2],
                    redner.float_ptr(d_generic_uv_scale.data_ptr()))

            if d_normal_map is None:
                d_normal_map = redner.Texture3(\
                    [], [], [], 0, redner.float_ptr(0))
            else:
                d_normal_map = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_normal_map],
                    [x.shape[1] for x in d_normal_map],
                    [x.shape[0] for x in d_normal_map],
                    3,
                    redner.float_ptr(d_normal_map_uv_scale.data_ptr()))
            buffers.d_materials.append(redner.DMaterial(\
                d_diffuse_tex, d_specular_tex, d_roughness_tex,
                d_generic_tex, d_normal_map))

        buffers.d_intensity_list = []
        buffers.d_area_lights = []
        for light in ctx.area_lights:
            d_intensity = torch.zeros(3, device = device)
            buffers.d_intensity_list.append(d_intensity)
            buffers.d_area_lights.append(\
                redner.DAreaLight(redner.float_ptr(d_intensity.data_ptr())))

        buffers.d_envmap = None
        if ctx.envmap is not None:
            envmap = ctx.envmap
            buffers.d_envmap_values = []
            for l in range(envmap.get_levels()):
                size = envmap.get_size(l)
                buffers.d_envmap_values.append(\
                    torch.zeros(size[1],
                                size[0],
                                3, device = device))
            buffers.d_envmap_uv_scale = torch.zeros(2, device = device)
            d_envmap_tex = redner.Texture3(\
                [redner.float_ptr(x.data_ptr()) for x in buffers.d_envmap_values],
                [x.shape[1] for x in buffers.d_envmap_values],
                [x.shape[0] for x in buffers.d_envmap_values],
                3,
                redner.float_ptr(buffers.d_envmap_uv_scale.data_ptr()))
            buffers.d_world_to_env = torch.zeros(4, 4, device = device)
            buffers.d_envmap = redner.DEnvironmentMap(\
                d_envmap_tex,
                redner.float_ptr(buffers.d_world_to_env.data_ptr()))

        device_index = device.index
        if device.index is None:
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        buffers.d_scene = redner.DScene(buffers.d_camera,
                                        buffers.d_shapes,
                                        buffers.d_materials,
                                        buffers.d_area_lights,
                                        buffers.d_envmap,
                                        device.type == 'cuda',
                                        device_index)
        return buffers

    @staticmethod
    def visualize_screen_gradient(grad_img: torch.Tensor,
                                  seed: int,
                                  scene: pyredner.Scene,
                                  num_samples: Union[int, Tuple[int, int]],
                                  max_bounces: int,
                                  channels: List = [redner.channels.radiance],
                                  sampler_type = redner.SamplerType.independent,
                                  use_primary_edge_sampling: bool = True,
                                  use_secondary_edge_sampling: bool = True,
                                  sample_pixel_center: bool = False):
        """
            Given a serialized scene and output an 2-channel image,
            which visualizes the derivatives of pixel color with respect to 
            the screen space coordinates.

            Args
            ====
            grad_img: Optional[torch.Tensor]
                The "adjoint" of the backpropagation gradient. If you don't know
                what this means just give None
            seed: int
                seed for the Monte Carlo random samplers
            See serialize_scene for the explanation of the rest of the arguments.
        """

        args = RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = num_samples,
            max_bounces = max_bounces,
            sampler_type = sampler_type,
            channels = channels,
            sample_pixel_center = sample_pixel_center)
        args_ctx = RenderFunction.unpack_args(\
            seed, args, use_primary_edge_sampling, use_secondary_edge_sampling)
        channels = args_ctx.channels
        options = args_ctx.options
        resolution = args_ctx.resolution
        viewport = args_ctx.viewport
        scene = args_ctx.scene

        buffers = RenderFunction.create_gradient_buffers(args_ctx)
        num_channels = redner.compute_num_channels(channels,
                                                   scene.max_generic_texture_dimension)
        img_height = viewport[2] - viewport[0]
        img_width = viewport[3] - viewport[1]
        screen_gradient_image = torch.zeros(\
            img_height, img_width, 2, device = device)
        if grad_img is not None:
            assert(grad_img.shape[0] == img_height)
            assert(grad_img.shape[1] == img_width)
            assert(grad_img.shape[2] == num_channels)
        else:
            grad_img = torch.ones(img_height, img_width, num_channels, device = device)
        start = time.time()
        redner.render(scene,
                      options,
                      redner.float_ptr(0), # rendered_image
                      redner.float_ptr(grad_img.data_ptr()), # d_rendered_image
                      buffers.d_scene,
                      redner.float_ptr(screen_gradient_image.data_ptr()),
                      redner.float_ptr(0)) # debug_image
        time_elapsed = time.time() - start
        if get_print_timing():
            print('Visualize gradient, time: %.5f s' % time_elapsed)

        return screen_gradient_image

    @staticmethod
    def backward(ctx,
                 grad_img):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()
        assert(torch.isfinite(grad_img).all())
        scene = ctx.scene
        options = ctx.options
        camera = ctx.camera

        buffers = RenderFunction.create_gradient_buffers(ctx)

        if not get_use_correlated_random_number():
            # Decouple the forward/backward random numbers by adding a big prime number
            options.seed += 1000003

        options.num_samples = ctx.num_samples[1]
        start = time.time()
        redner.render(scene, options,
                      redner.float_ptr(0), # rendered_image
                      redner.float_ptr(grad_img.data_ptr()),
                      buffers.d_scene,
                      redner.float_ptr(0), # translational_gradient_image
                      redner.float_ptr(0)) # debug_image
        time_elapsed = time.time() - start
        if get_print_timing():
            print('Backward pass, time: %.5f s' % time_elapsed)

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_materials
        ret_list.append(None) # num_lights
        if camera.use_look_at:
            ret_list.append(buffers.d_cam_position.cpu())
            ret_list.append(buffers.d_cam_look.cpu())
            ret_list.append(buffers.d_cam_up.cpu())
            ret_list.append(None) # cam_to_world
            ret_list.append(None) # world_to_cam
        else:
            ret_list.append(None) # pos
            ret_list.append(None) # look
            ret_list.append(None) # up
            ret_list.append(buffers.d_cam_to_world.cpu())
            ret_list.append(buffers.d_world_to_cam.cpu())
        ret_list.append(buffers.d_intrinsic_mat_inv.cpu())
        ret_list.append(buffers.d_intrinsic_mat.cpu())
        if not camera.has_distortion_params():
            ret_list.append(None) # distortion_params
        else:
            ret_list.append(buffers.d_distortion_params.cpu())
        ret_list.append(None) # clip near
        ret_list.append(None) # resolution
        ret_list.append(None) # viewport
        ret_list.append(None) # camera_type

        num_shapes = len(ctx.shapes)
        for i in range(num_shapes):
            ret_list.append(buffers.d_vertices_list[i])
            ret_list.append(None) # indices
            ret_list.append(buffers.d_uvs_list[i])
            ret_list.append(buffers.d_normals_list[i])
            ret_list.append(None) # uv_indices
            ret_list.append(None) # normal_indices
            ret_list.append(buffers.d_colors_list[i])
            ret_list.append(None) # material id
            ret_list.append(None) # light id

        num_materials = len(ctx.materials)
        for i in range(num_materials):
            ret_list.append(None) # num_levels
            for d_diffuse in buffers.d_diffuse_list[i]:
                ret_list.append(d_diffuse)
            ret_list.append(buffers.d_diffuse_uv_scale_list[i])
            ret_list.append(None) # num_levels
            for d_specular in buffers.d_specular_list[i]:
                ret_list.append(d_specular)
            ret_list.append(buffers.d_specular_uv_scale_list[i])
            ret_list.append(None) # num_levels
            for d_roughness in buffers.d_roughness_list[i]:
                ret_list.append(d_roughness)
            ret_list.append(buffers.d_roughness_uv_scale_list[i])
            if buffers.d_generic_list[i] is None:
                ret_list.append(None) # num_levels
            else:
                ret_list.append(None) # num_levels
                for d_generic in buffers.d_generic_list[i]:
                    ret_list.append(d_generic)
                ret_list.append(buffers.d_generic_uv_scale_list[i])
            if buffers.d_normal_map_list[i] is None:
                ret_list.append(None) # num_levels
            else:
                ret_list.append(None) # num_levels
                for d_normal_map in buffers.d_normal_map_list[i]:
                    ret_list.append(d_normal_map)
                ret_list.append(buffers.d_normal_map_uv_scale_list[i])
            ret_list.append(None) # compute_specular_lighting
            ret_list.append(None) # two sided
            ret_list.append(None) # use_vertex_color

        num_area_lights = len(ctx.area_lights)
        for i in range(num_area_lights):
            ret_list.append(None) # shape id
            ret_list.append(buffers.d_intensity_list[i].cpu())
            ret_list.append(None) # two_sided
            ret_list.append(None) # directly_visible

        if ctx.envmap is not None:
            ret_list.append(None) # num_levels
            for d_values in buffers.d_envmap_values:
                ret_list.append(d_values)
            ret_list.append(buffers.d_envmap_uv_scale)
            ret_list.append(None) # env_to_world
            ret_list.append(buffers.d_world_to_env.cpu())
            ret_list.append(None) # sample_cdf_ys
            ret_list.append(None) # sample_cdf_xs
            ret_list.append(None) # pdf_norm
            ret_list.append(None) # directly_visible
        else:
            ret_list.append(None)

        ret_list.append(None) # num samples
        ret_list.append(None) # num bounces
        ret_list.append(None) # channels
        ret_list.append(None) # sampler type
        ret_list.append(None) # use_primary_edge_sampling
        ret_list.append(None) # use_secondary_edge_sampling
        ret_list.append(None) # sample_pixel_center
        ret_list.append(None) # device

        return tuple(ret_list)


import pyredner
import random
import redner
import torch
import math
from typing import Union, Tuple, Optional, List

class DeferredLight:
    pass

class AmbientLight(DeferredLight):
    """
        Ambient light for deferred rendering.
    """
    def __init__(self,
                 intensity: torch.Tensor):
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        return self.intensity.to(albedo.device) * albedo

class PointLight(DeferredLight):
    """
        Point light with squared distance falloff for deferred rendering.
    """
    def __init__(self,
                 position: torch.Tensor,
                 intensity: torch.Tensor):
        self.position = position
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        light_dir = self.position.to(position.device) - position
        # the d^2 term:
        light_dist_sq = torch.sum(light_dir * light_dir, dim = -1, keepdim = True)
        light_dist = torch.sqrt(light_dist_sq)
        # Normalize light direction
        light_dir = light_dir / light_dist
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(dot_l_n.device) * dot_l_n * (albedo / math.pi) / light_dist_sq 

class DirectionalLight(DeferredLight):
    """
        Directional light for deferred rendering.
    """
    def __init__(self,
                 direction: torch.Tensor,
                 intensity: torch.Tensor):
        self.direction = direction
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        # Normalize light direction
        light_dir = -self.direction / torch.norm(self.direction)
        light_dir = light_dir.view(1, 1, 3)
        light_dir = light_dir.to(normal.device)
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(dot_l_n.device) * dot_l_n * (albedo / math.pi)

class SpotLight(DeferredLight):
    """
        Spot light with cosine falloff for deferred rendering.
        Note that we do not provide the cosine cutoff here since it is not
        differentiable.
    """
    def __init__(self,
                 position: torch.Tensor,
                 spot_direction: torch.Tensor,
                 spot_exponent: torch.Tensor,
                 intensity: torch.Tensor):
        self.position = position
        self.spot_direction = spot_direction
        self.spot_exponent = spot_exponent
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        light_dir = self.position.to(position.device) - position
        # Normalize light direction
        light_dir = light_dir / torch.norm(light_dir, dim = -1, keepdim = True)
        # Normalize spot direction
        spot_direction = -self.spot_direction / torch.norm(self.spot_direction)
        spot_direction = spot_direction.to(light_dir.device)
        spot_cosine = torch.sum(light_dir * spot_direction, dim = -1, keepdim = True)
        spot_cosine = torch.max(spot_cosine, torch.zeros_like(spot_cosine))
        spot_factor = torch.pow(spot_cosine, self.spot_exponent.to(spot_cosine.device))
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(spot_factor.device) * spot_factor * dot_l_n * (albedo / math.pi)

def render_deferred(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                    lights: Union[List[DeferredLight], List[List[DeferredLight]]],
                    alpha: bool = False,
                    aa_samples: int = 2,
                    seed: Optional[Union[int, List[int]]] = None,
                    sample_pixel_center: bool = False,
                    device: Optional[torch.device] = None):
    """
        Render the scenes using `deferred rendering <https://en.wikipedia.org/wiki/Deferred_shading>`_.
        We generate G-buffer images containing world-space position,
        normal, and albedo using redner, then shade the G-buffer
        using PyTorch code. Assuming Lambertian shading and does not
        compute shadow.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        lights: Union[List[DeferredLight], List[List[DeferredLight]]]
            Lights for deferred rendering. If the scene is a list, and only
            a single list of lights is provided, the same lights are applied
            to all scenes. If a list of lists of lights is provided, each scene
            is lit by the corresponding lights.
        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        aa_samples: int
            Number of samples used for anti-aliasing at both x, y dimensions
            (e.g. if aa_samples=2, 4 samples are used).
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    if device is None:
        device = pyredner.get_device()

    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    if isinstance(scene, pyredner.Scene):
        if seed == None:
            seed = random.randint(0, 16777216)
        # We do full-screen anti-aliasing: increase the rendering resolution
        # and downsample it after lighting
        org_res = scene.camera.resolution
        org_viewport = scene.camera.viewport
        scene.camera.resolution = (org_res[0] * aa_samples,
                                   org_res[1] * aa_samples)
        if org_viewport is not None:
            scene.camera.viewport = [i * aa_samples for i in org_viewport]
        scene_args = RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = (1, 1),
            max_bounces = 0,
            sampler_type = redner.SamplerType.sobol,
            channels = channels,
            use_secondary_edge_sampling = False,
            sample_pixel_center = sample_pixel_center,
            device = device)
        # Need to revert the resolution back
        scene.camera.resolution = org_res
        scene.camera.viewport = org_viewport
        g_buffer = RenderFunction.apply(seed, *scene_args)
        pos = g_buffer[:, :, :3]
        normal = g_buffer[:, :, 3:6]
        albedo = g_buffer[:, :, 6:9]
        img = torch.zeros(g_buffer.shape[0], g_buffer.shape[1], 3, device = device)
        for light in lights:
            img = img + light.render(pos, normal, albedo)
        if alpha:
            # alpha is in the last channel
            img = torch.cat((img, g_buffer[:, :, 9:10]), dim = -1)
        if aa_samples > 1:
            # Downsample
            img = img.permute(2, 0, 1) # HWC -> CHW
            img = img.unsqueeze(0) # CHW -> NCHW
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            img = torch.nn.functional.interpolate(img, size = org_size, mode = 'area')
            img = img.squeeze(dim = 0) # NCHW -> CHW
            img = img.permute(1, 2, 0)
        return img
    else:
        assert(isinstance(scene, list))
        if seed == None:
            # Randomly generate a list of seed
            seed = []
            for i in range(len(scene)):
                seed.append(random.randint(0, 16777216))
        assert(len(seed) == len(scene))
        if len(lights) > 0 and not isinstance(lights[0], list):
            # Specialize version: stack g buffers and light all images in parallel
            g_buffers = []
            # Render each scene in the batch and stack them together
            for sc, se in zip(scene, seed):
                # We do full-screen anti-aliasing: increase the rendering resolution
                # and downsample it after lighting
                org_res = sc.camera.resolution
                org_viewport = sc.camera.viewport
                sc.camera.resolution = (org_res[0] * aa_samples,
                                        org_res[1] * aa_samples)
                if org_viewport is not None:
                    sc.camera.viewport = [i * aa_samples for i in org_viewport]
                scene_args = RenderFunction.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device = device)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffers.append(RenderFunction.apply(se, *scene_args))
            g_buffers = torch.stack(g_buffers)
            pos = g_buffers[:, :, :, :3]
            normal = g_buffers[:, :, :, 3:6]
            albedo = g_buffers[:, :, :, 6:9]
            imgs = torch.zeros(g_buffers.shape[0],
                               g_buffers.shape[1],
                               g_buffers.shape[2],
                               3,
                               device = device)
            for light in lights:
                imgs = imgs + light.render(pos, normal, albedo)
            if alpha:
                imgs = torch.cat((imgs, g_buffers[:, :, :, 9:10]), dim = -1)
        else:
            # If each scene has a different lighting: light them in the loop
            imgs = []
            # Render each scene in the batch and stack them together
            for sc, se, lgts in zip(scene, seed, lights):
                # We do full-screen anti-aliasing: increase the rendering resolution
                # and downsample it after lighting
                org_res = sc.camera.resolution
                org_viewport = sc.camera.viewport
                sc.camera.resolution = (org_res[0] * aa_samples,
                                        org_res[1] * aa_samples)
                if org_viewport is not None:
                    sc.camera.viewport = [i * aa_samples for i in org_viewport]
                scene_args = RenderFunction.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device = device)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffer = RenderFunction.apply(se, *scene_args)
                pos = g_buffer[:, :, :3]
                normal = g_buffer[:, :, 3:6]
                albedo = g_buffer[:, :, 6:9]
                img = torch.zeros(g_buffer.shape[0],
                                  g_buffer.shape[1],
                                  3,
                                  device = device)
                for light in lgts:
                    img = img + light.render(pos, normal, albedo)
                if alpha:
                    # alpha is in the last channel
                    img = torch.cat((img, g_buffer[:, :, 9:10]), dim = -1)
                imgs.append(img)
            imgs = torch.stack(imgs)
        if aa_samples > 1:
            # Downsample
            imgs = imgs.permute(0, 3, 1, 2) # NHWC -> NCHW
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            imgs = torch.nn.functional.interpolate(imgs, size = org_size, mode = 'area')
            imgs = imgs.permute(0, 2, 3, 1) # NCHW -> NHWC
        return imgs

def render_generic(scene: pyredner.Scene,
                   channels: List,
                   max_bounces: int = 1,
                   sampler_type = pyredner.sampler_type.sobol,
                   num_samples: Union[int, Tuple[int, int]] = (4, 4),
                   seed: Optional[int] = None,
                   sample_pixel_center: bool = False,
                   device: Optional[torch.device] = None):
    """
        A generic rendering function that can be either pathtracing or
        g-buffer rendering or both.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.alpha
            | pyredner.channels.depth
            | pyredner.channels.position
            | pyredner.channels.geometry_normal
            | pyredner.channels.shading_normal
            | pyredner.channels.uv
            | pyredner.channels.barycentric_coordinates
            | pyredner.channels.diffuse_reflectance
            | pyredner.channels.specular_reflectance
            | pyredner.channels.roughness
            | pyredner.channels.generic_texture
            | pyredner.channels.vertex_color
            | pyredner.channels.shape_id
            | pyredner.channels.triangle_id
            | pyredner.channels.material_id
        max_bounces: int
            Number of bounces for global illumination, 1 means direct lighting only.
        sampler_type: pyredner.sampler_type
            | Which sampling pattern to use? See 
              `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`_
              for an explanation of the difference between different samplers.
            | Following samplers are supported\:
            | pyredner.sampler_type.independent
            | pyredner.sampler_type.sobol
        num_samples: int
            Number of samples per pixel for forward and backward passes.
            Can be an integer or a tuple of 2 integers.
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
    """
    if device is None:
        device = pyredner.get_device()

    if isinstance(scene, pyredner.Scene):
        if seed==None:
            seed = random.randint(0, 16777216)
        scene_args = RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = num_samples,
            max_bounces = max_bounces,
            sampler_type = sampler_type,
            channels = channels,
            sample_pixel_center = sample_pixel_center,
            device = device)
        return RenderFunction.apply(seed, *scene_args)
    else:
        assert(isinstance(scene, list))
        if seed == None:
            # Randomly generate a list of seed
            seed = []
            for i in range(len(scene)):
                seed.append(random.randint(0, 16777216))
        assert(len(seed) == len(scene))
        # Render each scene in the batch and stack them together
        imgs = []
        for sc, se in zip(scene, seed):
            scene_args = RenderFunction.serialize_scene(\
                scene = sc,
                num_samples = num_samples,
                max_bounces = max_bounces,
                sampler_type = sampler_type,
                channels = channels,
                sample_pixel_center = sample_pixel_center,
                device = device)
            imgs.append(RenderFunction.apply(se, *scene_args))
        imgs = torch.stack(imgs)
        return imgs

def render_g_buffer(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                    channels: List,
                    num_samples: Union[int, Tuple[int, int]] = (1, 1),
                    seed: Optional[Union[int, List[int]]] = None,
                    sample_pixel_center: bool = False,
                    device: Optional[torch.device] = None):
    """
        Render G buffers from the scene.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.alpha
            | pyredner.channels.depth
            | pyredner.channels.position
            | pyredner.channels.geometry_normal
            | pyredner.channels.shading_normal
            | pyredner.channels.uv
            | pyredner.channels.barycentric_coordinates
            | pyredner.channels.diffuse_reflectance
            | pyredner.channels.specular_reflectance
            | pyredner.channels.roughness
            | pyredner.channels.generic_texture
            | pyredner.channels.vertex_color
            | pyredner.channels.shape_id
            | pyredner.channels.triangle_id
            | pyredner.channels.material_id
        num_samples: Union[int, Tuple[int, int]]
            Number of samples for forward and backward passes, respectively.
            If a single integer is provided, use the same number of samples
            for both.
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
    """
    return render_generic(scene = scene,
                          channels = channels,
                          max_bounces = 0,
                          sampler_type = redner.SamplerType.sobol,
                          num_samples = num_samples,
                          seed = seed,
                          sample_pixel_center = sample_pixel_center,
                          device = device)

def render_pathtracing(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                       alpha: bool = False,
                       max_bounces: int = 1,
                       sampler_type = pyredner.sampler_type.sobol,
                       num_samples: Union[int, Tuple[int, int]] = (4, 4),
                       seed: Optional[Union[int, List[int]]] = None,
                       sample_pixel_center: bool = False,
                       device: Optional[torch.device] = None):
    """
        Render a pyredner scene using pathtracing.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        max_bounces: int
            Number of bounces for global illumination, 1 means direct lighting only.
        sampler_type: pyredner.sampler_type
            | Which sampling pattern to use? See 
              `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`_
              for an explanation of the difference between different samplers.
            | Following samplers are supported\:
            | pyredner.sampler_type.independent
            | pyredner.sampler_type.sobol
        num_samples: int
            Number of samples per pixel for forward and backward passes.
            Can be an integer or a tuple of 2 integers.
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    channels = [redner.channels.radiance]
    if alpha:
        channels.append(redner.channels.alpha)
    return render_generic(scene = scene,
                          channels = channels,
                          max_bounces = max_bounces,
                          sampler_type = sampler_type,
                          num_samples = num_samples,
                          seed = seed,
                          sample_pixel_center = sample_pixel_center,
                          device = device)

def render_albedo(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                  alpha: bool = False,
                  num_samples: Union[int, Tuple[int, int]] = (16, 4),
                  seed: Optional[Union[int, List[int]]] = None,
                  sample_pixel_center: bool = False,
                  device: Optional[torch.device] = None):
    """
        Render the diffuse albedo colors of the scenes.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        num_samples: Union[int, Tuple[int, int]]
            number of samples for forward and backward passes, respectively
            if a single integer is provided, use the same number of samples
            for both
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    channels = [redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    return render_g_buffer(scene = scene,
                           channels = channels,
                           num_samples = num_samples,
                           seed = seed,
                           sample_pixel_center = sample_pixel_center,
                           device = device)
