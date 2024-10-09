import bpy
import glob

import SLRenderer_utils
from SLRenderer_export import export

class SLRENDERER_projector_depth_camera:
    def __init__(self, ctx:bpy.types.Context, projector:bpy.types.Object, collection:bpy.types.Collection = None) -> None:
        '''
        如果是physical projector，请确保在此之前已经处理好了fov相关的参数，即根据projector的len-lightsource distance算好scalex, scaley.
        '''
        self.ctx = ctx
        self.projector = projector
        location = projector.matrix_world.translation
        rotation_euler = projector.matrix_world.to_euler()
        self.camera = SLRenderer_utils.create_camera(ctx.scene, f"{projector.name}_depth_camera", collection, location, rotation_euler)
        self.setup_intrinsic()

    def setup_intrinsic(self):
        proj_data = self.projector.data
        proj_node_tree_group_tree = proj_data.node_tree.nodes["Group"].node_tree
        scale_x = proj_node_tree_group_tree.nodes["Mapping.001"].inputs[3].default_value[0]
        scale_y = proj_node_tree_group_tree.nodes["Mapping.001"].inputs[3].default_value[1]

        self.camera.data.sensor_height = self.camera.data.sensor_width / scale_x * scale_y
        self.camera.data.lens = self.camera.data.sensor_width / scale_x

    def destroy(self):
        if self.camera is not None:
            SLRenderer_utils.delete_camera(self.camera)
            self.camera = None

    def render(self, output_dir_path = None, export_img = False):
        original_camera = self.ctx.scene.camera
        self.ctx.scene.camera = self.camera
        img_tex_node:bpy.types.ShaderNodeTexImage = self.projector.data.node_tree.nodes["Image Texture"]
        reso_x = img_tex_node.image.size[0]
        reso_y = img_tex_node.image.size[1]
        export(self.ctx, reso_x, reso_y, output_dir_path, 0, "BW", 'OPEN_EXR',
               f"image_{self.projector.name}", f"depth_{self.projector.name}", f"normal_{self.projector.name}",
               False, True, False)
        if export_img:
            export(self.ctx, reso_x, reso_y, output_dir_path, 0, 'BW', 'PNG',
                   f"image_{self.projector.name}", f"depth_{self.projector.name}", f"normal_{self.projector.name}",
                   True, False, False)
        self.ctx.scene.camera = original_camera

    
    def __del__(self):
        self.destroy()