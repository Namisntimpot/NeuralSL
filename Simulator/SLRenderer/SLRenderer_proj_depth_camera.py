import bpy

from SLRenderer_utils import (create_camera, delete_camera)

class SLRENDERER_projector_depth_camera:
    existed_depth_camera_number = 0

    def __init__(self, ctx:bpy.types.Context, projector:bpy.types.Object, collection:bpy.types.Collection = None) -> None:
        self.projector = projector
        self.camera = create_camera(ctx.scene, f"{projector.name}_depth_camera", collection, projector.location, projector.rotation_euler)
        # TODO: 设置相机参数使之与投影仪内参吻合...
        pass