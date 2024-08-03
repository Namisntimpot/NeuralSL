import bpy
import os
import sys

# 确保Blender可以找到其他脚本文件
SLRendererDir = "D:\\Lijiaheng\\NeuralSL\\Simulator\\SLRenderer"
sys.path.append(SLRendererDir)

from SLRenderer_export import export
from SLrenderer_ui import SLRENDERER_PT_Setting_Panel

class SLRendererSettings(bpy.types.PropertyGroup):
    resolution_x: bpy.props.IntProperty(name="Resolution X", default=1920)
    resolution_y: bpy.props.IntProperty(name="Resolution Y", default=1080)
    export_id: bpy.props.IntProperty(name="Results ID", default=0)
    export_img: bpy.props.BoolProperty(name="Export Image", default=True)
    export_depth: bpy.props.BoolProperty(name="Export Depth", default=True)
    export_normal: bpy.props.BoolProperty(name="Export Normal", default=False)
    specify_pattern_dir: bpy.props.BoolProperty(name="Specify Pattern Dir", default=False)
    pattern_dir_path: bpy.props.StringProperty(name="Pattern Dir Path", default="//", subtype="DIR_PATH")
    output_dir_path: bpy.props.StringProperty(name="Output Dir Path", default="//", subtype='DIR_PATH')
    img_format: bpy.props.EnumProperty(
        name = "Image Format",
        items=[
            ('PNG', 'PNG', ''),
            ('JPEG','JPEG',''),
            ('OPEN_EXR', 'OpenEXR format. It is a HDR format with 32-bit float data.', '')
        ],
        default='PNG'
    )
    color_mode: bpy.props.EnumProperty(
        name="Color Mode",
        items=[
            ('BW', "Black and White", ""),
            ('RGB', "RGB", ""),
            ('RGBA', "RGBA", ""),
        ],
        default='RGB'
    )

class SLRENDERER_OT_Export(bpy.types.Operator):
    bl_idname = "slrenderer.export"
    bl_label = "Export Rendering Results"
    projector_identifier = "Projector.Spot"

    def execute(self, context):
        scene = context.scene
        settings : SLRendererSettings = scene.slrenderer_settings

        abs_path = convert_path_to_abs_path(settings.output_dir_path)
        if not os.path.isdir(abs_path):
            self.report({'ERROR'}, "Invalid output dir path")
            return {'CANCELLED'}

        if not settings.specify_pattern_dir:    # 没有指定pattern的目录，认为已经在blender中设置好，直接渲染即可.
            # 先depth, normal
            if settings.export_depth or settings.export_normal:
                export(
                    context, settings.resolution_x, settings.resolution_y,
                    settings.output_dir_path, settings.export_id,
                    settings.color_mode, 'OPEN_EXR',
                    False, settings.export_depth, settings.export_normal
                )
            # 再img
            if settings.export_img:
                export(
                    context, settings.resolution_x, settings.resolution_y,
                    settings.output_dir_path, settings.export_id,
                    settings.color_mode, settings.img_format,
                    settings.export_img, False, False
                )

        else:
            pattern_dir = convert_path_to_abs_path(settings.pattern_dir_path)
            try:
                pattern_file_lists = sorted(os.listdir(pattern_dir))
            except Exception as e:
                self.report({'ERROR'}, "Invalid pattern dir path. ERROR: {}".format(e))
                return {"CANCELLED"}
            n_frames = len(pattern_file_lists)
            # 找到projector
            projector = self.find_projector(scene)
            # 最开始只导出一次深度图和normal（如果需要）
            self.setup_frame(scene, 0)
            if settings.export_depth or settings.export_normal:
                export(
                    context, settings.resolution_x, settings.resolution_y,
                    settings.output_dir_path, settings.export_id,
                    settings.color_mode, 'OPEN_EXR', False, settings.export_depth, settings.export_normal
                )   
            # 再逐帧渲染图片
            if settings.export_img:
                for i in range(n_frames):
                    self.setup_frame(scene, i)
                    tex_path = os.path.join(pattern_dir, pattern_file_lists[i])
                    self.apply_texture_to_projector(scene, projector, tex_path)
                    if i == 0:
                        # 只有第一次需要从头准备节点.
                        export(
                            context, settings.resolution_x, settings.resolution_y,
                            settings.output_dir_path, settings.export_id,
                            settings.color_mode, settings.img_format, settings.export_img, False, False
                        )
                    else:
                        bpy.ops.render.render()         
        
        return {'FINISHED'}
    
    def find_projector(self, scene:bpy.types.Scene):
        ind = scene.objects.find(SLRENDERER_OT_Export.projector_identifier)
        if ind == -1:
            return None
        return scene.objects[ind]
    
    def setup_frame(self, scene:bpy.types.Scene, frame_num:int):
        scene.frame_set(frame_num)
        bpy.context.view_layer.update()

    def apply_texture_to_projector(self, scene:bpy.types.Scene, projector:bpy.types.Object, texture_path:str):
        '''
        把texture_path所指定的图片赋值到projector上，注意可能因为文件问题报错！
        '''
        loaded_texture = bpy.data.images.load(texture_path, check_existing=True)
        tree: bpy.types.NodeTree = projector.data.node_tree
        texture_node: bpy.types.ShaderNodeTexImage = tree.nodes["Image Texture"]
        texture_node.image = loaded_texture
        

def convert_path_to_abs_path(p: str):
    return bpy.path.abspath(p) if p.startswith('//') else os.path.abspath(p)


def register():
    bpy.utils.register_class(SLRendererSettings)
    try:
        bpy.utils.register_class(SLRENDERER_PT_Setting_Panel)
    except:
        bpy.utils.unregister_class(SLRENDERER_PT_Setting_Panel)
        bpy.utils.register_class(SLRENDERER_PT_Setting_Panel)
    bpy.utils.register_class(SLRENDERER_OT_Export)
    bpy.types.Scene.slrenderer_settings = bpy.props.PointerProperty(type=SLRendererSettings)

def unregister():
    bpy.utils.unregister_class(SLRENDERER_PT_Setting_Panel)
    bpy.utils.unregister_class(SLRENDERER_OT_Export)
    bpy.utils.unregister_class(SLRendererSettings)

if __name__ == "__main__":
    register()
