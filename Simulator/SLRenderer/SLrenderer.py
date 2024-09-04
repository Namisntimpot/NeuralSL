import bpy
import os
import sys
import argparse
import numpy as np

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
    export_img_noisy: bpy.props.BoolProperty(name="Export Noisy Image", default=False, description="WARNING: Not implemented yet.")
    export_depth: bpy.props.BoolProperty(name="Export Depth", default=True)
    export_normal: bpy.props.BoolProperty(name="Export Normal", default=False)
    export_w_and_b: bpy.props.BoolProperty(name="Export White and Black images", default=False, description="used for capturing ambient and albedo. BLACK shutdown the projector, WHITE projects a white pattern.")
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
                    settings.color_mode, 'OPEN_EXR', "image",
                    False, settings.export_depth, settings.export_normal
                )
            # 再img
            if settings.export_img:
                export(
                    context, settings.resolution_x, settings.resolution_y,
                    settings.output_dir_path, settings.export_id,
                    settings.color_mode, settings.img_format, "image",
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
                    settings.color_mode, 'OPEN_EXR', "image", False, settings.export_depth, settings.export_normal
                )   
            # 再逐帧渲染图片
            if settings.export_img:
                for i in range(n_frames):
                    self.setup_frame(scene, i)
                    tex_path = os.path.join(pattern_dir, pattern_file_lists[i])
                    self.apply_texture_to_projector(settings, scene, projector, tex_path)
                    if i == 0:
                        # 只有第一次需要从头准备节点.
                        export(
                            context, settings.resolution_x, settings.resolution_y,
                            settings.output_dir_path, settings.export_id,
                            settings.color_mode, settings.img_format, "image", settings.export_img, False, False
                        )
                    else:
                        bpy.ops.render.render()      
            # 再渲染BLACK or WHITE
            if settings.export_w_and_b:
                for p in ['WHITE', 'BLACK']:
                    ori = self.apply_texture_to_projector(settings, scene, projector, p)
                    # if p == 'WHITE' and not settings.export_img:
                    if True:
                        export(
                            context, settings.resolution_x, settings.resolution_y,
                            settings.output_dir_path, settings.export_id,
                            settings.color_mode, settings.img_format, p.lower(), True, False, False
                        )
                    else:
                        bpy.ops.render.render()
                    tmp = self.apply_texture_to_projector(settings, scene, projector, ori)
                    bpy.data.images.remove(tmp)
                    
        
        return {'FINISHED'}
    
    def find_projector(self, scene:bpy.types.Scene):
        ind = scene.objects.find(SLRENDERER_OT_Export.projector_identifier)
        if ind == -1:
            return None
        return scene.objects[ind]
    
    def setup_frame(self, scene:bpy.types.Scene, frame_num:int):
        scene.frame_set(frame_num)
        bpy.context.view_layer.update()

    def apply_texture_to_projector(self, settings:SLRendererSettings, scene:bpy.types.Scene, projector:bpy.types.Object, texture_path:str|bpy.types.Image):
        '''
        把texture_path所指定的图片赋值到projector上，注意可能因为文件问题报错！
        '''
        tree: bpy.types.NodeTree = projector.data.node_tree
        texture_node: bpy.types.ShaderNodeTexImage = tree.nodes["Image Texture"]
        ori_pic = texture_node.image
        if not isinstance(texture_path, str):  # 不是字符串，说明它直接是一个texture对象，直接赋值.
            texture_node.image = texture_path
            return ori_pic
        if texture_path == "BLACK":
            black_pic = bpy.data.images.new(name="TempBlackPicture", width=settings.resolution_x, height=settings.resolution_y, alpha=False, float_buffer=True)
            pix = np.zeros((settings.resolution_y, settings.resolution_x, 4), dtype=np.float32)
            pix[:,:,-1] = 1
            black_pic.pixels = pix.flatten().tolist()
            texture_node.image = black_pic
            black_pic.update()
            return ori_pic
        elif texture_path == "WHITE":
            white_pic = bpy.data.images.new(name="TempWhitePicture", width=settings.resolution_x, height=settings.resolution_y, alpha=False, float_buffer=True)
            pix = np.ones((settings.resolution_x * settings.resolution_y * 4), dtype=np.float32)
            white_pic.pixels = pix.tolist()
            texture_node.image = white_pic
            white_pic.update()
            return ori_pic
        else:
            loaded_texture = bpy.data.images.load(texture_path, check_existing=True)
            texture_node.image = loaded_texture
            return ori_pic
        

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--pattern-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    
    args, _ = parser.parse_known_args()
    if args.render:
        if args.pattern_dir != "":
            bpy.context.scene.slrenderer_settings.pattern_dir_path = args.pattern_dir
        if args.output_dir != "":
            bpy.context.scene.slrenderer_settings.output_dir_path = args.output_dir
        bpy.ops.slrenderer.export()
    print("Done")