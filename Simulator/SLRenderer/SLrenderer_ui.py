import os
import sys
import bpy
import bpy.utils

class SLRENDERER_PT_Setting_Panel(bpy.types.Panel):
    bl_label = "SLrenderer Settings"
    bl_idname = "SLRENDERER_PT_setting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.slrenderer_settings    # 记得需要提前注册！
        
        layout.prop(settings, "resolution_x")
        layout.prop(settings, "resolution_y")
        layout.prop(settings, "export_id")
        layout.prop(settings, "export_img")
        layout.prop(settings, "export_depth")
        layout.prop(settings, "export_normal")
        layout.prop(settings, "specify_pattern_dir")
        layout.prop(settings, "pattern_dir_path")
        layout.prop(settings, "output_dir_path")
        layout.prop(settings, "color_mode")  # 添加枚举属性
        
        layout.operator("slrenderer.export", icon='ADD', text="Export")