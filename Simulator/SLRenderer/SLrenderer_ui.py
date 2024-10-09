import os
import sys
import bpy
from bpy.types import Context
import bpy.utils

ui_modules_list = []
def register():
    def decoder(cls):
        ui_modules_list.append(cls)
        return cls
    return decoder

def bpy_register_ui_components():
    for module in ui_modules_list:
        try:
            bpy.utils.register_class(module)
        except:
            bpy.utils.unregister_class(module)
            bpy.utils.register_class(module)

def bpy_unregister_ui_components():
    for module in ui_modules_list:
        bpy.utils.unregister_class(module)

@register()
class SLRENDERER_UL_HideObjectList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_index, context_index):
        layout.label(text=item.name, icon='OBJECT_DATA')

@register()
class SLRENDERER_OT_AddHideObject(bpy.types.Operator):
    bl_label = "Add Selected Hidden Object"
    bl_idname = "slrenderer.add_hide"
    def execute(self, context: Context):
        scene = context.scene
        settings = scene.slrenderer_settings
        selected_objects = context.selected_objects

        current_hidden_names = [o.name for o in settings.hidden_object_list]
        for selected_obj in selected_objects:
            if selected_obj.name not in current_hidden_names and selected_obj.type == 'MESH':
                item = settings.hidden_object_list.add()
                item.name = selected_obj.name
        return {'FINISHED'}

@register()
class SLRENDERER_OT_RemoveHideObject(bpy.types.Operator):
    bl_label = "Remove Selected Hidden Object"
    bl_idname = "slrenderer.remove_hide"
    index: bpy.props.IntProperty()

    def execute(self, context: Context):
        settings = context.scene.slrenderer_settings
        settings.hidden_object_list.remove(self.index)
        return {'FINISHED'}
    


@register()
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
        layout.prop(settings, "export_w_and_b")
        layout.prop(settings, "export_depth")
        layout.prop(settings, "export_normal")
        layout.prop(settings, "export_mask")
        self.draw_hidden_obj_selection_panel(layout, scene, settings)

        layout.prop(settings, "use_physical_projector")        

        layout.prop(settings, "specify_pattern_dir")
        layout.prop(settings, "pattern_dir_path")
        layout.prop(settings, "output_dir_path")
        layout.prop(settings, "color_mode")  # 添加枚举属性
        layout.prop(settings, "img_format")
        
        layout.operator("slrenderer.export", icon='ADD', text="Export")

    def draw_hidden_obj_selection_panel(self, layout:bpy.types.UILayout, scene:bpy.types.Scene, settings):
        # 选中后，被选中的index会被设置到settings.hidden_object_list_index...
        layout.template_list("SLRENDERER_UL_HideObjectList", "hidden_object_list", settings, "hidden_object_list", settings, "hidden_object_list_index")
        layout.operator("slrenderer.add_hide", icon="ADD", text="Add")
        layout.operator("slrenderer.remove_hide", icon="REMOVE", text="Remove").index = settings.hidden_object_list_index


@register()
class SLRENDERER_PT_Proj_Setting_Panel(bpy.types.Panel):
    bl_label = "SLrenderer Physical Projector Properties"
    bl_idname = "SLRENDERER_PT_proj_setting_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context):
        layout = self.layout
        obj = context.object

        if "Projector.Spot" in obj.name and context.scene.slrenderer_settings.use_physical_projector:
            layout.prop(obj.phy_proj_settings, "light_source_size")
            layout.prop(obj.phy_proj_settings, "light_source_distance")
            layout.prop(obj.phy_proj_settings, "focus_z")
            layout.prop(obj.phy_proj_settings, "F_stop")
        else:
            layout.label(text="not projectors or physical projector mode off, ignore.")