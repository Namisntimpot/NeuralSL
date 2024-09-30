import bpy

properties_module_list = []
def register():
    def decoder(cls):
        properties_module_list.append(cls)
        return cls
    return decoder

@register()
class SLRendererPhysicalProjectorSettings(bpy.types.PropertyGroup):  # 它将被绑定在property上...
    light_source_size: bpy.props.FloatProperty(name = "Light Source Size", default=-1, 
                                               description="The diagonal length of the light source in mm. If it is negative, use the same as the main camera")
    light_source_distance: bpy.props.FloatProperty(name="Light Source Distance", default=-1,
                                                   description="The distance between the projector's light source and the lens. If it is negative, use the same as the main camera")
    focus_z: bpy.props.FloatProperty(name="Focus Z", default=1, 
                                     description="focus at... the unit is m.")


@register()
class SLRendererSettings(bpy.types.PropertyGroup):
    resolution_x: bpy.props.IntProperty(name="Resolution X", default=1920)
    resolution_y: bpy.props.IntProperty(name="Resolution Y", default=1080)
    export_id: bpy.props.IntProperty(name="Results ID", default=0)
    export_img: bpy.props.BoolProperty(name="Export Image", default=True)
    export_img_noisy: bpy.props.BoolProperty(name="Export Noisy Image", default=False, description="WARNING: Not implemented yet.")
    export_depth: bpy.props.BoolProperty(name="Export Depth", default=True)
    export_normal: bpy.props.BoolProperty(name="Export Normal", default=False)
    export_w_and_b: bpy.props.BoolProperty(name="Export White and Black images", default=False, description="used for capturing ambient and albedo. BLACK shutdown the projector, WHITE projects a white pattern.")
    export_mask: bpy.props.BoolProperty(name="Export Mask image", default=False, description="export a binarized mask image.")
    hidden_object_list: bpy.props.CollectionProperty(name="Hidden objects list", type=bpy.types.PropertyGroup)
    hidden_object_list_index: bpy.props.IntProperty(name="Hidden object's index")
    
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

    use_physical_projector: bpy.props.BoolProperty(name="Use Physical Projector", description="use physical projector (thin-lens model) or not.", default=False)
    # projector_settings: bpy.props.PointerProperty(
    #     name="Projector Settings",
    #     type=SLRendererPhysicalProjectorSettings
    # )   # 应该把它绑定在具体的projector上.


def bpy_register_properties():
    for cls in properties_module_list:
        try:
            bpy.utils.register_class(cls)
        except:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.slrenderer_settings = bpy.props.PointerProperty(type=SLRendererSettings)


def bpy_unregister_properties():
    for cls in properties_module_list:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.slrenderer_settings