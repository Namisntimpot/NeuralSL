import os
import bpy

def export(
    context: bpy.types.Context,
    reso_x, reso_y, output_dir_path:str = None, id = 0,
    color_mode = 'RGB',
    image = True, depth = True, normal = True
):
    # 开启必要的功能
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = True

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.image_settings.color_mode = color_mode

    # 必须设置，否则无法输出深度
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"

    # 必须设置，否则无法输出法向
    bpy.context.view_layer.use_pass_normal = True

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # output depth
    if depth:
        depth_file_output : bpy.types.CompositorNodeOutputFile = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    # output normal (scale from (-1, 1) to (0, 1))
    if normal:
        scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        scale_normal.blend_type = 'MULTIPLY'
        scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])
        bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        bias_normal.blend_type = 'ADD'
        bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_normal.outputs[0], bias_normal.inputs[1])
        normal_file_output : bpy.types.CompositorNodeOutputFile = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    # output image.
    if image:
        image_file_output : bpy.types.CompositorNodeOutputFile = tree.nodes.new(type="CompositorNodeOutputFile")
        image_file_output.label = 'Image'
        links.new(render_layers.outputs['Image'], image_file_output.inputs[0])

    # bpy.data.scenes['Scene'] ?
    scene = bpy.context.scene
    # 设置输出分辨率，可以自行修改
    scene.render.resolution_x = reso_x
    scene.render.resolution_y = reso_y

    if output_dir_path is None:
        output_dir_path = scene.render.filepath
    else:
        scene.render.filepath = output_dir_path

    if depth:
        depth_file_output.base_path = output_dir_path
        depth_file_output.file_slots[0].path = 'depth'
    if normal:
        normal_file_output.base_path = output_dir_path
        normal_file_output.file_slots[0].path = 'normal'
    if image:
        image_file_output.base_path = output_dir_path
        image_file_output.file_slots[0].path = 'image'

    bpy.ops.render.render()