import os
import bpy

class PhysicalProjectorEffects:
    template_path = os.path.join(os.path.dirname(__file__), "node_templates.blend")
    node_group = "NodeGroup.001"

    namedict = {"defocus": "Defocus_Effect_Node_Group", "focus": "Zfocus", "lendist": "sensor_len_dist",
                "aperture": "aperture", "height": "light_source_height", "width": "light_source_width",
                "z": "depth_texture"}
    labeldict= {"defocus": "Defocus Effect", "focus": "Zfocus (m)", "lendist": "sensor len distance (mm)",
                "aperture": "aperture", "height": "light source height", "width": "light source width",
                "z": "depth texture"}
    
    @staticmethod
    def __load_defocus_node_group():
        with bpy.data.libraries.load(PhysicalProjectorEffects.template_path, link=False) as (data_from, data_to):
            assert(PhysicalProjectorEffects.node_group in data_from.node_groups)
            data_to.node_groups = [PhysicalProjectorEffects.node_group]
        node_group:bpy.types.ShaderNodeTree = data_to.node_groups[0]
        node_group.name = "Defocus Group Node Tree"
        return node_group

    @staticmethod
    def create_defocus_nodes(projector:bpy.types.Object, scene:bpy.types.Scene):
        node_tree : bpy.types.NodeTree = projector.data.node_tree
        # Create Defocus Effect Node Group
        if PhysicalProjectorEffects.namedict['defocus'] in node_tree.nodes:
            group = node_tree.nodes[PhysicalProjectorEffects.namedict['defocus']]
            groupdata:bpy.types.ShaderNodeTree = group.node_tree
            groupdataname = groupdata.name
            node_tree.nodes.remove(group)
            if bpy.data.node_groups[groupdataname].users == 0:
                bpy.data.node_groups.remove(bpy.data.node_groups[groupdataname])
        defocus_node_tree = PhysicalProjectorEffects.__load_defocus_node_group()  
        defocus_node:bpy.types.ShaderNodeGroup = node_tree.nodes.new("ShaderNodeGroup")
        defocus_node.name = PhysicalProjectorEffects.namedict['defocus']
        defocus_node.label = PhysicalProjectorEffects.labeldict['defocus']
        defocus_node.node_tree = defocus_node_tree

        # Create Depth Texture Node
        if PhysicalProjectorEffects.namedict['z'] in node_tree.nodes:
            node_tree.nodes.remove(node_tree.nodes[PhysicalProjectorEffects.namedict['z']])
        depth_node:bpy.types.ShaderNodeTexImage = node_tree.nodes.new("ShaderNodeTexImage")
        depth_node.name = PhysicalProjectorEffects.namedict['z']
        depth_node.label= PhysicalProjectorEffects.labeldict['z']
        
        def create_new_value_node(name, label, value):
            '''
            如果已经存在此节点，就删除然后重新创建。这是为了确保与它相连的links被删除，从而使得接下来创建连接的时候不用考虑重复链接.
            '''
            if name in node_tree.nodes:
                node_tree.nodes.remove(node_tree.nodes[name])
            newnode:bpy.types.ShaderNodeValue = node_tree.nodes.new(type="ShaderNodeValue")
            newnode.name = name
            newnode.label = label
            newnode.outputs[0].default_value = value
            return newnode

        focus_node = create_new_value_node(PhysicalProjectorEffects.namedict['focus'], PhysicalProjectorEffects.labeldict['focus'], projector.phy_proj_settings.focus_z)
        
        lendist_node = create_new_value_node(PhysicalProjectorEffects.namedict['lendist'], PhysicalProjectorEffects.labeldict['lendist'],
                                              projector.phy_proj_settings.light_source_distance if projector.phy_proj_settings.light_source_distance >0 else scene.camera.data.lens)
        aperture_node = create_new_value_node(PhysicalProjectorEffects.namedict['aperture'], PhysicalProjectorEffects.labeldict['aperture'],
                                              projector.phy_proj_settings.F_stop)
        w = projector.phy_proj_settings.light_source_size if projector.phy_proj_settings.light_source_size > 0 else scene.camera.data.sensor_width
        width_node = create_new_value_node(PhysicalProjectorEffects.namedict['width'], PhysicalProjectorEffects.labeldict['width'], w)
        
        reso_x = projector.data.node_tree.nodes["Image Texture"].image.size[0]
        reso_y = projector.data.node_tree.nodes["Image Texture"].image.size[1]
        h = w * reso_y / reso_x
        height_node = create_new_value_node(PhysicalProjectorEffects.namedict['height'], PhysicalProjectorEffects.labeldict['height'], h)

        coord_group_node:bpy.types.ShaderNodeGroup = node_tree.nodes['Group']

        PhysicalProjectorEffects.__setup_fov_info(projector)

        # create links..
        links = node_tree.links
        links.new(coord_group_node.outputs[0], depth_node.inputs[0])   # coordinate group node -> depth texture node (vector)
        links.new(depth_node.outputs[0], defocus_node.inputs[0])      # depth texture node output color(z) -> defocus node "scene Z"
        links.new(coord_group_node.outputs[0], defocus_node.inputs[4]) # coordinate group node -> defocus node "original coordinate"
        links.new(focus_node.outputs[0], defocus_node.inputs[1])       # z-focus node output -> defocus node "focus Z"
        links.new(lendist_node.outputs[0], defocus_node.inputs[2])     # lendist node output -> defocus node "sensor-len distance"
        links.new(aperture_node.outputs[0], defocus_node.inputs[3])    # aperture node -> defocus node "Aperture"
        links.new(width_node.outputs[0], defocus_node.inputs[5])       # width node -> defocus node "light source width (mm)"
        links.new(height_node.outputs[0], defocus_node.inputs[6])      # height node -> defocus node "light source height (mm)"
        links.new(defocus_node.outputs[0], node_tree.nodes["Image Texture"].inputs[0])   # defocus node output "dithered coordinate" -> pattern texture input.
        
        
    @staticmethod
    def remove_defocus_nodes(projector:bpy.types.Object, scene:bpy.types.Scene):
        node_tree:bpy.types.ShaderNodeTree = projector.data.node_tree
        for k, v in PhysicalProjectorEffects.namedict.items():
            if v in node_tree.nodes:
                node_tree.nodes.remove(node_tree.nodes[v])
        node_tree.links.new(node_tree.nodes['Group'].outputs[0], node_tree.nodes['Image Texture'].inputs[0])

    
    @staticmethod
    def setup_depth_texture(projector:bpy.types.Object, texture_path:str):
        depth_node:bpy.types.ShaderNodeTexImage = projector.data.node_tree.nodes[PhysicalProjectorEffects.namedict['z']]
        from pathlib import Path
        path = Path(texture_path)
        fname = path.name
        depth = bpy.data.images.get(fname, None)
        if depth is not None:
            depth.reload()
        else:
            depth = bpy.data.images.load(texture_path, check_existing=True)
        depth_node.image = depth

    @staticmethod
    def __setup_value_nodes(projector:bpy.types.Object, scene:bpy.types.Scene):
        nodes = projector.data.node_tree.nodes

        def __set_default_value(key, value):
            n:bpy.types.ShaderNodeValue = nodes[PhysicalProjectorEffects.namedict[key]]
            n.outputs[0].default_value = value

        __set_default_value('focus', projector.phy_proj_settings.focus_z)
        __set_default_value('lendist', projector.phy_proj_settings.light_source_distance if projector.phy_proj_settings.light_source_distance >0 else scene.camera.data.lens)        
        __set_default_value('aperture', projector.phy_proj_settings.F_stop)
        w = projector.phy_proj_settings.light_source_size if projector.phy_proj_settings.light_source_size > 0 else scene.camera.data.sensor_width
        reso_x = projector.data.node_tree.nodes["Image Texture"].image.size[0]
        reso_y = projector.data.node_tree.nodes["Image Texture"].image.size[1]
        h = w * reso_y / reso_x
        __set_default_value('width', w)
        __set_default_value('height', h)

    @staticmethod
    def __setup_fov_info(projector:bpy.types.Object):
        '''
        此时，lendist和with, height都已经设置好，这里要设置一些fov相关的参数  
        需要在调用相机渲染深度图之前调用.
        '''
        nodes = projector.data.node_tree.nodes
        w = nodes[PhysicalProjectorEffects.namedict['width']].outputs[0].default_value
        h = nodes[PhysicalProjectorEffects.namedict['height']].outputs[0].default_value
        lendist = nodes[PhysicalProjectorEffects.namedict['lendist']].outputs[0].default_value
        scale_x = w / lendist
        scale_y = scale_x / w * h
        nodes["Group"].node_tree.nodes["Mapping.001"].inputs[3].default_value[0] = scale_x
        nodes["Group"].node_tree.nodes["Mapping.001"].inputs[3].default_value[1] = scale_y

    @staticmethod
    def setup_physical_projector_nodes(projector:bpy.types.Object, scene:bpy.types.Scene):
        PhysicalProjectorEffects.__setup_value_nodes(projector, scene)
        PhysicalProjectorEffects.__setup_fov_info(projector)


def bpy_register_physical_proj(dirpath = None):
    if dirpath is not None:
        PhysicalProjectorEffects.template_path = os.path.join(dirpath, "node_templates.blend")

def bpy_unregister_physical_proj():
    return