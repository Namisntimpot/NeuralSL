import os
import bpy

def convert_path_to_abs_path(p: str):
    return bpy.path.abspath(p) if p.startswith('//') else os.path.abspath(p)

def find_projector(scene:bpy.types.Scene):
    ind = scene.objects.find("Projector.Spot")
    if ind == -1:
        return None
    return scene.objects[ind]

def find_all_projector(scene:bpy.types.Scene):
    projs = [o for o in bpy.data.objects if "Projector.Spot" in o.name]
    return projs


def create_camera(scene:bpy.types.Scene, name:str, collection:bpy.types.Collection = None, location = None, rotation_euler = None):
    new_camera_data = bpy.data.cameras.new(f"{name}_data")
    new_camera_object = bpy.data.objects.new(name, new_camera_data)

    if collection is not None:
        collection.objects.link(new_camera_object)
    else:
        scene.collection.objects.link(new_camera_object)

    if location is not None:
        new_camera_object.location = location
    if rotation_euler is not None:
        new_camera_object.rotation_euler = rotation_euler
    return new_camera_object


def delete_camera(camera:bpy.types.Object | str):
    if type(camera) == str:
        camera = bpy.data.objects.get(camera, None)
    if camera is not None:
        bpy.data.objects.remove(camera, do_unlink=True)