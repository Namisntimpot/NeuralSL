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