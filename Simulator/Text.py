import bpy

for g in bpy.data.node_groups:
    print(g.name)
    if g.users == 0:
        bpy.data.node_groups.remove(g)
        
print(__file__)