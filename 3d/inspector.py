
import trimesh
import sys

def inspect_obj(filepath):
    try:
        # Attempt to load the mesh, force loading as a single mesh if it's a scene
        mesh = trimesh.load(filepath, force='mesh')
        
        print(f"--- Inspection Results for: {filepath} ---")
        
        # Watertightness is a good indicator of a clean, enclosed mesh
        print(f"  - Is watertight: {mesh.is_watertight}")
        
        # A watertight mesh should have an Euler number of 2 for a single shell
        print(f"  - Euler number: {mesh.euler_number}")
        
        # Volume calculation can fail or be incorrect for non-watertight meshes
        try:
            print(f"  - Volume: {mesh.volume}")
        except Exception as e:
            print(f"  - Could not compute volume: {e}")
            
        # Check for the presence of normals
        print(f"  - Has face normals: {'face_normals' in mesh.faces.dtype.names}")
        print(f"  - Has vertex normals: {'vertex_normals' in mesh.vertices.dtype.names}")

    except Exception as e:
        print(f"Could not load or inspect {filepath}. Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            inspect_obj(path)
    else:
        print("Please provide paths to .obj files as arguments.")
