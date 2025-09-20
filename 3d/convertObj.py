import trimesh
import numpy as np

def convertObjToSdf(obj_file_path, output_npy_path, resolution=64):
    """
    Converts a .obj file to a signed distance function (SDF) and saves it as a .npy file.

    Args:
        obj_file_path (str): The path to the input .obj file.
        output_npy_path (str): The path to save the output .npy file.
        resolution (int): The resolution of the voxel grid along the longest side.
    """
    print(f"Loading mesh from {obj_file_path}...")
    loaded_geom = trimesh.load(obj_file_path)

    if isinstance(loaded_geom, trimesh.Scene):
        # If the loaded object is a scene, merge all meshes into a single mesh
        mesh = trimesh.util.concatenate(loaded_geom.dump())
    else:
        mesh = loaded_geom

    if not mesh.is_watertight:
        print("Warning: The mesh is not watertight. Trying to fix it...")
        mesh.fill_holes()
        if not mesh.is_watertight:
            print("Warning: Could not make the mesh watertight. The SDF might be inaccurate.")

    # calculate pitch based on resolution
    maxDim = np.max(mesh.extents)
    pitch = maxDim / resolution
    
    print(f"Voxelising mesh with a pitch of {pitch} (resolution: {resolution})...")
    bounds = mesh.bounds
    voxelized_mesh = mesh.voxelized(pitch=pitch)
    
    print("Converting voxel grid to a signed distance function...")
    shape = voxelized_mesh.shape
    x, y, z = np.indices(shape)
    indices = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    transform = voxelized_mesh.transform
    points = trimesh.transform_points(indices, transform)
    sdf = trimesh.proximity.signed_distance(mesh, points)
    
    # Reshape the sdf to the voxel grid shape
    sdf = sdf.reshape(shape)

    # Check if the SDF is inverted by checking the centre of the voxel grid
    centreVoxelIndex = tuple(s // 2 for s in shape)
    if sdf[centreVoxelIndex] > 0:
        print("SDF seems to be inverted, flipping the sign.")
        sdf = -sdf

    print(f"Saving SDF to {output_npy_path}...")
    np.save(output_npy_path, {"sdf": sdf, "bounds": bounds, "pitch": pitch})

    print("Conversion complete.")

if __name__ == '__main__':
    # Example usage:
    # convertObjToSdf('my_object.obj', 'my_object_sdf.npy', resolution=128)
    pass
