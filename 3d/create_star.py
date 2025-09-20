import numpy as np

def create_star_obj(filename="3d/star.obj", h=1.5, r_outer=1.5, r_inner=0.5):
    """Creates a 3D star shape and saves it as a watertight .obj file."""

    verts = []
    # Top point
    verts.append([0, 0, h])
    # Bottom point
    verts.append([0, 0, -h])

    # Outer and inner vertices
    for i in range(5):
        # Outer
        angle_outer = i * 2 * np.pi / 5
        verts.append([r_outer * np.cos(angle_outer), r_outer * np.sin(angle_outer), 0])
        # Inner
        angle_inner = (i + 0.5) * 2 * np.pi / 5
        verts.append([r_inner * np.cos(angle_inner), r_inner * np.sin(angle_inner), 0])

    faces = []
    # Top faces
    for i in range(10):
        faces.append([1, i + 3, (i + 1) % 10 + 3])
        
    # Bottom faces
    for i in range(10):
        faces.append([2, (i + 1) % 10 + 3, i + 3])

    with open(filename, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # obj files are 1-indexed
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Generated {filename}")

if __name__ == '__main__':
    create_star_obj()