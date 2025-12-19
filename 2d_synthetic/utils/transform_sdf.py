import argparse
from rdkit import Chem
import numpy as np

def get_center_of_mass(coordinates):
    """Calculates the center of mass of a set of coordinates."""
    return np.mean(coordinates, axis=0)

def apply_se3_transformation(coordinates, rotation_matrix, translation_vector):
    """Applies rotation and translation to coordinates."""
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)
    transformed_coordinates = rotated_coordinates + translation_vector
    return transformed_coordinates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply SE(3) transformation around the center of mass of an SDF file.")
    parser.add_argument("sdf_file", help="Path to the input SDF file")
    parser.add_argument("--rx", type=float, default=0.0, help="Rotation angle around x-axis in degrees")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotation angle around y-axis in degrees")
    parser.add_argument("--rz", type=float, default=0.0, help="Rotation angle around z-axis in degrees")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation along x-axis")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation along y-axis")
    parser.add_argument("--tz", type=float, default=0.0, help="Translation along z-axis")
    parser.add_argument("--output", default="transformed_fragment.sdf", help="Path to the output SDF file")

    args = parser.parse_args()
    sdf_file = args.sdf_file
    rx_degrees = args.rx
    ry_degrees = args.ry
    rz_degrees = args.rz
    tx = args.tx
    ty = args.ty
    tz = args.tz
    output_file = args.output

    try:
        supplier = Chem.SDMolSupplier(sdf_file)
        mol = supplier[0]
        conformer = mol.GetConformer()
        num_atoms = conformer.GetNumAtoms()
        original_coords = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])

        # Calculate the center of mass
        center_of_mass = get_center_of_mass(original_coords)

        # Translate the fragment so that the center of mass is at the origin
        translated_coords = original_coords - center_of_mass

        # Define rotation matrices around each axis
        rx_radians = np.deg2rad(rx_degrees)
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx_radians), -np.sin(rx_radians)],
            [0, np.sin(rx_radians), np.cos(rx_radians)]
        ])

        ry_radians = np.deg2rad(ry_degrees)
        rotation_y = np.array([
            [np.cos(ry_radians), 0, np.sin(ry_radians)],
            [0, 1, 0],
            [-np.sin(ry_radians), 0, np.cos(ry_radians)]
        ])

        rz_radians = np.deg2rad(rz_degrees)
        rotation_z = np.array([
            [np.cos(rz_radians), -np.sin(rz_radians), 0],
            [np.sin(rz_radians), np.cos(rz_radians), 0],
            [0, 0, 1]
        ])

        # Combine rotations
        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

        # Apply the rotation to the translated coordinates
        rotated_translated_coords = np.dot(translated_coords, rotation_matrix.T)

        # Define the final translation vector
        final_translation_vector = np.array([tx, ty, tz])

        # Translate the rotated coordinates
        new_coords = rotated_translated_coords + center_of_mass + final_translation_vector

        # Update the conformer with the new coordinates
        for i in range(num_atoms):
            conformer.SetAtomPosition(i, new_coords[i])

        # Save the modified molecule to the specified output SDF file
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()

        print(f"Fragment rotated around its center of mass and translated. Saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The SDF file '{sdf_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")