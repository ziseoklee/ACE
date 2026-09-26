"""SDF serialization without the four-decimal coordinate rounding of V2000."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def serialize_sdf(molecule: "Mol") -> bytes:
    from rdkit import Chem

    parameters = Chem.MolWriterParams()
    parameters.forceV3000 = True
    parameters.precision = 17
    return (Chem.MolToMolBlock(molecule, parameters) + "$$$$\n").encode("utf-8")
