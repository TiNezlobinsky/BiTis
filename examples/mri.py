# load dicom file
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pydicom.uid as uid



path = Path(
    "/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/2 Organized exports/Zep3/MRI/"
)

files = list(path.glob("*.dcm"))
files = sorted([f.name for f in files])

# filename = "SER_40000_sl0300_ph0001.dcm"


def load_dicom(path, filename):
    dcm = pydicom.dcmread(path / filename, force=True)
    return dcm


dcm_array = []

for filename in files:
    dcm = load_dicom(path, filename)
    dcm.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian

    res = dcm.pixel_array
    dcm_array.append(res)

dcm_array = np.array(dcm_array)

print(dcm_array.shape)

dcm_array = dcm_array[:, :, :260]

import finitewave as fw

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(dcm_array)
mesh_grid = mesh_builder.add_scalar(dcm_array, 'fibrosis')
mesh_grid.plot()
