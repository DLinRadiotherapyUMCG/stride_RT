import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.draw import polygon as sk_polygon
import scipy.ndimage as ndi

# =========================
# LOAD CT
# =========================

def load_ct_series(ct_dir):
    files = sorted(Path(ct_dir).glob("**/*.dcm"))
    slices = [pydicom.dcmread(str(f)) for f in files if hasattr(pydicom.dcmread(str(f)), "ImagePositionPatient")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        arrays.append(arr * slope + intercept)

    ct = np.stack(arrays, axis=0)

    xy_spacing = list(map(float, slices[0].PixelSpacing))
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = abs(z_positions[1] - z_positions[0])

    meta = {
        "origin": list(map(float, slices[0].ImagePositionPatient)),
        "spacing": [xy_spacing[1], xy_spacing[0], dz],
        "z_positions": z_positions,
        "shape": ct.shape
    }

    return ct, meta


# =========================
# LOAD RTDOSE (resample to CT grid)
# =========================

def load_rtdose(rtdose_path, ct_meta):
    ds = pydicom.dcmread(rtdose_path)
    dose = ds.pixel_array.astype(np.float32) * ds.DoseGridScaling

    dose_origin = list(map(float, ds.ImagePositionPatient))
    dose_spacing = list(map(float, ds.PixelSpacing))

    dose_z = [dose_origin[2] + z for z in ds.GridFrameOffsetVector]

    ct_shape = ct_meta["shape"]
    ct_origin = ct_meta["origin"]
    ct_spacing = ct_meta["spacing"]
    ct_z = np.array(ct_meta["z_positions"])

    ct_x = ct_origin[0] + np.arange(ct_shape[2]) * ct_spacing[0]
    ct_y = ct_origin[1] + np.arange(ct_shape[1]) * ct_spacing[1]

    dose_x = dose_origin[0] + np.arange(dose.shape[2]) * dose_spacing[1]
    dose_y = dose_origin[1] + np.arange(dose.shape[1]) * dose_spacing[0]

    def nearest(arr, vals):
        arr = np.array(arr)
        idx = np.searchsorted(arr, vals)
        idx = np.clip(idx, 0, len(arr)-1)
        return idx

    ix = nearest(dose_x, ct_x)
    iy = nearest(dose_y, ct_y)
    iz = nearest(dose_z, ct_z)

    dose_ct = dose[np.ix_(iz, iy, ix)]
    return dose_ct


# =========================
# LOAD RTSTRUCT
# =========================

def load_rtstruct(path):
    return pydicom.dcmread(path)


def get_structures(rtstruct):
    roi_map = {r.ROINumber: r.ROIName for r in rtstruct.StructureSetROISequence}
    structures = []

    for rc in rtstruct.ROIContourSequence:
        num = rc.ReferencedROINumber
        name = roi_map[num]

        if name != "PCM_Sup":
            continue

        contours = []
        for c in rc.ContourSequence:
            pts = np.array(c.ContourData).reshape(-1, 3)
            contours.append(pts)

        structures.append({"name": name, "contours": contours})

    return structures


# =========================
# POINT IN POLYGON
# =========================

def point_in_polygon(point, poly):
    x, y = point
    inside = False
    j = len(poly) - 1

    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]

        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi)*(y - yi)/(yj - yi + 1e-8) + xi):
            inside = not inside
        j = i

    return inside


# =========================
# FRACTIONAL MASK (HYBRID)
# =========================

def contours_to_occupancy(structure, ct_meta, supersample=3):
    shape = ct_meta["shape"]
    origin = ct_meta["origin"]
    spacing = ct_meta["spacing"]
    z_positions = np.array(ct_meta["z_positions"])

    occ = np.zeros(shape, dtype=np.float32)

    x0, y0 = origin[0], origin[1]
    dx, dy = spacing[0], spacing[1]

    offsets = np.linspace(0.5/supersample, 1-0.5/supersample, supersample)

    for pts in structure["contours"]:
        z = pts[0, 2]
        zi = int(np.argmin(np.abs(z_positions - z)))

        cols = (pts[:, 0] - x0) / dx
        rows = (pts[:, 1] - y0) / dy

        rr, cc = sk_polygon(rows, cols, shape=(shape[1], shape[2]))

        mask = np.zeros((shape[1], shape[2]), dtype=np.uint8)
        mask[rr, cc] = 1

        eroded = ndi.binary_erosion(mask)
        boundary = mask ^ eroded
        interior = mask & (~boundary)

        occ[zi, interior] = 1.0

        for r, c in zip(*np.where(boundary)):
            inside = 0
            total = supersample * supersample

            for oy in offsets:
                for ox in offsets:
                    pt = np.array([c + ox, r + oy])
                    if point_in_polygon(pt, np.column_stack([cols, rows])):
                        inside += 1

            occ[zi, r, c] = inside / total

    return occ


# =========================
# DVH + MEAN DOSE
# =========================

def compute_dvh(dose, occ, bins=500):
    d = dose.flatten()
    w = occ.flatten()

    valid = w > 0
    d = d[valid]
    w = w[valid]

    dose_bins = np.linspace(d.min(), d.max(), bins)

    dvh = np.array([
        100 * np.sum(w[d >= db]) / np.sum(w)
        for db in dose_bins
    ])

    mean = np.sum(d * w) / np.sum(w)

    return dose_bins, dvh, mean


# =========================
# MAIN
# =========================

def run(ct_dir, rtstruct_path, rtdose_path):

    ct, ct_meta = load_ct_series(ct_dir)
    dose = load_rtdose(rtdose_path, ct_meta)
    rtstruct = load_rtstruct(rtstruct_path)

    structures = get_structures(rtstruct)

    for s in structures:
        print(f"\nProcessing: {s['name']}")

        occ = contours_to_occupancy(s, ct_meta, supersample=3)

        bins, dvh, mean = compute_dvh(dose, occ)

        print(f"Mean dose: {mean:.2f} Gy")

        plt.plot(bins, dvh, label=s["name"])

    plt.xlabel("Dose (Gy)")
    plt.ylabel("Volume (%)")
    plt.title("DVH")
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run(
        ct_dir="CT/",
        rtstruct_path="RTSTRUCT.dcm",
        rtdose_path="RTDOSE.dcm"
    )