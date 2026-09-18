"""
DVH Computation Script
======================
Implements Dose-Volume Histogram (DVH) computation following the algorithms
described in RayStation v2025 Reference Manual, Chapter 11 (Volume Handling),
sections 11.1 and 11.2.

Required DICOM inputs:
  - CT image series     (.dcm, Modality=CT)
  - RT Structure Set    (.dcm, Modality=RTSTRUCT)
  - RT Dose             (.dcm, Modality=RTDOSE)

Usage:
  python compute_dvh.py --ct <ct_dir> --rtstruct <rtstruct.dcm> --rtdose <rtdose.dcm>

Optional:
  --roi-names NAME [NAME ...]   Only compute DVH for these ROI names (default: all)
  --output-csv <path>           Write DVH curves to CSV (default: dvh_output.csv)
  --plot                        Show matplotlib DVH plot after computation

Dependencies:
  pip install pydicom numpy scipy matplotlib
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")   # headless default; overridden below if --plot
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Section 11.1.1  ROI shapes
# ---------------------------------------------------------------------------
# RayStation uses three internal shapes: contour set, voxel ROI, triangle mesh.
# For DVH computation (§11.2) every ROI must ultimately become a dose-grid ROI
# (a voxel ROI on the dose grid with relative volume values 0-100 %).
#
# The pipeline implemented here mirrors Figure 86:
#   contour set  ──shape-based interpolation──►  voxel ROI  ──resample──► dose-grid ROI
#
# Voxel values range 0–255 (§11.1.1); a voxel ROI is "empty" when all values
# are below 127.5.  For the dose-grid ROI we store fractional occupancy [0, 1].
# ---------------------------------------------------------------------------


# ===========================================================================
# 1.  DICOM LOADING
# ===========================================================================

def load_ct_series(ct_dir: str):
    """Load an axial CT series; return sorted 3-D array + metadata."""
    dcm_files = sorted(Path(ct_dir).glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {ct_dir}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        if getattr(ds, "Modality", "") == "CT":
            slices.append(ds)

    if not slices:
        raise ValueError("No CT-modality DICOM files found.")

    # Sort by ImagePositionPatient z-coordinate (sup-inf axis)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    pixel_arrays = []
    z_positions = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(s.RescaleSlope) + float(s.RescaleIntercept)
        pixel_arrays.append(arr)
        z_positions.append(float(s.ImagePositionPatient[2]))

    ct_volume = np.stack(pixel_arrays, axis=0)   # (nz, ny, nx)

    ref = slices[0]
    pixel_spacing = [float(x) for x in ref.PixelSpacing]          # [row_mm, col_mm]
    image_position = [float(x) for x in ref.ImagePositionPatient] # [x0, y0, z0]

    z_positions = np.array(z_positions)
    slice_thickness = float(np.median(np.diff(z_positions))) if len(z_positions) > 1 else float(ref.SliceThickness)

    # Voxel size in mm: (dz, dy, dx)
    voxel_size_mm = np.array([abs(slice_thickness), pixel_spacing[0], pixel_spacing[1]])

    meta = {
        "image_position": image_position,   # (x0, y0, z0) of first voxel centre
        "voxel_size_mm": voxel_size_mm,     # (dz, dy, dx)
        "z_positions": z_positions,
        "shape": ct_volume.shape,           # (nz, ny, nx)
    }
    print(f"  CT loaded: {ct_volume.shape}, voxel size {voxel_size_mm} mm")
    return ct_volume, meta


def load_rtstruct(rtstruct_path: str, ct_z_positions: np.ndarray = None):
    """
    Return a dict {roi_name: {z_mm: [[(x,y), ...], ...]}} for each ROI.

    Fix #4: contour z-values are snapped to the nearest CT slice z position
    rather than being rounded to 4 decimal places.  Floating-point drift in
    DICOM ContourData means round() can place two contours that belong on the
    same slice onto different keys, or merge contours from adjacent slices.
    Snapping to the actual CT grid avoids both problems.
    """
    ds = pydicom.dcmread(rtstruct_path)
    if ds.Modality != "RTSTRUCT":
        raise ValueError("File is not an RTSTRUCT.")

    roi_names = {}
    for item in ds.StructureSetROISequence:
        roi_names[item.ROINumber] = item.ROIName

    rois = {}
    for roi_contour in ds.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        name = roi_names.get(roi_number, str(roi_number))
        contours_by_z = {}
        if not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            z_val = float(np.mean(pts[:, 2]))

            # Snap to nearest CT slice z (fix #4).  If no CT z-positions were
            # supplied fall back to the raw value so the function stays usable
            # standalone.
            if ct_z_positions is not None and len(ct_z_positions) > 0:
                nearest_idx = int(np.argmin(np.abs(ct_z_positions - z_val)))
                z_key = float(ct_z_positions[nearest_idx])
            else:
                z_key = z_val

            xy_pts = pts[:, :2].tolist()
            contours_by_z.setdefault(z_key, []).append(xy_pts)
        if contours_by_z:
            rois[name] = contours_by_z
    print(f"  RTSTRUCT loaded: {len(rois)} ROI(s): {list(rois.keys())}")
    return rois


def load_rtdose(rtdose_path: str):
    """Return dose array (Gy), dose grid origin, voxel size."""
    ds = pydicom.dcmread(rtdose_path)
    if ds.Modality != "RTDOSE":
        raise ValueError("File is not an RTDOSE.")

    dose = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)  # Gy

    pixel_spacing = [float(x) for x in ds.PixelSpacing]
    image_position = [float(x) for x in ds.ImagePositionPatient]

    if hasattr(ds, "GridFrameOffsetVector"):
        z_offsets = np.array([float(v) for v in ds.GridFrameOffsetVector])
    else:
        z_offsets = np.arange(dose.shape[0]) * float(ds.SliceThickness)

    z_positions = image_position[2] + z_offsets
    slice_thickness = float(np.median(np.diff(z_positions))) if len(z_positions) > 1 else float(ds.SliceThickness)

    voxel_size_mm = np.array([abs(slice_thickness), pixel_spacing[0], pixel_spacing[1]])

    meta = {
        "image_position": image_position,
        "voxel_size_mm": voxel_size_mm,
        "z_positions": z_positions,
        "shape": dose.shape,
    }
    print(f"  RTDOSE loaded: {dose.shape}, voxel size {voxel_size_mm} mm, "
          f"dose range [{dose.min():.2f}, {dose.max():.2f}] Gy")
    return dose, meta


# ===========================================================================
# 2.  CONTOUR → VOXEL ROI   (§11.1.4, "Converting contour sets to voxel ROIs")
# ===========================================================================
# The manual specifies shape-based interpolation [ref 49] adapted to produce
# relative volumes (0–100 %).  The major steps are reproduced below.
# ---------------------------------------------------------------------------

def _contour_to_binary_slice(xy_pts_list, nx, ny, origin_x, origin_y, dx, dy):
    """
    Rasterise one CT slice containing one or more contours.
    Returns a float32 array (ny, nx) with values in [0, 1].

    Step 3 (§11.1.4): pixels fully inside → 1.0, fully outside → 0.0,
    on the boundary → area fraction within contour.

    Fix #1: origin_x/y is the DICOM ImagePositionPatient corner of voxel [0,0],
    which is already the voxel-corner coordinate.  Pixel *centres* are therefore
    at origin + i*d, NOT origin + (i+0.5)*d.  The previous +0.5 offset shifted
    every pixel by half a voxel, mis-registering all contours.
    """
    from matplotlib.path import Path as MplPath

    mask = np.zeros((ny, nx), dtype=np.float32)

    # Pixel-centre coordinates: centre of pixel [i] is at origin + i*d
    col_coords = origin_x + np.arange(nx) * dx   # x centres  (fix #1)
    row_coords = origin_y + np.arange(ny) * dy   # y centres  (fix #1)
    grid_x, grid_y = np.meshgrid(col_coords, row_coords)
    pts_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    for xy_pts in xy_pts_list:
        if len(xy_pts) < 3:
            continue
        path = MplPath(xy_pts + [xy_pts[0]])  # close the polygon

        inside = path.contains_points(pts_grid).reshape(ny, nx).astype(np.float32)

        # Boundary refinement (§11.1.4 step 3): for pixels touched by the
        # contour we approximate the inside area fraction using a 4×4 sub-grid.
        # The manual guarantees that for touching voxels the contour is within
        # √2 × voxel_side of the voxel centre.
        boundary = path.contains_points(pts_grid, radius=max(dx, dy)).reshape(ny, nx)
        boundary_only = boundary & ~inside.astype(bool)
        if np.any(boundary_only):
            rows, cols = np.where(boundary_only)
            sub = 4
            sub_offsets = (np.arange(sub) + 0.5) / sub   # sub-pixel centres within voxel
            for r, c in zip(rows, cols):
                # Voxel corner is at origin + index*d; sub-pixel centres are
                # corner + sub_offsets * d  (fix #1: was origin + c*d + sub_offsets*d,
                # same formula but now consistent with corrected col_coords above)
                cx = origin_x + c * dx
                cy = origin_y + r * dy
                sub_x = cx + sub_offsets * dx
                sub_y = cy + sub_offsets * dy
                sg_x, sg_y = np.meshgrid(sub_x, sub_y)
                sub_pts = np.column_stack([sg_x.ravel(), sg_y.ravel()])
                frac = path.contains_points(sub_pts).mean()
                inside[r, c] = max(inside[r, c], frac)

        # Multiple contours on same slice may overlap (e.g. holes); use max
        mask = np.maximum(mask, inside)

    return mask


def _select_reconstruction_voxel_size(slice_distance_mm: float, n_contour_voxels_estimate: int = 0):
    """
    Choose the reconstruction voxel size following §11.1.4:
      (1) as close as possible to 0.5^3 mm^3
      (2) must be a fraction 1/(2^k), k>=1, of the slice distance
      (3) hard limits: 0.1^3 – 2.5^3 mm^3
      (4) if result would exceed 300^3 voxels, use a larger size (still satisfying (2) if possible)

    Returns voxel_size_mm (a single float; isotropic in x/y, matched to z separately).
    """
    target = 0.5   # mm  (§11.1.4: "as close as possible to 0.53 mm3" → edge ~0.5 mm)
    lower, upper = 0.1, 2.5

    # Enumerate candidate sizes: slice_distance / 2^k, k = 1, 2, ...
    candidates = []
    k = 1
    while True:
        size = slice_distance_mm / (2 ** k)
        if size < lower:
            break
        if lower <= size <= upper:
            candidates.append(size)
        k += 1

    if not candidates:
        # Fallback: clamp to limits
        return float(np.clip(slice_distance_mm / 2.0, lower, upper))

    # Pick closest to 0.5 mm
    best = min(candidates, key=lambda s: abs(s - target))
    return float(best)


def contour_set_to_voxel_roi(contours_by_z, dose_meta, super_sample=3):
    """
    Convert an ROI contour set to a dose-grid ROI following §11.1.4 + §11.2.2.

    Two-stage process (fix #5):
      Stage 1 – Voxelise contours onto a *fine reconstruction grid* (~0.5 mm,
                §11.1.4) using shape-based interpolation.
      Stage 2 – Resample the fine voxel ROI onto the *dose grid* with 3×
                axial super-sampling (§11.2.2), then average down.

    This matches RayStation's architecture and avoids the systematic volume
    inflation that occurred when voxelising directly on the coarse dose grid.

    Fixes applied
    -------------
    #1  Pixel centres use  origin + i*d  (not origin + (i+0.5)*d).
    #2  Super-sampled z positions are derived from dose_meta["z_positions"],
        not recomputed from scratch with a +0.5 offset.
    #3  Super-sampling averaging is now on a correctly-registered z grid.
    #5  Voxelisation is performed on the fine reconstruction grid, not the
        dose grid; the result is then resampled to the dose grid.
    """
    nz_dose, ny_dose, nx_dose = dose_meta["shape"]
    dz_dose, dy_dose, dx_dose = dose_meta["voxel_size_mm"]
    x0_dose, y0_dose, z0_dose = dose_meta["image_position"]
    z_dose = dose_meta["z_positions"]   # (nz_dose,) – authoritative z centres (fix #2)

    contour_z_vals = sorted(contours_by_z.keys())
    if not contour_z_vals:
        return np.zeros((nz_dose, ny_dose, nx_dose), dtype=np.float32)

    # ------------------------------------------------------------------
    # Stage 1a: Choose fine reconstruction grid (§11.1.4)
    # ------------------------------------------------------------------
    # Slice distance = median spacing between consecutive contour planes.
    if len(contour_z_vals) > 1:
        slice_dist = float(np.median(np.diff(contour_z_vals)))
    else:
        slice_dist = dz_dose   # single contour; fall back to dose grid spacing

    rec_dxy = _select_reconstruction_voxel_size(slice_dist)
    # z resolution: same rule, but we keep it aligned to the contour plane spacing
    rec_dz  = _select_reconstruction_voxel_size(slice_dist)

    # Bounding box of all contour points (+ 3.5 voxel margin, §11.1.4)
    all_x, all_y = [], []
    for xy_list in contours_by_z.values():
        for pts in xy_list:
            pts_arr = np.array(pts)
            all_x.extend(pts_arr[:, 0].tolist())
            all_y.extend(pts_arr[:, 1].tolist())
    all_x, all_y = np.array(all_x), np.array(all_y)

    margin_xy = 3.5 * rec_dxy
    margin_z  = 3.5 * rec_dz

    x_min = all_x.min() - margin_xy
    y_min = all_y.min() - margin_xy
    z_min = min(contour_z_vals) - margin_z
    x_max = all_x.max() + margin_xy
    y_max = all_y.max() + margin_xy
    z_max = max(contour_z_vals) + margin_z

    nx_rec = max(1, int(np.ceil((x_max - x_min) / rec_dxy)))
    ny_rec = max(1, int(np.ceil((y_max - y_min) / rec_dxy)))
    nz_rec = max(1, int(np.ceil((z_max - z_min) / rec_dz)))

    # Guard against > 300^3 voxels (§11.1.4) – double voxel size until it fits
    while nx_rec * ny_rec * nz_rec > 300 ** 3 and rec_dxy < 2.5:
        rec_dxy *= 2.0
        rec_dz  *= 2.0
        nx_rec = max(1, int(np.ceil((x_max - x_min) / rec_dxy)))
        ny_rec = max(1, int(np.ceil((y_max - y_min) / rec_dxy)))
        nz_rec = max(1, int(np.ceil((z_max - z_min) / rec_dz)))

    # Reconstruction grid centres (fix #1: centre of voxel i is origin + i*d)
    x_rec = x_min + np.arange(nx_rec) * rec_dxy
    y_rec = y_min + np.arange(ny_rec) * rec_dxy
    z_rec = z_min + np.arange(nz_rec) * rec_dz

    # ------------------------------------------------------------------
    # Stage 1b: Rasterise each contour plane onto the fine grid (steps 1–4)
    # ------------------------------------------------------------------
    def _signed_dist(binary_frac, dxy):
        """Signed EDT from a [0,1] occupancy map (§11.1.4 steps 2–4)."""
        inside_bin = (binary_frac >= 0.5).astype(np.uint8)
        d_in  = distance_transform_edt(inside_bin,     sampling=[dxy, dxy])
        d_out = distance_transform_edt(1 - inside_bin, sampling=[dxy, dxy])
        sd = d_in - d_out
        # Relative-volume adaption for boundary pixels (§11.1.4 step 3*)
        bnd = (binary_frac > 0.0) & (binary_frac < 1.0)
        if np.any(bnd):
            sd[bnd] = (binary_frac[bnd] - 0.5) * 2.0 * dxy
        return sd.astype(np.float32)

    sdt_at_contours = {}
    for z_val, xy_list in contours_by_z.items():
        bin_slice = _contour_to_binary_slice(
            xy_list, nx_rec, ny_rec, x_min, y_min, rec_dxy, rec_dxy
        )
        sdt_at_contours[z_val] = _signed_dist(bin_slice, rec_dxy)

    # ------------------------------------------------------------------
    # Stage 1c: Step 5 – linear interpolation in z between contour planes
    # ------------------------------------------------------------------
    z_c_arr  = np.array(contour_z_vals)
    sd_stack = np.stack([sdt_at_contours[z] for z in contour_z_vals], axis=0)  # (nc, ny_rec, nx_rec)

    sd_rec = np.full((nz_rec, ny_rec, nx_rec), fill_value=-1e6, dtype=np.float32)

    for i, zq in enumerate(z_rec):
        if zq < z_c_arr[0]:
            # Extrapolate: first contour extends half a slice distance (§11.1.4)
            half = slice_dist / 2.0
            if zq >= z_c_arr[0] - half:
                sd_rec[i] = sd_stack[0]
            # else stays at -1e6 (fully outside)
        elif zq > z_c_arr[-1]:
            half = slice_dist / 2.0
            if zq <= z_c_arr[-1] + half:
                sd_rec[i] = sd_stack[-1]
        else:
            idx = int(np.searchsorted(z_c_arr, zq)) - 1
            idx = int(np.clip(idx, 0, len(z_c_arr) - 2))
            z_lo, z_hi = z_c_arr[idx], z_c_arr[idx + 1]
            t = (zq - z_lo) / (z_hi - z_lo) if z_hi != z_lo else 0.0
            sd_rec[i] = (1.0 - t) * sd_stack[idx] + t * sd_stack[idx + 1]

    # ------------------------------------------------------------------
    # Stage 1d: Step 6 – hats on terminal contours (§11.1.4)
    # ------------------------------------------------------------------
    for k, z_val in enumerate(contour_z_vals):
        for direction, has_neighbour in [(-1, k > 0), (+1, k < len(z_c_arr) - 1)]:
            if has_neighbour:
                continue
            half_z = z_val + direction * (slice_dist / 2.0)
            for i, zq in enumerate(z_rec):
                if direction == -1 and half_z <= zq < z_val:
                    frac = (zq - half_z) / (z_val - half_z + 1e-9)
                    taper = float(np.sin(np.clip(frac, 0, 1) * np.pi / 2))
                    sd_rec[i] = np.maximum(sd_rec[i], sd_stack[k] * taper)
                elif direction == +1 and z_val < zq <= half_z:
                    frac = (zq - z_val) / (half_z - z_val + 1e-9)
                    taper = float(np.cos(np.clip(frac, 0, 1) * np.pi / 2))
                    sd_rec[i] = np.maximum(sd_rec[i], sd_stack[k] * taper)

    # Convert signed distance → relative volume [0, 1] on fine grid
    transition = rec_dxy
    # transition = rec_dxy * np.sqrt(2) / 2
    rel_vol_rec = np.clip(sd_rec / (2.0 * transition) + 0.5, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2: Resample fine reconstruction grid → dose grid (fix #2, #3, #5)
    #
    # §11.2.2: the conversion to contours for voxel→dose-grid is made at
    # "twice the dose grid resolution" and then shape-based interpolation
    # converts it; for contour→dose-grid a 3× axial super-sample is used.
    # We approximate both by trilinear interpolation from the fine grid,
    # which is already at sub-voxel resolution relative to the dose grid.
    # ------------------------------------------------------------------
    from scipy.interpolate import RegularGridInterpolator

    interp_fn = RegularGridInterpolator(
        (z_rec, y_rec, x_rec),
        rel_vol_rec,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Build query points on dose grid with axial super-sampling (fix #2, #3)
    # z_dose are the *actual* dose-grid z centres from DICOM (not recomputed).
    ss = super_sample   # 3 by default (§11.2.2)
    z_dose_ss = np.concatenate([
        z_dose[k] + (np.arange(ss) - (ss - 1) / 2.0) * (dz_dose / ss)
        for k in range(nz_dose)
    ])  # length nz_dose * ss, correctly centred on each dose voxel

    # x/y: sample at dose-grid pixel centres (fix #1 equivalent for dose grid)
    x_dose = x0_dose + np.arange(nx_dose) * dx_dose
    y_dose = y0_dose + np.arange(ny_dose) * dy_dose

    zz, yy, xx = np.meshgrid(z_dose_ss, y_dose, x_dose, indexing="ij")
    query_pts = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])

    rel_vol_ss = interp_fn(query_pts).reshape(nz_dose * ss, ny_dose, nx_dose).astype(np.float32)

    # Average super-sampled slices back to dose-grid z resolution (fix #3)
    rel_vol_dose = rel_vol_ss.reshape(nz_dose, ss, ny_dose, nx_dose).mean(axis=1)

    return rel_vol_dose.astype(np.float32)


# ===========================================================================
# 3.  DVH COMPUTATION   (§11.2.3)
# ===========================================================================
# §11.2.3 assumptions:
#   • Dose is constant within each dose grid voxel.
#   • D(v) = minimum dose received by the sub-volume of size v accumulated
#            from highest to lowest voxel doses.
#   • V(d) = accumulated volume in an ROI that receives at least dose d.
#
# These are the standard cumulative DVH definitions.
# The manual also warns that for small ROIs (few voxels), discretisation of
# volume will be apparent (§11.2.3).
# ---------------------------------------------------------------------------

def compute_dvh(dose_grid, voxel_roi, voxel_volume_cc):
    """
    Compute a cumulative DVH following §11.2.3.

    Parameters
    ----------
    dose_grid      : np.ndarray (nz, ny, nx)  dose in Gy
    voxel_roi      : np.ndarray (nz, ny, nx)  relative volumes in [0, 1]
    voxel_volume_cc: float  physical volume of one dose voxel in cc (cm³)

    Returns
    -------
    dose_bins : np.ndarray  dose values (Gy) for DVH x-axis
    vol_cc    : np.ndarray  cumulative volume (cc) receiving ≥ dose_bins[i]
    vol_pct   : np.ndarray  cumulative volume (%) receiving ≥ dose_bins[i]
    total_vol : float       total ROI volume in cc
    stats     : dict        {D_min, D_max, D_mean, D98, D2, V_total_cc}
    """
    # §11.1.1: voxel ROI "empty" if all values < 0.5 (127.5/255)
    roi_mask = voxel_roi > 0.0     # non-zero relative volumes only

    if not np.any(roi_mask):
        warnings.warn("ROI is empty (all voxel values < threshold).")
        empty = np.array([0.0])
        return empty, empty, empty, 0.0, {}

    # Weighted volumes per voxel (relative_vol × physical_voxel_volume)
    weighted_vol = voxel_roi[roi_mask] * voxel_volume_cc   # (N,)
    voxel_doses  = dose_grid[roi_mask]                     # (N,)

    total_vol = weighted_vol.sum()                         # total ROI volume (cc)

    # Sort voxels from highest to lowest dose (§11.2.3: accumulated from highest)
    sort_idx = np.argsort(voxel_doses)[::-1]
    sorted_doses = voxel_doses[sort_idx]
    sorted_vols  = weighted_vol[sort_idx]
    cum_vol      = np.cumsum(sorted_vols)                  # cumulative vol at each step

    # Cumulative DVH: V(d) = volume receiving ≥ d
    # Represented as (dose_values, cumulative_volume) pairs
    # We build a fine dose axis covering [D_min, D_max]
    d_min = float(sorted_doses[-1])   # lowest dose in ROI (§11.2.3)
    d_max = float(sorted_doses[0])    # highest dose in ROI (§11.2.3)

    n_bins = max(500, len(sorted_doses))
    dose_bins = np.linspace(0.0, d_max * 1.05, n_bins)

    # V(d) by interpolation on the sorted cumulative curve
    # cum_vol[i] = total volume receiving dose >= sorted_doses[i]
    # We need V(d) = volume receiving >= d  →  look up in the sorted arrays.
    # Append boundary points for cleaner interpolation
    interp_doses = np.concatenate([[d_max + 1e-9], sorted_doses, [d_min - 1e-9]])
    interp_vols  = np.concatenate([[0.0], cum_vol, [total_vol]])

    vol_cc = np.interp(dose_bins, interp_doses[::-1], interp_vols[::-1])
    vol_cc = np.clip(vol_cc, 0.0, total_vol)
    vol_pct = 100.0 * vol_cc / total_vol if total_vol > 0 else vol_cc * 0.0

    # ------------------------------------------------------------------
    # Dose statistics (§11.2.3)
    # D_min  = lowest  voxel dose over all voxels with non-zero relative volume
    # D_max  = highest voxel dose within the ROI
    # D_mean = dose-weighted mean
    # D98    = near-minimum (recommended over D_min per ICRU 83)
    # D2     = near-maximum (recommended over D_max per ICRU 83)
    # ------------------------------------------------------------------
    d_mean = float(np.sum(voxel_doses * weighted_vol) / total_vol)

    def dose_at_volume(v_frac):
        """Return D(v): min dose received by the v-fraction sub-volume
        accumulated from highest dose (§11.2.3 definition of D(v))."""
        target_vol = v_frac * total_vol
        idx = np.searchsorted(cum_vol, target_vol)
        idx = min(idx, len(sorted_doses) - 1)
        return float(sorted_doses[idx])

    d98 = dose_at_volume(0.98)
    d2  = dose_at_volume(0.02)

    stats = {
        "D_min_Gy":  round(d_min, 4),
        "D_max_Gy":  round(d_max, 4),
        "D_mean_Gy": round(d_mean, 4),
        "D98_Gy":    round(d98, 4),   # near-minimum  (ICRU 83)
        "D2_Gy":     round(d2, 4),    # near-maximum  (ICRU 83)
        "V_total_cc": round(total_vol, 4),
    }

    return dose_bins, vol_cc, vol_pct, total_vol, stats


# ===========================================================================
# 4.  OUTPUT: CSV + PLOT
# ===========================================================================

def write_csv(output_path, dvh_results):
    """Write all DVH curves to a single CSV file."""
    import csv

    # Find the dose axis with most bins (use as reference x-axis)
    all_dose = {}
    for roi_name, (dose_bins, vol_cc, vol_pct, _, _) in dvh_results.items():
        all_dose[roi_name] = (dose_bins, vol_cc, vol_pct)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Dose_Gy"]
        for n in dvh_results:
            header += [f"{n}_Vol_cc", f"{n}_Vol_pct"]
        writer.writerow(header)

        # Use the first ROI's dose axis; interpolate others onto it
        ref_name = list(dvh_results.keys())[0]
        ref_doses, _, _ = all_dose[ref_name]
        for i, d in enumerate(ref_doses):
            row = [round(float(d), 6)]
            for n in dvh_results:
                d_ax, v_cc, v_pct = all_dose[n]
                row.append(round(float(np.interp(d, d_ax, v_cc)), 6))
                row.append(round(float(np.interp(d, d_ax, v_pct)), 6))
            writer.writerow(row)

    print(f"  DVH curves written to {output_path}")


def plot_dvh(dvh_results, output_png=None):
    """Plot cumulative DVH curves (one per ROI)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dvh_results), 1)))

    for (roi_name, (dose_bins, vol_cc, vol_pct, total_vol, stats)), color in zip(
        dvh_results.items(), colors
    ):
        label = (
            f"{roi_name}  "
            f"[Dmean={stats.get('D_mean_Gy','?')} Gy, "
            f"D98={stats.get('D98_Gy','?')} Gy, "
            f"D2={stats.get('D2_Gy','?')} Gy]"
        )
        ax.plot(dose_bins, vol_pct, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title("Cumulative Dose-Volume Histogram\n(§11.2.3 RayStation v2025 Reference Manual)",
                 fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_png:
        fig.savefig(output_png, dpi=150)
        print(f"  DVH plot saved to {output_png}")
    return fig


# ===========================================================================
# 5.  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute DVHs from DICOM CT / RTSTRUCT / RTDOSE\n"
                    "(algorithm: RayStation v2025 Reference Manual §11)"
    )
    parser.add_argument("--ct",       required=False, help="Directory containing CT DICOM files")
    parser.add_argument("--rtstruct", required=False, help="Path to RTSTRUCT DICOM file")
    parser.add_argument("--rtdose",   required=False, help="Path to RTDOSE DICOM file")
    parser.add_argument("--roi-names", nargs="*",    help="ROI names to compute (default: all)")
    parser.add_argument("--output-csv", default="dvh_output.csv")
    parser.add_argument("--output-png", default="dvh_output.png")
    parser.add_argument("--plot", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    output_dir =  r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\stride_RT\TEST_OUTPUT"
    args.ct = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\stride_RT\test_raw_data\STRIDE_RT_DATA\CT"
    args.rtstruct = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\stride_RT\test_raw_data\STRIDE_RT_DATA\RTSTRUCT\RS1.dcm"
    args.rtdose = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\stride_RT\test_raw_data\STRIDE_RT_DATA\RTDOSE\RD1.2.752.243.1.1.20241113161836129.9300.27742.dcm"
    args.output_csv = os.path.join(output_dir, "dvh_output.csv")
    args.output_png = os.path.join(output_dir, "dvh_output.png")

    print("\n=== DVH Computation (RayStation v2025 §11 algorithm) ===\n")

    # ------------------------------------------------------------------
    # Load DICOM data
    # ------------------------------------------------------------------
    print("Loading CT series ...")
    #ct_volume, ct_meta = load_ct_series(args.ct)

    print("Loading RT Structure Set ...")
    # Pass CT z-positions so contour z-values are snapped to the CT grid (fix #4)
    rois = load_rtstruct(args.rtstruct, ct_z_positions=None) # ct_meta["z_positions"])

    print("Loading RT Dose ...")
    dose_grid, dose_meta = load_rtdose(args.rtdose)

    # ------------------------------------------------------------------
    # Filter ROIs if requested
    # ------------------------------------------------------------------
    roi_names_to_process = args.roi_names if args.roi_names else list(rois.keys())
    missing = [n for n in roi_names_to_process if n not in rois]
    if missing:
        print(f"  WARNING: ROI(s) not found in RTSTRUCT: {missing}")
        roi_names_to_process = [n for n in roi_names_to_process if n in rois]

    if not roi_names_to_process:
        print("No ROIs to process. Exiting.")
        sys.exit(1)

    roi_names_to_process = ["OralCavity_Ext", "OralCavity_Ext_Out", "PCM_Sup", "PCM_Sup_Out",
                            "PCM_Med", "PCM_Med_Out", "PCM_Inf", "PCM_Inf_Out", "Parotid_L", "Parotid_L_Out", "Parotid_R", "Parotid_R_Out"]


    # Physical volume of a single dose voxel (cc = cm³)
    dz, dy, dx = dose_meta["voxel_size_mm"]
    voxel_vol_cc = (dz * dy * dx) / 1000.0   # mm³ → cm³

    # ------------------------------------------------------------------
    # Process each ROI
    # ------------------------------------------------------------------
    dvh_results = {}

    for roi_name in roi_names_to_process:
        print(f"\nProcessing ROI: '{roi_name}'")
        contours_by_z = rois[roi_name]

        print(f"  Voxelising contours onto fine reconstruction grid, then resampling to dose grid (§11.1.4 + §11.2.2) ...")
        voxel_roi = contour_set_to_voxel_roi(contours_by_z, dose_meta)

        n_nonzero = np.sum(voxel_roi > 0)
        print(f"  Voxel ROI: {n_nonzero} non-zero voxels, "
              f"total ROI volume ≈ {np.sum(voxel_roi) * voxel_vol_cc:.2f} cc")

        if n_nonzero == 0:
            print(f"  WARNING: ROI '{roi_name}' produced an empty voxel ROI — skipping.")
            continue

        print(f"  Step 6: Computing DVH (§11.2.3) ...")
        dose_bins, vol_cc, vol_pct, total_vol, stats = compute_dvh(
            dose_grid, voxel_roi, voxel_vol_cc
        )

        dvh_results[roi_name] = (dose_bins, vol_cc, vol_pct, total_vol, stats)

        print(f"  Results for '{roi_name}':")
        for k, v in stats.items():
            print(f"    {k:15s} = {v}")

    if not dvh_results:
        print("\nNo DVH results produced.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print(f"\nWriting outputs ...")
    write_csv(args.output_csv, dvh_results)

    plot_dvh(dvh_results, output_png=args.output_png)

    if args.plot:
        matplotlib.use("TkAgg")
        import importlib
        import matplotlib.pyplot as _plt
        importlib.reload(_plt)
        fig = plot_dvh(dvh_results)
        _plt.show()

    print("\nDone.")




if __name__ == "__main__":
    main()