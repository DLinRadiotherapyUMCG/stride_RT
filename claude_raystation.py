"""
RTSTRUCT DVH Computation Script
================================
Loads DICOM CT + RTSTRUCT files and computes Dose-Volume Histograms (DVHs)
for each structure. Also computes standard dose statistics (Dmean, Dmin, Dmax,
D2, D50, D98, V20Gy, V30Gy, etc.).

Usage
-----
    python rtstruct_dvh.py --ct_dir /path/to/ct --rtstruct /path/to/RS.dcm \
                            --rtdose /path/to/RD.dcm --output_dir /path/to/output

    # or as a library:
    from rtstruct_dvh import RTStructDVHCalculator
    calc = RTStructDVHCalculator(ct_dir, rtstruct_path, rtdose_path)
    calc.run()

Dependencies
------------
    pip install pydicom numpy scipy matplotlib scikit-image tqdm
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom
from pydicom import dcmread
from skimage.draw import polygon as sk_polygon
import scipy.ndimage as ndi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# DICOM Loader helpers
# ===========================================================================

def load_ct_series(ct_dir: str) -> Tuple[np.ndarray, dict]:
    """
    Load a CT DICOM series from a folder and return the 3-D HU array
    (z, y, x) and a metadata dict.

    Returns
    -------
    ct_array : np.ndarray, dtype float32, shape (nz, ny, nx)
    meta     : dict  with keys 'origin', 'spacing', 'direction', 'z_positions'
    """
    files = sorted(Path(ct_dir).glob("**/*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {ct_dir}")

    slices = []
    for f in files:
        try:
            ds = dcmread(str(f), force=True)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            pass

    if not slices:
        raise ValueError("Could not read any CT slices.")

    # Sort by z-position (ImagePositionPatient[2])
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    log.info("Loaded %d CT slices", len(slices))

    # Build 3-D array and apply rescale slope/intercept → HU
    arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        arrays.append(arr * slope + intercept)

    ct_array = np.stack(arrays, axis=0)  # (nz, ny, nx)

    # Spacing
    s0 = slices[0]
    xy_spacing = [float(v) for v in s0.PixelSpacing]   # [row_spacing, col_spacing]
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    if len(z_positions) > 1:
        z_spacing = abs(z_positions[1] - z_positions[0])
    else:
        z_spacing = float(getattr(s0, "SliceThickness", 1.0))

    origin = [float(v) for v in s0.ImagePositionPatient]  # (x, y, z)

    meta = {
        "origin":      origin,          # (x0, y0, z0) in mm
        "spacing":     [xy_spacing[1],  # dx (col, x)
                        xy_spacing[0],  # dy (row, y)
                        z_spacing],     # dz
        "z_positions": z_positions,
        "slices":      slices,
        "shape":       ct_array.shape,  # (nz, ny, nx)
    }
    return ct_array, meta


def load_rtdose(rtdose_path: str, ct_meta: dict) -> np.ndarray:
    """
    Load an RTDOSE DICOM file and resample it onto the CT grid using
    nearest-neighbour interpolation in physical space.

    Returns
    -------
    dose_array : np.ndarray, dtype float32, shape = ct_meta['shape'],  units = Gy
    """
    ds = dcmread(rtdose_path)
    raw = ds.pixel_array.astype(np.float64)   # (z, y, x)
    dgs = float(ds.DoseGridScaling)
    dose_raw = (raw * dgs).astype(np.float32)  # Gy

    # Physical coordinates of dose grid voxels
    dose_origin  = [float(v) for v in ds.ImagePositionPatient]  # (x,y,z)
    dose_spacing_xy = [float(v) for v in ds.PixelSpacing]
    try:
        dose_z_positions = [
            float(o) for o in ds.GridFrameOffsetVector
        ]
        dose_z_positions = [dose_origin[2] + dz for dz in dose_z_positions]
    except Exception:
        nz_d = raw.shape[0]
        dose_z_positions = [dose_origin[2] + i * float(ds.SliceThickness)
                            for i in range(nz_d)]

    log.info("RTDOSE grid: %s, DoseGridScaling=%.6f, max dose=%.2f Gy",
             dose_raw.shape, dgs, dose_raw.max())

    # ---------- resample onto CT grid ----------
    ct_shape  = ct_meta["shape"]            # (nz, ny, nx)
    ct_origin  = ct_meta["origin"]          # (x0, y0, z0)
    ct_spacing = ct_meta["spacing"]         # (dx, dy, dz)
    ct_z       = ct_meta["z_positions"]

    # Build CT voxel-centre coordinate arrays in physical mm
    ct_x = ct_origin[0] + np.arange(ct_shape[2]) * ct_spacing[0]  # cols
    ct_y = ct_origin[1] + np.arange(ct_shape[1]) * ct_spacing[1]  # rows
    ct_z_arr = np.array(ct_z)

    dose_z_arr = np.array(dose_z_positions)

    # Sort dose z-axis so interpolation in index space is well-defined
    sort_order = np.argsort(dose_z_arr)
    dose_z_sorted = dose_z_arr[sort_order]
    dose_raw_sorted = dose_raw[sort_order, :, :]

    # Convert CT physical coordinates to floating dose-array indices
    ix_f = (ct_x - dose_origin[0]) / dose_spacing_xy[1]
    iy_f = (ct_y - dose_origin[1]) / dose_spacing_xy[0]
    iz_f = np.interp(
        ct_z_arr,
        dose_z_sorted,
        np.arange(len(dose_z_sorted), dtype=np.float64),
        left=0.0,
        right=float(len(dose_z_sorted) - 1),
    )

    # Broadcast to full CT grid and trilinearly sample dose grid
    iz_grid = np.broadcast_to(iz_f[:, None, None], ct_shape)
    iy_grid = np.broadcast_to(iy_f[None, :, None], ct_shape)
    ix_grid = np.broadcast_to(ix_f[None, None, :], ct_shape)

    dose_on_ct = ndi.map_coordinates(
        dose_raw_sorted,
        [iz_grid, iy_grid, ix_grid],
        order=1,
        mode="nearest",
    )
    log.info("Resampled RTDOSE onto CT grid %s", dose_on_ct.shape)
    return dose_on_ct.astype(np.float32)


# ===========================================================================
# RTSTRUCT  →  voxel masks
# ===========================================================================

def load_rtstruct(rtstruct_path: str) -> Tuple[pydicom.Dataset, List[dict]]:
    """
    Load an RTSTRUCT file and return the Dataset plus a list of structure
    dicts with keys: 'name', 'number', 'color', 'contours'.
    """
    ds = dcmread(rtstruct_path)
    structures = []

    roi_name_map = {
        item.ROINumber: item.ROIName
        for item in ds.StructureSetROISequence
    }
    roi_color_map = {}
    if hasattr(ds, "ROIContourSequence"):
        for item in ds.ROIContourSequence:
            try:
                roi_color_map[item.ReferencedROINumber] = list(item.ROIDisplayColor)
            except Exception:
                roi_color_map[item.ReferencedROINumber] = [255, 0, 0]

    if not hasattr(ds, "ROIContourSequence"):
        log.warning("RTSTRUCT has no ROIContourSequence.")
        return ds, []

    for roi_contour in ds.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        roi_name   = roi_name_map.get(roi_number, f"ROI_{roi_number}")
        color      = roi_color_map.get(roi_number, [255, 0, 0])

        contour_slices = []
        if hasattr(roi_contour, "ContourSequence"):
            for contour in roi_contour.ContourSequence:
                try:
                    pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
                    contour_slices.append(pts)   # (N, 3) x,y,z
                except Exception:
                    pass

        structures.append({
            "name":     roi_name,
            "number":   roi_number,
            "color":    color,
            "contours": contour_slices,
        })
        log.info("  ROI %3d  %-30s  %d contour slices",
                 roi_number, roi_name, len(contour_slices))

    return ds, structures


# RayStation-style super-sampling parameters
_SUPER_XY = 3   # in-plane (x, y) super-sampling factor
_SUPER_Z  = 3   # axial (z) super-sampling factor


def contours_to_mask(
    structure: dict,
    ct_meta: dict,
    super_xy: int = _SUPER_XY,
    super_z:  int = _SUPER_Z,
) -> np.ndarray:
    """
    Rasterise all contour slices of a single structure into a floating-point
    partial-volume mask on the CT grid, approximating RayStation's 3×
    super-sampling approach.

    In-plane (x, y)
        Each contour polygon is rasterised onto a grid ``super_xy`` times
        finer than the CT pixel grid, then average-pooled back.  Voxels
        whose centres straddle the polygon boundary receive a fractional
        weight in (0, 1) instead of a hard 0 or 1.

    Axial (z)
        For each CT slice, ``super_z`` sub-positions are sampled at evenly
        spaced offsets within the slice thickness.  A sub-position only
        contributes if its nearest contour plane is within half a CT-slice
        thickness.  This models the partial axial occupancy of boundary
        slices.

    Returns
    -------
    mask : np.ndarray, dtype float32, shape = ct_meta['shape']  (nz, ny, nx)
        Values in [0, 1] – fractional voxel occupancy.
    """
    ct_shape   = ct_meta["shape"]       # (nz, ny, nx)
    ct_origin  = ct_meta["origin"]      # (x0, y0, z0) in mm
    ct_spacing = ct_meta["spacing"]     # (dx, dy, dz)
    ct_z       = np.array(ct_meta["z_positions"])

    nz, ny, nx = ct_shape
    x0, y0     = ct_origin[0], ct_origin[1]
    dx, dy, dz = ct_spacing

    # --- group valid contours by nearest CT slice -----------------------
    contours_by_slice: Dict[int, List[np.ndarray]] = {}
    for pts in structure["contours"]:
        if len(pts) < 3:
            continue
        z_phys  = pts[0, 2]
        z_diffs = np.abs(ct_z - z_phys)
        zi      = int(np.argmin(z_diffs))
        if z_diffs[zi] <= dz * 1.5:
            contours_by_slice.setdefault(zi, []).append(pts)

    if not contours_by_slice:
        return np.zeros(ct_shape, dtype=np.float32)

    slices_with_contours = sorted(contours_by_slice.keys())

    # --- super-sampled in-plane grid ------------------------------------
    nx_s = nx * super_xy
    ny_s = ny * super_xy
    dx_s = dx / super_xy
    dy_s = dy / super_xy

    # Axial sub-sample offsets within one CT slice (units of dz).
    # Evenly spaced, centred on the slice centre:
    #   e.g. super_z=3 → -1/3, 0, +1/3
    sub_offsets = [(k + 0.5) / super_z - 0.5 for k in range(super_z)]

    accum = np.zeros((nz, ny, nx), dtype=np.float32)

    for zi in range(nz):
        for sub_dz in sub_offsets:
            z_sub = ct_z[zi] + sub_dz * dz

            # Find the nearest contoured slice to this sub-z position
            nearest_zi = min(
                slices_with_contours,
                key=lambda s: abs(ct_z[s] - z_sub),
            )
            if abs(ct_z[nearest_zi] - z_sub) > dz * 0.5:
                continue   # no contour within half a slice → empty

            # Render all contours for that slice at super-sampled resolution.
            # XOR fill (^= 1) handles even-odd rule for hollow structures.
            slice_s = np.zeros((ny_s, nx_s), dtype=np.uint8)
            for pts in contours_by_slice[nearest_zi]:
                cols_s = (pts[:, 0] - x0) / dx_s
                rows_s = (pts[:, 1] - y0) / dy_s
                rr, cc = sk_polygon(rows_s, cols_s, shape=(ny_s, nx_s))
                slice_s[rr, cc] ^= 1

            # Average-pool back to native CT resolution → fractional weights
            pooled = (
                slice_s
                .reshape(ny, super_xy, nx, super_xy)
                .mean(axis=(1, 3), dtype=np.float32)
            )
            accum[zi] += pooled

    # Normalise by the number of axial sub-samples → [0, 1]
    return accum / super_z


# ===========================================================================
# DVH computation  (per RayStation reference manual §11.2.3)
# ===========================================================================

def compute_dvh(
    dose_array: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a cumulative DVH for dose values within `mask`.

    The DVH follows the RayStation convention (§11.2.3):
      - D(V%) = minimum dose received by the hottest V% of the volume
      - V(D)  = fraction of volume receiving ≥ D

    Parameters
    ----------
    dose_array : (nz, ny, nx) float32  in Gy
    mask       : (nz, ny, nx) bool
    n_bins     : number of dose bins

    Returns
    -------
    dose_bins  : ndarray shape (n_bins,)  in Gy
    volume_pct : ndarray shape (n_bins,)  in %   (cumulative DVH)
    """
    # Support fractional (float) masks from super-sampling as well as binary.
    sel = mask > 0
    d   = dose_array[sel].astype(np.float64)
    w   = mask[sel].astype(np.float64)        # partial-volume weights

    if d.size == 0:
        return np.array([0.0, 0.0]), np.array([100.0, 0.0])

    total_w  = w.sum()
    d_min, d_max = d.min(), d.max()
    if d_max <= d_min:
        return np.array([d_min, d_min]), np.array([100.0, 0.0])

    dose_bins = np.linspace(d_min, d_max, n_bins)
    # Weighted cumulative DVH: fraction of volume receiving >= each dose level
    volume_pct = np.array(
        [100.0 * w[d >= db].sum() / total_w for db in dose_bins]
    )
    return dose_bins, volume_pct


# ===========================================================================
# Dose statistics  (matching RayStation TG-186 / §11.2.3 convention)
# ===========================================================================

def compute_dose_statistics(
    dose_array: np.ndarray,
    mask: np.ndarray,
    voxel_volume_cc: float,
) -> dict:
    """
    Compute standard dose statistics for a structure.

    Parameters
    ----------
    dose_array      : (nz, ny, nx) float32  in Gy
    mask            : (nz, ny, nx) bool
    voxel_volume_cc : volume of one voxel in cm³

    Returns
    -------
    stats : dict
        Dmean, Dmin, Dmax, D2, D5, D50, D95, D98, V5Gy, V10Gy, V20Gy,
        V30Gy, V40Gy, V45Gy, V60Gy, volume_cc, n_voxels
    """
    # Support fractional (float) masks from super-sampling as well as binary.
    sel = mask > 0
    d   = dose_array[sel].astype(np.float64)
    w   = mask[sel].astype(np.float64)   # partial-volume weights in [0, 1]

    if d.size == 0:
        return {k: np.nan for k in [
            "Dmean_Gy", "Dmin_Gy", "Dmax_Gy",
            "D2_Gy", "D5_Gy", "D50_Gy", "D95_Gy", "D98_Gy",
            "V5Gy_pct", "V10Gy_pct", "V20Gy_pct",
            "V30Gy_pct", "V40Gy_pct", "V45Gy_pct", "V60Gy_pct",
            "volume_cc", "n_voxels"
        ]}

    total_w = w.sum()
    # Volume = weighted voxel count × voxel size
    vol_cc  = total_w * voxel_volume_cc

    # Weighted D(x%): sort dose descending, walk the weight CDF.
    # D95 = dose at which the cumulative weight from the top reaches 95 %.
    _sort_idx  = np.argsort(d)[::-1]
    _d_sorted  = d[_sort_idx]
    _w_cumsum  = np.cumsum(w[_sort_idx])

    def dx(pct: float) -> float:
        """Weighted D_pct: dose such that pct% of volume receives ≥ this dose."""
        target = pct / 100.0 * total_w
        idx    = np.searchsorted(_w_cumsum, target, side="left")
        return float(_d_sorted[min(idx, len(_d_sorted) - 1)])

    def vx_pct(dose_level_gy: float) -> float:
        """Weighted V_dose: fraction of volume (%) receiving ≥ dose_level_gy."""
        return 100.0 * float(w[d >= dose_level_gy].sum()) / total_w

    stats = {
        "Dmean_Gy": float(np.average(d, weights=w)),
        # Dmin / Dmax reflect the extreme dose at any voxel with non-zero weight
        "Dmin_Gy":  float(d.min()),
        "Dmax_Gy":  float(d.max()),
        "D2_Gy":    dx(2),
        "D5_Gy":    dx(5),
        "D50_Gy":   dx(50),
        "D95_Gy":   dx(95),
        "D98_Gy":   dx(98),
        # Volume metrics
        "V5Gy_pct":  vx_pct(5.0),
        "V10Gy_pct": vx_pct(10.0),
        "V20Gy_pct": vx_pct(20.0),
        "V30Gy_pct": vx_pct(30.0),
        "V40Gy_pct": vx_pct(40.0),
        "V45Gy_pct": vx_pct(45.0),
        "V60Gy_pct": vx_pct(60.0),
        "volume_cc": vol_cc,
        "n_voxels":  int(d.size),   # integer count of non-zero-weight voxels
    }
    return stats


# ===========================================================================
# Plotting
# ===========================================================================

def plot_dvhs(
    dvh_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Dose-Volume Histograms",
):
    """
    Plot all structure DVHs on a single figure and save as PNG.

    Parameters
    ----------
    dvh_data    : dict  { struct_name: (dose_bins, volume_pct) }
    output_path : str   path for the output PNG
    title       : str
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = cm.tab20(np.linspace(0, 1, max(len(dvh_data), 1)))

    for (name, (d_bins, v_pct)), color in zip(dvh_data.items(), colors):
        ax.plot(d_bins, v_pct, label=name, linewidth=1.5, color=color)

    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        loc="upper right",
        fontsize=7,
        framealpha=0.8,
        ncol=max(1, len(dvh_data) // 20),
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("DVH figure saved → %s", output_path)


def plot_dose_overlay(
    ct_array: np.ndarray,
    dose_array: np.ndarray,
    masks: Dict[str, np.ndarray],
    output_path: str,
    slice_idx: Optional[int] = None,
):
    """
    Save a PNG with the central CT slice overlaid by the dose distribution
    and structure contours.
    """
    nz = ct_array.shape[0]
    zi = slice_idx if slice_idx is not None else nz // 2

    fig, ax = plt.subplots(figsize=(8, 8))

    # CT background (window/level: W=400, L=40)
    ww, wl = 400, 40
    ct_disp = np.clip((ct_array[zi] - (wl - ww / 2)) / ww, 0, 1)
    ax.imshow(ct_disp, cmap="gray", aspect="equal", origin="upper")

    # Dose overlay
    dose_slice = dose_array[zi]
    if dose_slice.max() > 0:
        dose_masked = np.ma.masked_where(dose_slice < 0.05 * dose_array.max(), dose_slice)
        ax.imshow(dose_masked, cmap="jet", alpha=0.4,
                  vmin=0, vmax=dose_array.max(), aspect="equal", origin="upper")

    # Structure contours
    colors = cm.tab20(np.linspace(0, 1, max(len(masks), 1)))
    for (name, mask), color in zip(masks.items(), colors):
        struct_slice = mask[zi].astype(np.uint8)
        if struct_slice.sum() == 0:
            continue
        import cv2 as _cv2
        contours, _ = _cv2.findContours(struct_slice, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color=color, linewidth=1, label=name)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=6, loc="lower right")
    ax.set_title(f"CT + Dose overlay  (slice {zi})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Overlay figure saved → %s", output_path)


# ===========================================================================
# Statistics CSV export
# ===========================================================================

def save_statistics_csv(
    all_stats: Dict[str, dict],
    output_path: str,
):
    """Write a CSV table with one row per structure."""
    import csv

    if not all_stats:
        return

    fieldnames = ["structure"] + list(next(iter(all_stats.values())).keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in all_stats.items():
            row = {"structure": name, **stats}
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    log.info("Statistics CSV saved → %s", output_path)


# ===========================================================================
# Main calculator class
# ===========================================================================

class RTStructDVHCalculator:
    """
    End-to-end pipeline:
        CT series  +  RTSTRUCT  +  (optional) RTDOSE  →  DVH + statistics
    """

    def __init__(
        self,
        ct_dir: str,
        rtstruct_path: str,
        rtdose_path: Optional[str] = None,
        output_dir: str = "dvh_output",
        structures_to_include: Optional[List[str]] = None,
        n_dvh_bins: int = 500,
    ):
        self.ct_dir              = ct_dir
        self.rtstruct_path       = rtstruct_path
        self.rtdose_path         = rtdose_path
        self.output_dir          = output_dir
        self.structures_to_include = structures_to_include
        self.n_dvh_bins          = n_dvh_bins

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #

    def run(self):
        log.info("=" * 60)
        log.info("RTStructDVHCalculator  starting")
        log.info("  CT dir      : %s", self.ct_dir)
        log.info("  RTSTRUCT    : %s", self.rtstruct_path)
        log.info("  RTDOSE      : %s", self.rtdose_path or "(none)")
        log.info("  Output dir  : %s", self.output_dir)
        log.info("=" * 60)

        # ---- 1. Load CT ------------------------------------------------
        log.info("Loading CT series …")
        ct_array, ct_meta = load_ct_series(self.ct_dir)
        dx, dy, dz = ct_meta["spacing"]
        voxel_vol_cc = (dx * dy * dz) / 1000.0   # mm³ → cm³
        log.info("CT shape: %s  spacing: %.2f × %.2f × %.2f mm",
                 ct_array.shape, dx, dy, dz)

        # ---- 2. Load RTSTRUCT ------------------------------------------
        log.info("Loading RTSTRUCT …")
        _, structures = load_rtstruct(self.rtstruct_path)
        log.info("Found %d structures", len(structures))

        # ---- 3. Load RTDOSE (optional) ---------------------------------
        if self.rtdose_path:
            log.info("Loading RTDOSE …")
            dose_array = load_rtdose(self.rtdose_path, ct_meta)
        else:
            log.info("No RTDOSE provided – using synthetic dose (CT HU rescaled)")
            # Synthetic: rescale HU linearly to 0–70 Gy for demonstration
            dose_norm = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min() + 1e-9)
            dose_array = (dose_norm * 70.0).astype(np.float32)

        # ---- 4. Build masks & DVHs ------------------------------------
        dvh_data   = {}
        all_stats  = {}
        all_masks  = {}

        for struct in structures:
            name = struct["name"]

            # Filter by user-provided list (case-insensitive partial match)
            if self.structures_to_include is not None:
                if not any(inc.lower() in name.lower()
                           for inc in self.structures_to_include):
                    continue

            log.info("Processing structure: %s", name)

            # Rasterise contours
            mask = contours_to_mask(struct, ct_meta)
            n_vox = int(mask.sum())

            if n_vox == 0:
                log.warning("  → empty mask (0 voxels), skipping")
                continue

            log.info("  → mask has %d voxels  (%.2f cc)", n_vox, n_vox * voxel_vol_cc)

            # DVH
            dose_bins, vol_pct = compute_dvh(dose_array, mask, self.n_dvh_bins)
            dvh_data[name]  = (dose_bins, vol_pct)
            all_masks[name] = mask

            # Dose statistics
            stats = compute_dose_statistics(dose_array, mask, voxel_vol_cc)
            all_stats[name] = stats

            self._log_stats(name, stats)

        # ---- 5. Save outputs ------------------------------------------
        if dvh_data:
            self._save_individual_dvh_csvs(dvh_data)
            plot_dvhs(
                dvh_data,
                os.path.join(self.output_dir, "dvh_all_structures.png"),
            )
            save_statistics_csv(
                all_stats,
                os.path.join(self.output_dir, "dose_statistics.csv"),
            )

        if all_masks:
            plot_dose_overlay(
                ct_array, dose_array, all_masks,
                os.path.join(self.output_dir, "dose_ct_overlay.png"),
            )

        log.info("Done.  Results in: %s", self.output_dir)
        return dvh_data, all_stats

    # ------------------------------------------------------------------ #

    def _save_individual_dvh_csvs(self, dvh_data: dict):
        import csv
        for name, (d_bins, v_pct) in dvh_data.items():
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
            path = os.path.join(self.output_dir, f"dvh_{safe}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dose_Gy", "volume_pct"])
                for d, v in zip(d_bins, v_pct):
                    writer.writerow([f"{d:.5f}", f"{v:.4f}"])

    @staticmethod
    def _log_stats(name: str, stats: dict):
        log.info(
            "  Stats  Dmean=%.2f  Dmin=%.2f  Dmax=%.2f  D2=%.2f  "
            "D50=%.2f  D98=%.2f  Vol=%.2f cc",
            stats.get("Dmean_Gy", np.nan),
            stats.get("Dmin_Gy",  np.nan),
            stats.get("Dmax_Gy",  np.nan),
            stats.get("D2_Gy",    np.nan),
            stats.get("D50_Gy",   np.nan),
            stats.get("D98_Gy",   np.nan),
            stats.get("volume_cc", np.nan),
        )


# ===========================================================================
# CLI
# ===========================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Load DICOM CT + RTSTRUCT (+ optional RTDOSE) and compute DVHs.",
#         formatter_class=argparse.RawTextHelpFormatter,
#     )
#     parser.add_argument("--ct_dir",    required=True, help="Directory containing CT DICOM files")
#     parser.add_argument("--rtstruct",  required=True, help="Path to RTSTRUCT DICOM file")
#     parser.add_argument("--rtdose",    default=None,  help="Path to RTDOSE DICOM file (optional)")
#     parser.add_argument("--output_dir", default="dvh_output", help="Output directory (default: dvh_output)")
#     parser.add_argument(
#         "--structures", default=None, nargs="+",
#         help="Subset of structure names to include (partial, case-insensitive match).\n"
#              "If omitted, all structures are processed.",
#     )
#     parser.add_argument("--dvh_bins", type=int, default=500, help="Number of DVH dose bins (default: 500)")

#     args = parser.parse_args()

#     calc = RTStructDVHCalculator(
#         ct_dir              = args.ct_dir,
#         rtstruct_path       = args.rtstruct,
#         rtdose_path         = args.rtdose,
#         output_dir          = args.output_dir,
#         structures_to_include = args.structures,
#         n_dvh_bins          = args.dvh_bins,
#     )
#     calc.run()


# if __name__ == "__main__":
#     main()