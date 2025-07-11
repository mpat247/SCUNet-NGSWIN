#!/usr/bin/env python3
# inspect_sindeep_dataset.py
# ------------------------------------------------------------
# Fast, parallel audit of a SynDeepLesion (or any) dataset tree
# ------------------------------------------------------------
import os, sys, csv, argparse, logging, math, multiprocessing as mp
from datetime import datetime
from pathlib import Path
from collections import Counter
import humanize
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

# ---------- constants -----------------------------------------------------
CSV_HEADER = ("path", "shape", "ndim", "dtype",
              "min", "mean", "max", "size_MB", "h5_keys")
SENTINEL   = "__FIN__"
SUP_EXT    = {".h5", ".hdf5", ".npy",
              ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------- small helpers -------------------------------------------------
def first_dataset_key(h5: h5py.File):
    for k in h5.keys():
        if isinstance(h5[k], h5py.Dataset):
            return k
    raise KeyError("HDF5 has no dataset at root")

def load_numpy(path: Path):
    suf = path.suffix.lower()
    if suf in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            k0 = "image" if "image" in f       else \
                 "sinogram" if "sinogram" in f else \
                 first_dataset_key(f)
            return f[k0][:], ";".join(keys)
    if suf == ".npy":
        return np.load(path), ""
    if suf in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return np.asarray(Image.open(path)), ""
    raise ValueError(f"Unsupported file type '{suf}'")

def save_preview(arr: np.ndarray, out_png: Path):
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] >= 20:
        img = arr[0]          # (V,H,W) → slice 0
    elif arr.ndim in {2, 3}:
        img = arr
    else:
        return                # skip 1-D or >3-D tensors
    plt.imsave(out_png, img, cmap="gray" if img.ndim == 2 else None,
               vmin=float(img.min()), vmax=float(img.max()))

# ----------   CSV writer process  ----------------------------------------
def writer_worker(csv_path: Path, q: "mp.Queue"):
    """Pull rows off the queue and append to CSV until SENTINEL."""
    with open(csv_path, "w", newline="") as cf:
        wr = csv.writer(cf);  wr.writerow(CSV_HEADER)
        while True:
            rec = q.get()
            if rec == SENTINEL:
                break
            wr.writerow(rec)

# ----------   worker that inspects one file ------------------------------
def inspect_file(path: Path):
    """Return row-tuple for CSV, or ('__ERR__', path, msg) on failure."""
    try:
        arr, keys = load_numpy(path)
    except Exception as e:
        return ("__ERR__", str(path), str(e))

    stats  = (float(arr.min()), float(arr.mean()), float(arr.max()))
    mb     = round(path.stat().st_size / 1024**2, 3)
    return (str(path), str(arr.shape), arr.ndim, str(arr.dtype),
            *[f"{x:.5g}" for x in stats], mb, keys)

# ----------   main --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Blazing-fast dataset audit")
    ap.add_argument("--root", required=True,
                    help="root holding HDF5 / NPY / image files")
    ap.add_argument("--quick_looks", type=int, default=10,
                    help="how many preview PNGs to save")
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()),
                    help="# concurrent processes (default: %(default)s)")
    args = ap.parse_args()

    # pick sensible start-method
    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)
    else:
        # fork is a tiny bit faster on POSIX, spawn fallback when unavailable
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            mp.set_start_method("spawn", force=True)

    out_dir  = Path("dataset_audit_results_2");  out_dir.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"audit_{ts}.log"
    csv_path = out_dir / "summary.csv"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])
    log = logging.getLogger("audit")

    log.info("Root        : %s", args.root)
    log.info("Quick-looks : %d", args.quick_looks)
    log.info("Workers     : %d", args.workers)

    # discover files
    files = [Path(dp)/fn
             for dp,_,fns in os.walk(args.root)
             for fn in fns if Path(fn).suffix.lower() in SUP_EXT]
    if not files:
        log.error("No supported files found under %s", args.root)
        return
    files.sort()
    log.info("Files found : %d", len(files))

    # CSV writer process
    q     = mp.Queue(maxsize=args.workers*4)
    csv_p = mp.Process(target=writer_worker, args=(csv_path, q), daemon=True)
    csv_p.start()

    # inspection pool
    with mp.Pool(args.workers) as pool:
        shape_hist  = Counter(); ndim_hist = Counter(); bytes_total=0

        for idx, row in enumerate(pool.imap_unordered(inspect_file, files), 1):
            if row[0] == "__ERR__":
                log.error("READ FAIL: %s  (%s)", row[1], row[2])
                continue

            q.put(row)

            shape_hist[row[1]] += 1
            ndim_hist[row[2]]  += 1
            bytes_total += float(row[6]) * 1024**2  # MB → bytes

            # quick looks
            if idx <= args.quick_looks:
                arr, _ = load_numpy(Path(row[0]))
                save_preview(arr, out_dir / f"quicklook_{idx:02}.png")

            if idx % 500 == 0 or idx == len(files):
                log.info("Processed %d / %d", idx, len(files))

    # tell writer to finish, wait for it
    q.put(SENTINEL);  csv_p.join()

    # -------------- summary ------------------------------------------------
    log.info("\n=== SUMMARY ===")
    log.info("Total files : %d", len(files))
    log.info("Total size  : %s", humanize.naturalsize(bytes_total, binary=True))

    log.info("\nTop shapes:")
    for sh, cnt in shape_hist.most_common(10):
        log.info("  %-20s  %d", sh, cnt)

    log.info("\nndim histogram:")
    for nd, cnt in ndim_hist.items():
        log.info("  ndim=%d : %d", nd, cnt)

    log.info("\nCSV saved   : %s", csv_path.resolve())
    log.info("Quicklooks  : %s", out_dir.resolve())
    log.info("=== audit complete ===")

    print("\nRun finished ✓  –  CSV + PNGs in:", out_dir.resolve())
    print("Example command next time:")
    print(f"  python {Path(__file__).name} --root {args.root} --workers {args.workers}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
