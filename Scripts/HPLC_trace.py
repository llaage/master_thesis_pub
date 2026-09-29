import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def parse_hplc_folder(root_dir):
    """
    Parse an Agilent HPLC folder with fractions + UV chromatograms.
    """
    frac_path = os.path.join(root_dir, "fractions.csv")
    if not os.path.exists(frac_path):
        raise FileNotFoundError(f"fractions.csv not found in {root_dir}")

    fractions = {}
    Abs = {}
    record = False

    with open(frac_path, "r") as rf:
        for line in rf:
            line = line.strip().strip("'\"")

            if line.startswith("Sample Vial Number,"):
                parts = line.split(",")
                trace_id = parts[1]
                flow_rate_val = float(parts[3].split()[0])
                injection_vol_val = float(parts[5])

            if line.startswith("Yes,Signal"):
                parts = line.split(",")
                signal = parts[1].split()[-1]
                wl = parts[2].split()[0]
                Abs[signal] = wl

            if line.startswith("#,Location"):
                record = True
                fraclist = []

            if line.startswith("Reference Detector ="):
                record = False
                fraclist.append(flow_rate_val)
                fraclist.append(injection_vol_val)
                fractions[trace_id] = fraclist

            if record and not line.startswith("#,Location"):
                fraclist.append(line.split(","))

    vol_data = {f"vol{v}": [] for v in Abs.values()}
    Abs_data = {f"A{v}": [] for v in Abs.values()}
    other_data = {"trace_id": [], "fractions": [], "frac_vol": [], "frac_start_stop": [], "flow_rate": []}
    sec_data = {**vol_data, **Abs_data, **other_data}

    subfolders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for folder in sorted(subfolders):
        trace_id = os.path.basename(folder)
        if trace_id not in fractions:
            print(f"Warning: {trace_id} not in fractions.csv → skipping.")
            continue

        trace_info = fractions[trace_id]
        flow_rate = trace_info[-2]

        frac_info = np.array(trace_info[:-2])
        frac_id = frac_info[:, 1]
        frac_start = frac_info[:, 2].astype(float) * flow_rate
        frac_stop = frac_info[:, 3].astype(float) * flow_rate

        sec_data["trace_id"].append(trace_id)
        sec_data["fractions"].append(frac_id)
        sec_data["frac_vol"].append(frac_stop - frac_start)
        sec_data["frac_start_stop"].append(list(zip(frac_start, frac_stop)))
        sec_data["flow_rate"].append(flow_rate)

        for trace in glob.glob(os.path.join(folder, "*.CSV")):
            fname = os.path.basename(trace)

            key = None
            if "DAD1" in fname:
                key = fname.split("DAD1")[-1].replace(".CSV", "")
            elif "MWD1" in fname:
                key = fname.split("MWD1")[-1].replace(".CSV", "")
            elif "FLD1" in fname:
                key = fname.split("FLD1")[-1].replace(".CSV", "")

            if key is None or key not in Abs:
                continue

            wl = Abs[key]
            chroma = pd.read_csv(trace, names=["time", "Abs"])
            chroma["vol"] = chroma["time"] * flow_rate

            sec_data[f"A{wl}"].append(chroma["Abs"].to_numpy())
            sec_data[f"vol{wl}"].append(chroma["vol"].to_numpy())

    return pd.DataFrame.from_dict(sec_data)




def plot_traces_from_sec_df(sec_df, wavelength=280, save_dir=None):
    """
    Plot HPLC traces + fraction windows for each trace.
    If save_dir is provided, save plots instead of showing them.
    """
    wl = str(wavelength)
    vol_col = f"vol{wl}"
    Abs_col = f"A{wl}"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for _, r in sec_df.iterrows():
        vol_val = r[vol_col]
        Abs_val = r[Abs_col]

        if not isinstance(vol_val, (np.ndarray, list)):
            continue

        vol = np.asarray(vol_val)
        Abs = np.asarray(Abs_val)
        trace_id = r.get("trace_id", "unknown_trace")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(vol, Abs, alpha=0.9)

        windows = r.get("frac_start_stop", None)
        names = r.get("fractions", None)

        if isinstance(windows, list):
            ymax = float(np.nanmax(Abs)) if Abs.size else 1.0
            for (w, name) in zip(windows, names):
                start, stop = w
                ax.axvspan(start, stop, color="gray", alpha=0.1)
                ax.text((start + stop)/2, ymax * 0.05, str(name),
                        rotation=90, ha="center", va="bottom", fontsize=8)

        ax.set_title(trace_id)
        ax.set_xlabel("Elution volume (mL)")
        ax.set_ylabel(f"A{wl}")
        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            outpath = os.path.join(save_dir, f"{trace_id}_A{wl}.png")
            fig.savefig(outpath, dpi=300)
            plt.close(fig)
            print(f"Saved: {outpath}")

 

if __name__ == "__main__":
    hplc_folder = "/home/llaage/wetlab/purifications/260831_rh5_ako_lowsalt/HPLC"
    output_folder = "/home/llaage/wetlab/purifications/260831_rh5_ako_lowsalt/HPLC"

    sec_df = parse_hplc_folder(hplc_folder)

    # show plots:
    #plot_traces_from_sec_df(sec_df, wavelength=280)

    # save plots:
    plot_traces_from_sec_df(sec_df, wavelength=280, save_dir=output_folder)
