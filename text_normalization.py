import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from num2words import num2words
from PyEMD import EMD
from dotenv import load_dotenv


# CONFIGURATION

# Load variables from a .env file sitting next to this script.
# Change DATA_DIR / OUT_DIR in the .env file — never in this code.
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
OUT_DIR = os.getenv("OUT_DIR")

if not DATA_DIR or not OUT_DIR:
    raise RuntimeError(
        "DATA_DIR and/or OUT_DIR are not set. Create a .env file next to "
        "this script (see .env.example) with lines like:\n"
        "DATA_DIR=D:\\BIT\\BTECH\\6 th sem\\project\\bearing data\n"
        "OUT_DIR=C:\\Users\\bitd\\OneDrive\\Desktop\\norm"
    )

# Map filename -> (fault label, rpm)
FILES = {
    "ib600_2.csv":  ("Inner Race Fault", 600),
    "ib800_2.csv":  ("Inner Race Fault", 800),
    "ib1000_2.csv": ("Inner Race Fault", 1000),

    "ob600_2.csv":  ("Outer Race Fault", 600),
    "ob800_2.csv":  ("Outer Race Fault", 800),
    "ob1000_2.csv": ("Outer Race Fault", 1000),

    "tb600_2.csv":  ("Ball Fault", 600),
    "tb800_2.csv":  ("Ball Fault", 800),
    "tb1000_2.csv": ("Ball Fault", 1000),

    "n600_3_2.csv":  ("Normal", 600),
    "n800_3_2.csv":  ("Normal", 800),
    "n1000_3_2.csv": ("Normal", 1000),

    # Placeholder for future data -- not present in this dataset yet:
    # "cb600_2.csv": ("Cage Fault", 600),
    # "cb800_2.csv": ("Cage Fault", 800),
    # "cb1000_2.csv": ("Cage Fault", 1000),
}

USABLE_SAMPLES = 500_000    
N_SEGMENTS = 40               
WINDOW_LEN = USABLE_SAMPLES // N_SEGMENTS   
SUB_WINDOWS = 5                
SUB_LEN = WINDOW_LEN // SUB_WINDOWS          
N_IMF_SUM = 3                 
ROUND_DECIMALS = 4              

FEATURE_NAMES = [
    "rms-average",
    "rms-stdev",
    "rms-average + rms-stdev",
    "kurtosis-average",
    "kurtosis-stdev",
    "peak-average",
    "peak-stdev",
]


FEATURE_PHRASES = {
    "rms-average": "The rms average is {value}.",
    "rms-stdev": "The rms standard deviation is {value}.",
    "rms-average + rms-stdev": "The sum of rms average and rms standard deviation is {value}.",
    "kurtosis-average": "The kurtosis average is {value}.",
    "kurtosis-stdev": "The kurtosis standard deviation is {value}.",
    "peak-average": "The peak average is {value}.",
    "peak-stdev": "The peak standard deviation is {value}.",
}


RPM_PHRASE = "The rotational speed is {value} rpm."



# STEP 1: SIGNAL LOADING & WINDOWING


def load_signal(filepath):
    """Load a single-column raw AE signal CSV as a 1D numpy array."""
    data = pd.read_csv(filepath, header=None).values.flatten()
    return data.astype(np.float64)


def make_windows(signal, usable_samples=USABLE_SAMPLES,
                  n_segments=N_SEGMENTS, window_len=WINDOW_LEN):
    """Truncate signal to `usable_samples` and split into equal windows."""
    signal = signal[:usable_samples]
    windows = signal.reshape(n_segments, window_len)
    return windows



# STEP 2: EMD -> SUM OF FIRST N_IMF_SUM IMFs


def summed_imf_signal(window, n_imf_sum=N_IMF_SUM):
    """Run EMD on a window and return the sum of the first `n_imf_sum`
    intrinsic mode components (lowest-order / highest-frequency IMFs)."""
    emd = EMD()
    imfs = emd.emd(window)
    n_use = min(n_imf_sum, imfs.shape[0])
    return imfs[:n_use].sum(axis=0)



# STEP 3: 7 AE FEATURES (Table 1)


def extract_ae_features(summed_signal, sub_windows=SUB_WINDOWS, sub_len=SUB_LEN):
    """Split the summed-IMF signal into sub-windows, compute rms / peak /
    kurtosis for each sub-window, then derive the 7 AE features."""
    rms_vals, peak_vals, kurt_vals = [], [], []

    for i in range(sub_windows):
        seg = summed_signal[i * sub_len:(i + 1) * sub_len]
        rms_vals.append(np.sqrt(np.mean(seg ** 2)))
        peak_vals.append(np.max(np.abs(seg)))
        kurt_vals.append(scipy_kurtosis(seg, fisher=True, bias=False))

    rms_vals = np.array(rms_vals)
    peak_vals = np.array(peak_vals)
    kurt_vals = np.array(kurt_vals)

    rms_average = rms_vals.mean()
    rms_stdev = rms_vals.std(ddof=1)
    rms_sum = rms_average + rms_stdev
    kurtosis_average = kurt_vals.mean()
    kurtosis_stdev = kurt_vals.std(ddof=1)
    peak_average = peak_vals.mean()
    peak_stdev = peak_vals.std(ddof=1)

    return {
        "rms-average": rms_average,
        "rms-stdev": rms_stdev,
        "rms-average + rms-stdev": rms_sum,
        "kurtosis-average": kurtosis_average,
        "kurtosis-stdev": kurtosis_stdev,
        "peak-average": peak_average,
        "peak-stdev": peak_stdev,
    }



# STEP 4: TEXT NORMALIZATION (numbers -> words)


def number_to_words(value, decimals=ROUND_DECIMALS):
    """Convert a float to its English word representation, e.g.
    -1.2345 -> 'negative one point two three four five'
    Handles the integer and fractional parts separately so that, e.g.,
    98.6 -> 'ninety-eight point six' (digit-by-digit decimal part)."""
    value = round(float(value), decimals)
    negative = value < 0
    value = abs(value)

    int_part = int(value)
    frac_part = round(value - int_part, decimals)

    int_words = num2words(int_part)

    if frac_part == 0:
        words = int_words
    else:
        
        frac_str = f"{frac_part:.{decimals}f}".split(".")[1].rstrip("0")
        if frac_str == "":
            words = int_words
        else:
            digit_words = " ".join(num2words(int(d)) for d in frac_str)
            words = f"{int_words} point {digit_words}"

    if negative:
        words = f"negative {words}"

    return words


def features_to_text(feature_dict, label, rpm):
    """Build one text-normalized sample (multi-sentence string) from a
    dict of 7 AE feature values + class label.

    The rotational speed is always appended as one extra normalized
    sentence (word form), placed just before the class sentence. Everything
    else is identical to the paper's methodology — only this one context
    sentence is added.
    """
    sentences = []
    for name in FEATURE_NAMES:
        value_words = number_to_words(feature_dict[name])
        sentences.append(FEATURE_PHRASES[name].format(value=value_words))

    # Rotational speed as a normalized sentence, e.g. 600 -> "six hundred".
    rpm_words = num2words(int(rpm))
    sentences.append(RPM_PHRASE.format(value=rpm_words))

    sentences.append(f"This sample belongs to the {label} class.")
    return " ".join(sentences)



# MAIN PIPELINE


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    numeric_rows = []
    text_lines = []

    for filename, (label, rpm) in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename}  (label={label}, rpm={rpm}) ...")

        signal = load_signal(filepath)
        windows = make_windows(signal)

        for idx, window in enumerate(windows):
            summed = summed_imf_signal(window)
            features = extract_ae_features(summed)

            row = {
                "source_file": filename,
                "label": label,
                "rpm": rpm,
                "segment_index": idx,
            }
            row.update(features)
            numeric_rows.append(row)

            text_lines.append(features_to_text(features, label, rpm))

        print(f"  -> {len(windows)} samples generated")

    
    numeric_df = pd.DataFrame(numeric_rows)
    numeric_csv_path = os.path.join(OUT_DIR, "ae_features_numeric.csv")
    numeric_df.to_csv(numeric_csv_path, index=False)
    print(f"\nSaved numeric features -> {numeric_csv_path}  "
          f"({len(numeric_df)} rows)")

    text_path = os.path.join(OUT_DIR, "ae_features_text_rpm.txt")
    with open(text_path, "w") as f:
        f.write("\n".join(text_lines))
    print(f"Saved text-normalized samples -> {text_path}  "
          f"({len(text_lines)} lines)")

    if text_lines:
        print("\nExample (primary file):")
        print(" ", text_lines[0])


if __name__ == "__main__":
    main()