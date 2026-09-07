# Bearing Fault Diagnosis via Few-Shot Learning with GPT-2

Classifies rolling-element bearing faults (Normal / Inner Race / Outer Race /
Ball) from vibration signals, using a text-normalized feature representation
fed into a selectively fine-tuned GPT-2 backbone under few-shot (N examples
per class) conditions.

## Pipeline

```
data/raw/*.csv (raw vibration signals)
        │
        ▼
 text_normalization.py   — EMD + statistical features → text sentences
        │
        ▼
data/processed/ae_features_text_rpm.txt  (+ ae_features_numeric.csv)
        │
        ├──► Baseline_svm_nshot.py   — classical SVM baseline, N-shot eval
        │
        └──► gpt2_kway_nshot.py      — GPT-2 backbone, selective fine-tuning,
                                        N-shot K-way classification
```

### 1. `text_normalization.py` — signal → text feature pipeline

For each of the 12 raw files (4 fault classes × 3 rotational speeds):

1. Loads the raw vibration signal and splits it into 40 equal windows.
2. Runs Empirical Mode Decomposition (EMD) on each window and sums the first
   3 intrinsic mode functions (the lowest-order components).
3. Splits that summed signal into 5 sub-windows and computes RMS, peak, and
   kurtosis per sub-window, then derives 7 statistical features (RMS
   average/stdev, their sum, kurtosis average/stdev, peak average/stdev).
4. **Text-normalizes** every feature value into an English sentence (e.g.
   `"The rms average is zero point one two four one."`), plus one sentence
   for rotational speed and one stating the class label — producing one
   multi-sentence text sample per window, ready to feed to GPT-2's tokenizer.

Outputs both the numeric feature table (`ae_features_numeric.csv`, useful
for the SVM baseline / your own analysis) and the text-normalized dataset
(`ae_features_text_rpm.txt`, one sample per line) that both downstream
scripts consume.

**Already run for you**: `data/processed/` contains the output of this
step. Re-run `text_normalization.py` only if you want to regenerate it or
change its parameters — note it's slow (EMD is computationally expensive
across 12 files × 40 windows).

### 2. `Baseline_svm_nshot.py` — classical ML baseline

Parses the text-normalized dataset back into numeric feature vectors (via
regex + word-to-number parsing) and evaluates an RBF-kernel SVM under N-shot
conditions: for each class, train on N randomly drawn examples, test on the
rest, repeated across multiple runs with a reported confidence interval.
Exists to answer the same question as the WiFi project's RandomForest
baseline: how much of the achievable accuracy comes from the feature
engineering itself, independent of using an LLM at all.

### 3. `gpt2_kway_nshot.py` — GPT-2 backbone, selective fine-tuning

The main experiment. For each text sample:

1. Tokenizes with GPT-2's tokenizer.
2. Runs it through a pretrained `GPT2Model` backbone with a configurable
   fine-tuning mode (`--mode`):
   - `frozen` — entire GPT-2 backbone frozen; only the head trains.
   - `attention` (default) — each transformer block's attention weights and
     both layer norms (`ln_1`, `ln_2`), plus the final `ln_f`, are trainable;
     everything else (including the MLP blocks) stays frozen. This is the
     same selective fine-tuning principle used in the WiFi localization
     project's Gemma backbone: adapt how information flows and gets
     normalized, without retraining the feed-forward blocks that hold most
     of the model's pretrained parameters.
   - `full` — every GPT-2 parameter is trainable.
3. Pools the hidden state at each sequence's last real (non-padding) token.
4. Classifies via a small feed-forward head into one of the 4 fault classes,
   using either a plain linear layer or a cosine-similarity classifier
   (`--head cosine`, the default — more stable in the few-shot regime).
5. Trains with either plain cross-entropy or an entropy-regularized loss
   (`--loss entropy`, the default) that penalizes overconfident predictions
   to curb overfitting on very small support sets.
6. Evaluates N-shot, K-way classification accuracy across multiple random
   seeds (`--runs`), reporting mean accuracy with a confidence interval,
   per-class recall, and a run-averaged confusion matrix.

Also supports three evaluation splits (`--split`):
- `pooled` — shots drawn from all rotational speeds mixed together.
- `cross_speed` — train on some speeds, test on a fully held-out speed
  (tests generalization to an unseen operating condition).
- `per_speed` — independent run at each speed in isolation.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# .env already points at ./data/raw and ./data/processed by default —
# only change it if your data lives elsewhere.
```

GPT-2's pretrained weights download from Hugging Face on first run of
`gpt2_kway_nshot.py` — no license gate for GPT-2 (unlike TabPFN/Gemma in the
WiFi project), but it does need network access to huggingface.co.

## Run

```bash
# Step 1 (already done — outputs are included in data/processed/):
python text_normalization.py

# Classical baseline:
python Baseline_svm_nshot.py --nshot_list 5,10,20,30 --runs 10

# Main GPT-2 experiment, single N-shot config:
python gpt2_kway_nshot.py --n_shot 10 --mode attention --head cosine --loss entropy

# Sweep across several N values, with per-class recall + confusion matrices:
python gpt2_kway_nshot.py --nshot_sweep --nshot_list 5,10,20,30

# Test generalization to an unseen operating speed:
python gpt2_kway_nshot.py --split cross_speed --test_rpm 800 --n_shot 10
```

## Verified in this pass

- `text_normalization.py`'s feature extraction and text-generation functions
  were re-run on the reorganized file paths and produce **bit-identical**
  output to the already-included `data/processed/` files — confirming the
  path reorganization didn't change any behavior.
- `Baseline_svm_nshot.py` runs end-to-end against the processed dataset.
- `gpt2_kway_nshot.py`'s CLI and imports were verified; the actual GPT-2
  training run itself was not executed in this pass (needs network access
  to Hugging Face and is meant to run on your machine/Colab, same as the
  WiFi project's LLM fine-tuning script).

## Dataset

Jiangnan University (JNU) bearing dataset (https://github.com/ClarkGableWang/JNU-Bearing-Dataset):
vibration signals from a PCB MA352A60 accelerometer, 50 kHz sampling rate,
at 600/800/1000 rpm, with induced outer race, inner race, and rolling
element (ball) faults plus a normal baseline — 12 raw files, 480
text-normalized samples (40 windows × 12 files) after processing.
