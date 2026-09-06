import os, re, argparse
import numpy as np

# Load configuration (DATA_PATH) from a .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC                       # <-- the one algorithm used
from sklearn.metrics import confusion_matrix, accuracy_score

# To use a different classifier, replace this single function's return value, e.g.:
#   from sklearn.linear_model import LogisticRegression -> LogisticRegression(max_iter=1000)
#   from sklearn.neighbors import KNeighborsClassifier  -> KNeighborsClassifier(n_neighbors=3)
#   from sklearn.ensemble import RandomForestClassifier -> RandomForestClassifier(n_estimators=200)
def make_model():
    return SVC(C=10, gamma='scale')               # RBF kernel (default)

MODEL_NAME = "SVM-rbf"

T_DF = {1:6.314, 2:2.920, 3:2.353, 4:2.132, 5:2.015,
        6:1.943, 7:1.895, 8:1.860, 9:1.833, 10:1.812}

LABEL = re.compile(r"belongs to the (.+?) class", re.I)
PATS = [
    r"rms average is (.+?)\.",
    r"rms standard deviation is (.+?)\.",
    r"sum of rms average and rms standard deviation is (.+?)\.",
    r"kurtosis average is (.+?)\.",
    r"kurtosis standard deviation is (.+?)\.",
    r"peak average is (.+?)\.",
    r"peak standard deviation is (.+?)\.",
]

# ---- spelled-out-number -> float ("thirty-one point six seven" -> 31.67)
_UNITS = {w: i for i, w in enumerate(
    ['zero','one','two','three','four','five','six','seven','eight','nine','ten',
     'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen',
     'eighteen','nineteen'])}
_TENS = {'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,
         'eighty':80,'ninety':90}

def _words_to_int(s):
    total = cur = 0
    for tok in re.split(r'[\s\-]+', s.strip()):
        if tok in _UNITS:   cur += _UNITS[tok]
        elif tok in _TENS:  cur += _TENS[tok]
        elif tok == 'hundred':  cur = (cur or 1) * 100
        elif tok == 'thousand': total += (cur or 1) * 1000; cur = 0
    return total + cur

def _words_to_float(s):
    s = s.strip()
    if 'point' in s:
        ip, dp = s.split('point')
        ipart = _words_to_int(ip) if ip.strip() else 0
        digits = ''.join(str(_UNITS[w]) for w in dp.split() if w in _UNITS)
        return float(f"{ipart}.{digits}") if digits else float(ipart)
    return float(_words_to_int(s))


def load_features(path):
    X, y = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = LABEL.search(s)
            if not m:
                continue
            row, ok = [], True
            for p in PATS:
                mm = re.search(p, s, re.I)
                if not mm:
                    ok = False; break
                row.append(_words_to_float(mm.group(1)))
            if ok:
                X.append(row); y.append(m.group(1).strip())
    X = np.array(X, dtype=float); y = np.array(y)
    classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[c] for c in y])
    print(f"[INFO] Parsed {len(X)} samples, {X.shape[1]} features | classes: {classes}")
    return X, yi, classes


def _ci(vals, n):
    if n <= 1: return 0.0
    return T_DF.get(n - 1, 1.812) * float(np.std(vals, ddof=1) / np.sqrt(n))


def nshot_eval(X, yi, classes, n_shot, runs, seed=1):
    K = len(classes)
    rng = np.random.default_rng(seed)
    accs, conf_sum = [], np.zeros((K, K))
    for _ in range(runs):
        sup, qry = [], []
        for c in range(K):
            idx = np.where(yi == c)[0].copy(); rng.shuffle(idx)
            kpool = int(len(idx) * 0.80)
            pool, val = idx[:kpool], idx[kpool:]
            if n_shot > len(pool):
                raise ValueError(f"N={n_shot} exceeds pool {len(pool)} for class {classes[c]}")
            sup += list(rng.choice(pool, n_shot, replace=False))
            qry += list(val)
        sup, qry = np.array(sup), np.array(qry)
        sc = StandardScaler().fit(X[sup])
        clf = make_model().fit(sc.transform(X[sup]), yi[sup])
        pred = clf.predict(sc.transform(X[qry]))
        accs.append(accuracy_score(yi[qry], pred) * 100)
        conf_sum += confusion_matrix(yi[qry], pred, labels=list(range(K)))
    conf_avg = conf_sum / runs
    recalls = np.divide(np.diag(conf_avg), conf_avg.sum(1),
                        out=np.zeros(K), where=conf_avg.sum(1) > 0) * 100
    return float(np.mean(accs)), _ci(accs, runs), recalls, conf_avg


def _short(c): return c.split()[0]

def print_conf(conf_avg, classes):
    K = len(classes); shorts = [_short(c) for c in classes]
    w = max(6, max(len(s) for s in shorts))
    print("        " + "".join(f"{s:>{w+1}}" for s in shorts) + "    (cols=pred)")
    for i, s in enumerate(shorts):
        print(f"  {s:>6}" + "".join(f"{int(round(conf_avg[i, j])):>{w+1}}" for j in range(K)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.getenv('DATA_PATH',
                    '/content/drive/MyDrive/colab dataset/ae_features_text_rpm.txt'),
                    help="dataset path; defaults to DATA_PATH from the .env file")
    ap.add_argument('--nshot_list', default='5,10,20,30')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--show_conf', action='store_true')
    args = ap.parse_args()

    X, yi, classes = load_features(args.data)
    K = len(classes); shorts = [_short(c) for c in classes]
    max_n = int(min(np.bincount(yi)) * 0.80)
    n_list = [n for n in (int(x) for x in args.nshot_list.split(',')) if n <= max_n]

    print(f"\n{'='*78}\n  {K}-way N-shot BASELINE — {MODEL_NAME}  (runs={args.runs})\n{'='*78}")
    hdr = f"  {'N':>3} | {'Val acc':>15} | " + "  ".join(f"{s:>7}" for s in shorts) + \
          "   (per-class recall %)"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    conf_store = {}
    for n in n_list:
        acc, marg, rec, conf = nshot_eval(X, yi, classes, n, args.runs)
        conf_store[n] = conf
        print(f"  {n:>3} | {acc:6.2f} +/- {marg:4.2f} | " +
              "  ".join(f"{v:7.1f}" for v in rec))

    if args.show_conf:
        print(f"\n  Run-averaged confusion matrices (rows=true):")
        for n in n_list:
            print(f"\n  --- N = {n} ---")
            print_conf(conf_store[n], classes)
    print(f"\n{'='*78}")


if __name__ == '__main__':
    main()