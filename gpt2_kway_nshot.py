import os, re, random, argparse, warnings
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from transformers.utils import logging as hf_logging
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Keep the console clean: silence HuggingFace info logs and the per-run
# "Loading weights" progress bars so the sweep prints as one tidy table.
hf_logging.set_verbosity_error()
try:
    hf_logging.disable_progress_bar()
except Exception:
    pass

# Load configuration (e.g. DATA_PATH) from a .env file if python-dotenv is present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

LABEL_PAT = re.compile(r"This sample belongs to the (.+?) class\.?", re.IGNORECASE)
RPM_PAT   = re.compile(r"rotational speed is (.+?) rpm", re.IGNORECASE)

# Student-t 0.95 quantiles indexed by df = n_runs - 1 (for the 90% CI margin).
T_DF = {1:6.314, 2:2.920, 3:2.353, 4:2.132, 5:2.015,
        6:1.943, 7:1.895, 8:1.860, 9:1.833, 10:1.812}

# Friendly int <-> the rpm word-strings that actually appear in the file.
RPM_WORD = {600: 'six hundred', 800: 'eight hundred', 1000: 'one thousand'}


# ------------------------------------------------- OPTIONAL number->words (no-op here)
_ONES = ['zero','one','two','three','four','five','six','seven','eight','nine',
         'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
         'seventeen','eighteen','nineteen']
_TENS = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']

def _int_to_words(n):
    if n < 20:  return _ONES[n]
    if n < 100: return _TENS[n//10] + (('-' + _ONES[n%10]) if n%10 else '')
    if n < 1000:
        return _ONES[n//100] + ' hundred' + ((' ' + _int_to_words(n%100)) if n%100 else '')
    if n < 1_000_000:
        return _int_to_words(n//1000) + ' thousand' + ((' ' + _int_to_words(n%1000)) if n%1000 else '')
    return _int_to_words(n//1_000_000) + ' million' + ((' ' + _int_to_words(n%1_000_000)) if n%1_000_000 else '')

_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

def _num_to_words(m):
    tok = m.group(0); neg = tok.startswith('-'); tok = tok.lstrip('-')
    if '.' in tok:
        intp, decp = tok.split('.', 1)
        words = (_int_to_words(int(intp)) if intp else 'zero') + \
                ' point ' + ' '.join(_ONES[int(d)] for d in decp)
    else:
        words = _int_to_words(int(tok))
    return ('negative ' + words) if neg else words

def normalize_numbers_to_words(text):
    return _NUM_RE.sub(_num_to_words, text)


# ------------------------------------------------- DATA
def clean_text(line, normalize_numbers=False):
    txt = LABEL_PAT.sub("", line).strip()
    return normalize_numbers_to_words(txt) if normalize_numbers else txt


def load_dataset(filepath, normalize_numbers=False):
    """Returns list of (text, class, rpm_word), label_map, sorted rpm list."""
    samples = []
    rpm_set = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LABEL_PAT.search(line)
            if not m:
                continue
            cls = m.group(1).strip()
            r = RPM_PAT.search(line)
            rpm = r.group(1).strip().lower() if r else 'unknown'
            rpm_set.add(rpm)
            samples.append((clean_text(line, normalize_numbers), cls, rpm))

    classes = sorted({c for _, c, _ in samples})
    label_map = {lbl: i for i, lbl in enumerate(classes)}

    counts = defaultdict(lambda: defaultdict(int))
    for _, c, r in samples:
        counts[c][r] += 1
    rpm_list = sorted(rpm_set)
    print(f"[INFO] Loaded {len(samples)} samples | K={len(label_map)} classes: {label_map}")
    print(f"[INFO] Speeds present: {rpm_list}")
    print(f"[INFO] Class x speed counts:")
    for c in classes:
        row = "  ".join(f"{r}={counts[c][r]}" for r in rpm_list)
        print(f"         {c:<20} {row}")
    return samples, label_map, rpm_list


# ------------------------------------------------- SPLITTERS
def _draw_shots(pool, n_shot, rng, stratify_rpm):
    """pool = list of (text, rpm). Draw n_shot without replacement."""
    if n_shot > len(pool):
        raise ValueError(f"N={n_shot} exceeds available pool size {len(pool)}.")
    if not stratify_rpm:
        return rng.sample(pool, n_shot)
    groups = defaultdict(list)
    for item in pool:
        groups[item[1]].append(item)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups.keys()); rng.shuffle(order)
    picked, i = [], 0
    while len(picked) < n_shot:
        g = groups[order[i % len(order)]]
        if g:
            picked.append(g.pop())
        i += 1
        if all(len(v) == 0 for v in groups.values()):
            break
    return picked[:n_shot]


def split_pooled(samples, label_map, n_shot, seed, stratify_rpm=False):
    """N shots/class from all speeds mixed; held-out 20%/class -> validation."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for t, c, r in samples:
        by_class[c].append((t, r))
    tr_t, tr_l, val_t, val_l = [], [], [], []
    for cls, items in by_class.items():
        items = items[:]; rng.shuffle(items)
        n_pool = int(len(items) * 0.80)
        pool, val = items[:n_pool], items[n_pool:]
        for t, _ in _draw_shots(pool, n_shot, rng, stratify_rpm):
            tr_t.append(t); tr_l.append(label_map[cls])
        for t, _ in val:
            val_t.append(t); val_l.append(label_map[cls])
    return tr_t, tr_l, val_t, val_l


def split_cross_speed(samples, label_map, n_shot, seed, test_rpm, stratify_rpm=False):
    """Train N-shot/class on speeds != test_rpm; validate on ALL of test_rpm."""
    rng = random.Random(seed)
    train_pool, val_items = defaultdict(list), defaultdict(list)
    for t, c, r in samples:
        (val_items if r == test_rpm else train_pool)[c].append((t, r))
    tr_t, tr_l, val_t, val_l = [], [], [], []
    for cls in label_map:
        pool = train_pool[cls][:]; rng.shuffle(pool)
        for t, _ in _draw_shots(pool, n_shot, rng, stratify_rpm):
            tr_t.append(t); tr_l.append(label_map[cls])
        for t, _ in val_items[cls]:
            val_t.append(t); val_l.append(label_map[cls])
    return tr_t, tr_l, val_t, val_l


class BearingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.labels = labels
        self.enc = tokenizer(texts, padding='max_length', truncation=True,
                             max_length=max_length, return_tensors='pt')
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {'input_ids': self.enc['input_ids'][i],
                'attention_mask': self.enc['attention_mask'][i],
                'label': torch.tensor(self.labels[i], dtype=torch.long)}


# ------------------------------------------------- LOSS (Eq. 3)
class EntropyRegularizedLoss(nn.Module):
    """
    L = CE - beta * H(p),  H(p) = -sum_j p_j log p_j  (mean over batch).
    beta > 0 is a confidence penalty (rewards softer outputs) -> curbs overfitting,
    matching the paper's stated goal. Pass negative beta for the literal printed
    Eq. 3 sign; beta = 0 recovers plain CE (the paper's "old loss").
    """
    def __init__(self, beta=0.1):
        super().__init__(); self.beta = beta; self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        logp = F.log_softmax(logits, dim=-1); p = logp.exp()
        entropy = -(p * logp).sum(dim=-1).mean()
        return ce - self.beta * entropy


# ------------------------------------------------- MODEL
class CosineClassifier(nn.Module):
    """p = softmax(scale * cos(x, W) + b)  (Eq. 4)."""
    def __init__(self, in_features, num_classes, init_scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.bias   = nn.Parameter(torch.zeros(num_classes))
        self.scale  = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x):
        return self.scale * (F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).T) + self.bias


class GPT2Classifier(nn.Module):
    def __init__(self, num_classes, mode='attention', head='cosine', dropout=0.1, verbose=True):
        super().__init__()
        self.mode = mode; self.head = head
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        for p in self.gpt2.parameters():
            p.requires_grad = False
        if mode == 'full':
            for p in self.gpt2.parameters(): p.requires_grad = True
        elif mode == 'attention':
            for block in self.gpt2.h:
                for p in block.attn.parameters(): p.requires_grad = True
                for p in block.ln_1.parameters(): p.requires_grad = True
                for p in block.ln_2.parameters(): p.requires_grad = True
            for p in self.gpt2.ln_f.parameters(): p.requires_grad = True
        # mode == 'frozen' -> all frozen

        dim = self.gpt2.config.hidden_size
        self.fc  = nn.Linear(dim, 768)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.classifier = (CosineClassifier(768, num_classes) if head == 'cosine'
                           else nn.Linear(768, num_classes))
        if verbose:
            total     = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[MODEL] mode={mode} head={head} | trainable {trainable:,}/{total:,} "
                  f"({100*trainable/total:.1f}%)")

    def forward(self, input_ids, attention_mask):
        out    = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        seq_len = attention_mask.sum(dim=1) - 1
        idx     = torch.arange(hidden.size(0), device=hidden.device)
        pooled  = hidden[idx, seq_len]
        return self.classifier(self.drop(self.act(self.fc(pooled))))


# ------------------------------------------------- TRAIN / EVAL
def train_epoch(model, loader, optimizer, criterion):
    model.train(); correct = total = 0
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labs = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, labs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct += (logits.argmax(-1) == labs).sum().item(); total += labs.size(0)
    return 100 * correct / total


@torch.no_grad()
def eval_acc(model, loader):
    model.eval(); correct = total = 0
    preds_all, true_all = [], []
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labs = batch['label'].to(DEVICE)
        preds = model(ids, mask).argmax(-1)
        correct += (preds == labs).sum().item(); total += labs.size(0)
        preds_all.extend(preds.cpu().numpy()); true_all.extend(labs.cpu().numpy())
    return 100 * correct / total, preds_all, true_all


def _ci_margin(values, n_runs):
    if n_runs <= 1: return 0.0
    t = T_DF.get(n_runs - 1, 1.812)
    return t * float(np.std(values, ddof=1) / np.sqrt(n_runs))


def make_criterion(loss_name, beta):
    return nn.CrossEntropyLoss() if loss_name == 'ce' else EntropyRegularizedLoss(beta=beta)


def _make_split(samples, label_map, n_shot, seed, args, test_rpm=None):
    if args.split == 'cross_speed':
        return split_cross_speed(samples, label_map, n_shot, seed,
                                 RPM_WORD.get(test_rpm, test_rpm), args.stratify_rpm)
    return split_pooled(samples, label_map, n_shot, seed, args.stratify_rpm)


# ------------------------------------------------- ONE (K, N) EXPERIMENT
def run_nshot(samples, label_map, args, tokenizer, n_shot, loss_name,
              test_rpm=None, keep_best_state=False, verbose=False):
    criterion = make_criterion(loss_name, args.entropy_beta)
    val_bests, tr_bests = [], []
    best_val, best_state, best_split = -1.0, None, None
    K = len(label_map)
    conf_sum = np.zeros((K, K), dtype=float)      # accumulated best-epoch confusion

    for run in range(args.runs):
        run_seed = SEED + run
        torch.manual_seed(run_seed); np.random.seed(run_seed); random.seed(run_seed)

        tr_t, tr_l, val_t, val_l = _make_split(samples, label_map, n_shot, run_seed,
                                               args, test_rpm=test_rpm)
        if run == 0 and verbose:
            print(f"    [SPLIT] {args.split}/{loss_name} | N={n_shot} "
                  f"| train {len(tr_t)} ({n_shot}/class) | val {len(val_t)}")

        model = GPT2Classifier(len(label_map), mode=args.mode, head=args.head,
                               verbose=(run == 0 and verbose)).to(DEVICE)
        # Two LR groups: the randomly-initialized head needs a much higher LR to
        # fit the tiny support set; the pre-trained backbone stays gentle.
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('gpt2.')]
        base_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith('gpt2.')]
        opt = torch.optim.AdamW(
            [{'params': base_params, 'lr': args.lr},
             {'params': head_params, 'lr': args.head_lr}],
            betas=(0.9, 0.999), eps=1e-8)
        tr_loader  = DataLoader(BearingDataset(tr_t, tr_l, tokenizer, args.max_length),
                                batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(BearingDataset(val_t, val_l, tokenizer, args.max_length),
                                batch_size=args.batch_size)

        best_run_val, best_run_tr = -1.0, 0.0
        best_run_preds, best_run_true = None, None
        for epoch in range(1, args.epochs + 1):
            tr = train_epoch(model, tr_loader, opt, criterion)
            va, preds, true = eval_acc(model, val_loader)
            if va > best_run_val:
                best_run_val, best_run_tr = va, tr
                best_run_preds, best_run_true = preds, true
                if keep_best_state and va > best_val:
                    best_val, best_split = va, (val_t, val_l)
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"      run {run+1} ep {epoch:3d}/{args.epochs} "
                      f"| train {tr:5.1f}% | val {va:5.1f}%")
        val_bests.append(best_run_val); tr_bests.append(best_run_tr)
        conf_sum += confusion_matrix(best_run_true, best_run_preds, labels=list(range(K)))
        if verbose:
            print(f"    -> run {run+1}/{args.runs}: best val {best_run_val:.2f}%")

    conf_avg = conf_sum / args.runs                         # mean confusion per run
    recalls = np.divide(np.diag(conf_avg), conf_avg.sum(1),
                        out=np.zeros(K), where=conf_avg.sum(1) > 0) * 100
    return dict(
        n_shot=n_shot, loss=loss_name, test_rpm=test_rpm,
        val_mean=float(np.mean(val_bests)), val_margin=_ci_margin(val_bests, args.runs),
        tr_mean=float(np.mean(tr_bests)),   tr_margin=_ci_margin(tr_bests, args.runs),
        best_val=best_val if keep_best_state else max(val_bests),
        best_state=best_state, best_split=best_split,
        conf_avg=conf_avg, recalls=recalls)


# ------------------------------------------------- SWEEP (Table 3 style)
def run_sweep(samples, label_map, args, tokenizer, test_rpm=None):
    K = len(label_map)
    tag = f"split={args.split}" + (f", held-out {test_rpm}rpm" if test_rpm else "")
    print(f"\n{'='*74}\n  {K}-way N-shot SWEEP  ({tag}, runs={args.runs})\n{'='*74}")
    header = (f"  {'N':>2} | {'Old loss Val':>16} {'Old loss Train':>16} "
              f"| {'New loss Val':>16} {'New loss Train':>16}")
    print(header); print("  " + "-" * (len(header) - 2))
    for n in range(1, args.n_shot + 1):
        old = run_nshot(samples, label_map, args, tokenizer, n, 'ce', test_rpm=test_rpm)
        new = run_nshot(samples, label_map, args, tokenizer, n, 'entropy', test_rpm=test_rpm)
        print(f"  {n:>2} | "
              f"{old['val_mean']:6.2f} +/- {old['val_margin']:4.2f}   "
              f"{old['tr_mean']:6.2f} +/- {old['tr_margin']:4.2f}   | "
              f"{new['val_mean']:6.2f} +/- {new['val_margin']:4.2f}   "
              f"{new['tr_mean']:6.2f} +/- {new['tr_margin']:4.2f}")
    print(f"{'='*74}")


def _short(name):
    """Compact class label for tables: 'Inner Race Fault' -> 'Inner'."""
    return name.split()[0]


def _print_conf(conf_avg, classes):
    K = len(classes)
    shorts = [_short(c) for c in classes]
    w = max(6, max(len(s) for s in shorts))
    print("        " + "".join(f"{s:>{w+1}}" for s in shorts) + "    (cols=pred)")
    for i, s in enumerate(shorts):
        row = "".join(f"{int(round(conf_avg[i, j])):>{w+1}}" for j in range(K))
        print(f"  {s:>6}{row}")


# ------------------------------------------------- N-SHOT SWEEP (recall + confusion)
def run_nshot_sweep(samples, label_map, args, tokenizer, test_rpm=None):
    """
    Trains at several N values and reports, for each: overall val accuracy,
    per-class recall (so you can watch Ball recall climb), and the run-averaged
    confusion matrix -- the clearest view of whether more shots close the gap.
    """
    K = len(label_map)
    classes = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    shorts = [_short(c) for c in classes]
    n_list = [int(x) for x in args.nshot_list.split(',')]

    # guard against N exceeding the smallest class pool
    per_class = defaultdict(int)
    for _, c, _ in samples:
        per_class[c] += 1
    if args.split == 'cross_speed':
        max_n = 10**9  # depends on held-out speed; let the splitter raise if too big
    else:
        max_n = int(min(per_class.values()) * 0.80)
    n_list = [n for n in n_list if n <= max_n]
    if not n_list:
        print(f"[ERROR] all requested N exceed the pool cap ({max_n})."); return

    tag = f"split={args.split}, loss={args.loss}" + (f", held-out {test_rpm}rpm" if test_rpm else "")
    print(f"\n{'='*78}\n  {K}-way N-SHOT SWEEP  ({tag}, runs={args.runs})\n{'='*78}")
    hdr = f"  {'N':>3} | {'Val acc':>14} | {'Train':>7} | " + \
          "  ".join(f"{s:>7}" for s in shorts) + "   (per-class recall %)"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    results = []
    for n in n_list:
        r = run_nshot(samples, label_map, args, tokenizer, n, args.loss, test_rpm=test_rpm)
        results.append(r)
        rec = "  ".join(f"{v:7.1f}" for v in r['recalls'])
        print(f"  {n:>3} | {r['val_mean']:6.2f} +/- {r['val_margin']:4.2f} "
              f"| {r['tr_mean']:6.1f} | {rec}")

    print(f"\n  Run-averaged confusion matrices (rows=true):")
    for n, r in zip(n_list, results):
        print(f"\n  --- N = {n} ---")
        _print_conf(r['conf_avg'], classes)

    # ---- consolidated summary: accuracy for every N in one table
    print(f"\n{'='*78}")
    print(f"  SUMMARY — accuracy for every N-shot run  ({tag}, runs={args.runs})")
    print(f"{'='*78}")
    sh = f"  {'N':>3} | {'Val acc %':>16} | {'Train acc %':>11} | " + \
         "  ".join(f"{s:>7}" for s in shorts) + "   (per-class recall %)"
    print(sh); print("  " + "-" * (len(sh) - 2))
    for n, r in zip(n_list, results):
        rec = "  ".join(f"{v:7.1f}" for v in r['recalls'])
        print(f"  {n:>3} | {r['val_mean']:6.2f} +/- {r['val_margin']:4.2f}   "
              f"| {r['tr_mean']:11.2f} | {rec}")
    print(f"{'='*78}")


# ------------------------------------------------- FINAL EVALUATION REPORT
def final_report(model, tokenizer, val_t, val_l, label_map, args):
    """Classification report + confusion matrix for the best trained model."""
    inv = {v: k for k, v in label_map.items()}
    classes = [inv[i] for i in range(len(label_map))]
    loader = DataLoader(BearingDataset(val_t, val_l, tokenizer, args.max_length),
                        batch_size=args.batch_size)
    acc, pred_idx, true_idx = eval_acc(model, loader)
    preds = [inv[i] for i in pred_idx]
    true  = [inv[i] for i in true_idx]
    print(f"\n{'='*60}\n  FINAL EVALUATION — best N-shot model\n{'='*60}")
    print(f"  Accuracy: {acc:.2f}%  ({sum(p==t for p,t in zip(preds,true))}/{len(true)})\n")
    print(classification_report(true, preds, target_names=classes, zero_division=0))
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(f"  classes order: {classes}")
    print(confusion_matrix(true, preds, labels=classes))
    print(f"{'='*60}")


# ------------------------------------------------- MAIN
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.getenv('DATA_PATH',
                   '/content/drive/MyDrive/colab dataset/ae_features_text_rpm.txt'),
                   help="dataset path; defaults to DATA_PATH from the .env file")
    p.add_argument('--mode',       default='attention', choices=['frozen', 'attention', 'full'])
    p.add_argument('--head',       default='cosine',    choices=['linear', 'cosine'])
    p.add_argument('--loss',       default='entropy',   choices=['ce', 'entropy'])
    p.add_argument('--entropy_beta', type=float, default=0.1)
    p.add_argument('--n_shot',     type=int,   default=5)
    p.add_argument('--split',      default='pooled', choices=['pooled', 'cross_speed', 'per_speed'])
    p.add_argument('--test_rpm',   type=int,   default=None, choices=[600, 800, 1000],
                   help="held-out speed for cross_speed; if omitted, leave-one-speed-out over all")
    p.add_argument('--stratify_rpm', action='store_true',
                   help="spread the N shots evenly across speeds (pooled/cross_speed)")
    p.add_argument('--sweep',      action='store_true', help="N=1..n_shot x {ce,entropy}")
    p.add_argument('--nshot_sweep', action='store_true',
                   help="train at several N (see --nshot_list); report recall + confusion")
    p.add_argument('--nshot_list', default='5,10,20,30',
                   help="comma-separated N values for --nshot_sweep")
    p.add_argument('--normalize_numbers', action='store_true')
    p.add_argument('--epochs',     type=int,   default=40)
    p.add_argument('--runs',       type=int,   default=10)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--max_length', type=int,   default=256)   # was 128 -> truncated features
    p.add_argument('--lr',         type=float, default=3e-5)   # backbone (GPT-2) LR
    p.add_argument('--head_lr',    type=float, default=1e-3)   # head LR (fc + classifier)
    p.add_argument('--model_path', default=os.getenv('MODEL_PATH', 'gpt2_nshot.pt'))
    args = p.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    samples, label_map, rpm_list = load_dataset(args.data, args.normalize_numbers)
    word_to_int = {v: k for k, v in RPM_WORD.items()}
    speeds = [word_to_int.get(r, r) for r in rpm_list]

    # ---- N-shot sweep (recall + confusion at each N)
    if args.nshot_sweep:
        if args.split == 'cross_speed' and args.test_rpm is None:
            for rpm in speeds:
                run_nshot_sweep(samples, label_map, args, tokenizer, test_rpm=rpm)
        else:
            run_nshot_sweep(samples, label_map, args, tokenizer, test_rpm=args.test_rpm)
        return

    # ---- sweep
    if args.sweep:
        if args.split == 'cross_speed' and args.test_rpm is None:
            for rpm in speeds:
                run_sweep(samples, label_map, args, tokenizer, test_rpm=rpm)
        else:
            run_sweep(samples, label_map, args, tokenizer, test_rpm=args.test_rpm)
        return

    # ---- per_speed: independent run per operating speed
    if args.split == 'per_speed':
        for rpm in speeds:
            sub = [s for s in samples if s[2] == RPM_WORD.get(rpm, rpm)]
            print(f"\n########## SPEED = {rpm} rpm  ({len(sub)} samples) ##########")
            args_pooled = argparse.Namespace(**{**vars(args), 'split': 'pooled'})
            r = run_nshot(sub, label_map, args_pooled, tokenizer, args.n_shot,
                          args.loss, keep_best_state=False, verbose=True)
            print(f"  [{rpm}rpm] {len(label_map)}-way {args.n_shot}-shot val: "
                  f"{r['val_mean']:.2f}% +/- {r['val_margin']:.2f}%")
        return

    # ---- cross_speed leave-one-speed-out (no single test_rpm given)
    if args.split == 'cross_speed' and args.test_rpm is None:
        outs = []
        for rpm in speeds:
            print(f"\n########## HELD-OUT SPEED = {rpm} rpm ##########")
            r = run_nshot(samples, label_map, args, tokenizer, args.n_shot,
                          args.loss, test_rpm=rpm, verbose=True)
            outs.append(r['val_mean'])
            print(f"  [held-out {rpm}rpm] val: {r['val_mean']:.2f}% +/- {r['val_margin']:.2f}%")
        print(f"\n  Leave-one-speed-out mean val: {np.mean(outs):.2f}%")
        return

    # ---- single config (pooled, or cross_speed with a fixed test_rpm) -> best model
    K = len(label_map)
    print(f"\n[RUN] {K}-way {args.n_shot}-shot | split={args.split} | loss={args.loss} | "
          f"beta={args.entropy_beta} | runs={args.runs}")
    r = run_nshot(samples, label_map, args, tokenizer, args.n_shot, args.loss,
                  test_rpm=args.test_rpm, keep_best_state=True, verbose=True)
    print(f"\n{'='*60}")
    print(f"  {K}-way {args.n_shot}-shot RESULT (split={args.split}, loss={args.loss})")
    print(f"  Val accuracy : {r['val_mean']:.2f}% +/- {r['val_margin']:.2f}%  "
          f"(best run {r['best_val']:.2f}%)")
    print(f"  Train acc.   : {r['tr_mean']:.2f}% +/- {r['tr_margin']:.2f}%")
    print(f"{'='*60}")

    torch.save({'state': r['best_state'], 'label_map': label_map,
                'mode': args.mode, 'head': args.head,
                'normalize_numbers': args.normalize_numbers}, args.model_path)
    print(f"[SAVED] best N-shot model -> {args.model_path}")

    model = GPT2Classifier(len(label_map), mode=args.mode, head=args.head).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in r['best_state'].items()})
    final_report(model, tokenizer, r['best_split'][0], r['best_split'][1], label_map, args)


if __name__ == '__main__':
    main()