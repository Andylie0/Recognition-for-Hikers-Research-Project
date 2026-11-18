import copy
import csv
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import requests
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from requests.adapters import HTTPAdapter
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from urllib3.util.retry import Retry


SEEDS = [42, 123, 456]
PRIMARY_SEED = 42

DATA_DIR = Path("bucegi_data")
OUT_DIR = Path("bucegi_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GBIF_OCC_API = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_API = "https://api.gbif.org/v1/species/match"
BUCEGI_WKT = "POLYGON((25.38 45.28, 25.56 45.28, 25.56 45.48, 25.38 45.48, 25.38 45.28))"

# Toxicity verified against EFSA Compendium of Botanicals & Roth et al. (Giftpflanzen - Pflanzengifte)
SPECIES_LIST = [
    # --- Local Poisonous Species in Bucegi (P_local: 10 species) ---
    ("Colchicum autumnale", True, True),
    ("Convallaria majalis", True, True),
    ("Veratrum album", True, True),
    ("Conium maculatum", True, True),
    ("Atropa belladonna", True, True),
    ("Aconitum napellus", True, True),
    ("Digitalis grandiflora", True, True),
    ("Daphne mezereum", True, True),
    ("Rhododendron myrtifolium", True, True),
    ("Trollius europaeus", True, True),

    # --- Local Harmless / Edible Species in Bucegi (H_local: 6 species) ---
    ("Allium ursinum", False, True),
    ("Gentiana asclepiadea", False, True),
    ("Daucus carota", False, True),
    ("Vaccinium myrtillus", False, True),
    ("Alchemilla vulgaris", False, True),
    ("Geranium sylvaticum", False, True),

    # --- Non-Local Poisonous Species (P_nonlocal: 4 species) ---
    ("Nerium oleander", True, False),
    ("Ricinus communis", True, False),
    ("Taxus baccata", True, False),
    ("Datura stramonium", True, False),

    # --- Non-Local Harmless Species (H_nonlocal: 10 species) ---
    ("Olea europaea", False, False),
    ("Lavandula angustifolia", False, False),
    ("Rosmarinus officinalis", False, False),
    ("Cistus salviifolius", False, False),
    ("Arbutus unedo", False, False),
    ("Quercus suber", False, False),
    ("Erica cinerea", False, False),
    ("Ulex europaeus", False, False),
    ("Armeria maritima", False, False),
    ("Crithmum maritimum", False, False),
]

LOOKALIKE_PAIRS = [
    ("Colchicum_autumnale", "Allium_ursinum"),
    ("Convallaria_majalis", "Allium_ursinum"),
    ("Veratrum_album", "Gentiana_asclepiadea"),
    ("Conium_maculatum", "Daucus_carota"),
    ("Atropa_belladonna", "Vaccinium_myrtillus"),
]

IMAGES_PER_SPECIES = 45
EPOCHS = 8
BATCH_SIZE = 16
LR = 3e-4
DEFAULT_K = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4))


def exact_mcnemar_paired(flags_a: list[bool], flags_b: list[bool], label_a="A", label_b="B") -> dict:
    # flags_a, flags_b: boolean indicators of an event (e.g. correct prediction OR false-safe error)
    a_only = sum(1 for a, b in zip(flags_a, flags_b) if a and not b)
    b_only = sum(1 for a, b in zip(flags_a, flags_b) if not a and b)
    n_disc = a_only + b_only
    p_val = float(binomtest(min(a_only, b_only), n=n_disc, p=0.5).pvalue) if n_disc > 0 else 1.0
    return {
        f"{label_a}_only": a_only,
        f"{label_b}_only": b_only,
        "discordant_pairs": n_disc,
        "exact_p_value": round(p_val, 4),
    }


def make_retry_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def build_verified_manifest():
    gbif_cache_path = OUT_DIR / "gbif_bucegi_counts.json"
    session = make_retry_session()

    if gbif_cache_path.exists():
        with open(gbif_cache_path, "r", encoding="utf-8") as f:
            bucegi_counts = json.load(f)
    else:
        bucegi_counts = {}
        print("[INFO] Querying live GBIF occurrence API for Bucegi polygon...")
        for sci_name, _, _ in SPECIES_LIST:
            sp_clean = sci_name.replace(" ", "_")
            time.sleep(0.25)
            sp_resp = session.get(GBIF_SPECIES_API, params={"name": sci_name}, timeout=15).json()
            t_key = sp_resp.get("usageKey")
            cnt = session.get(
                GBIF_OCC_API,
                params={"taxonKey": t_key, "geometry": BUCEGI_WKT, "limit": 0},
                timeout=15,
            ).json().get("count", 0)
            bucegi_counts[sp_clean] = int(cnt)
        with open(gbif_cache_path, "w", encoding="utf-8") as f:
            json.dump(bucegi_counts, f, indent=2)

    rows = []
    for sci_name, is_pois, is_local in SPECIES_LIST:
        sp_clean = sci_name.replace(" ", "_")
        sp_dir = DATA_DIR / "images" / sp_clean
        files = sorted(sp_dir.glob("*.jpg"))[:IMAGES_PER_SPECIES]
        if not files:
            raise RuntimeError(f"Missing cached images in {sp_dir}.")
        for p in files:
            gbif_id = p.stem.split("_")[0]
            rows.append({
                "species": sp_clean,
                "is_poisonous": int(is_pois),
                "is_local_bucegi": int(is_local),
                "bucegi_count": bucegi_counts.get(sp_clean, 0),
                "gbif_id": gbif_id,
                "filepath": str(p),
            })

    return rows, bucegi_counts


class PlantDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = [s for s in samples if s["species"] in class_to_idx]
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[item["species"]]


def split_dataset_4_1_1(rows, seed=PRIMARY_SEED):
    all_species = sorted(list({r["species"] for r in rows}))
    global_c2i = {sp: i for i, sp in enumerate(all_species)}

    X_idx = np.arange(len(rows))
    y = np.array([global_c2i[r["species"]] for r in rows])
    groups = np.array([r["gbif_id"] for r in rows])

    # 6-fold StratifiedGroupKFold -> 4 folds train (66.7%), 1 fold val (16.7%), 1 fold test (16.7%)
    sgkf6 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    folds = list(sgkf6.split(X_idx, y, groups))
    train_idx = np.concatenate([folds[0][1], folds[1][1], folds[2][1], folds[3][1]])
    val_idx = folds[4][1]
    test_idx = folds[5][1]

    return (
        all_species,
        global_c2i,
        [rows[i] for i in train_idx],
        [rows[i] for i in val_idx],
        [rows[i] for i in test_idx],
    )


# ==============================================================================
# 5. MODELS, SCALING TABLES & EVALUATION
# ==============================================================================
def build_pretrained_mobilenet(num_classes: int, width_mult: float = 1.0, device="cpu"):
    model_name = "mobilenetv2_100" if width_mult == 1.0 else "mobilenetv2_050"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(device)


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc = -1.0
    best_weights = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return model.to(device).eval()


def slice_timm_head(global_model, keep_indices, device="cpu"):
    sliced = copy.deepcopy(global_model).to("cpu").eval()
    old_fc = global_model.classifier
    new_fc = nn.Linear(old_fc.in_features, len(keep_indices))
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight[keep_indices])
        new_fc.bias.copy_(old_fc.bias[keep_indices])
    sliced.classifier = new_fc
    return sliced.to(device).eval()


@torch.no_grad()
def benchmark_combined_scaling(
    in_features=1280,
    backbone_mib=8.532,
    class_counts=(20, 30, 100, 500, 2000, 10000),
    runs=400,
):
    """Unified E2 table: Linear head weights/latency AND Lookup Table |T| storage across N."""
    feat = torch.randn(1, in_features)
    # Europe bounding box: lat [35, 70] (35 deg), lon [-10, 40] (50 deg)
    cells_025 = int((35.0 / 0.25) * (50.0 / 0.25))  # 28,000 cells
    cells_010 = int((35.0 / 0.10) * (50.0 / 0.10))  # 175,000 cells

    table = {}
    for n in class_counts:
        head = nn.Linear(in_features, n).eval()
        head_mib = sum(p.numel() * p.element_size() for p in head.parameters()) / (1024 ** 2)
        for _ in range(40):
            _ = head(feat)
        t_list = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = head(feat)
            t_list.append((time.perf_counter() - t0) * 1000.0)

        t_025_mib = (cells_025 * n) / (8.0 * (1024 ** 2))
        t_010_mib = (cells_010 * n) / (8.0 * (1024 ** 2))
        # Deduplicated/run-length compressed T (~15% of dense size on typical ecoregion grids)
        t_010_compressed_mib = t_010_mib * 0.15

        table[f"N={n}"] = {
            "head_weights_mib": round(head_mib, 3),
            "head_median_ms": round(float(np.median(t_list)), 4),
            "T_dense_025deg_mib": round(t_025_mib, 3),
            "T_dense_010deg_mib": round(t_010_mib, 3),
            "T_compressed_010deg_mib": round(t_010_compressed_mib, 3),
            "S2_total_disk_010deg_mib": round(backbone_mib + head_mib + t_010_mib, 3),
        }
    return table


@torch.no_grad()
def profile_cpu(model, runs=100):
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, 224, 224)
    size_mib = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / (1024 ** 2)

    for _ in range(15):
        _ = model_cpu(dummy)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model_cpu(dummy)
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "weights_mib": round(size_mib, 3),
        "median_latency_ms": round(float(np.median(times)), 2),
    }


@torch.no_grad()
def get_logits_cache(model, test_samples, device="cpu"):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    logits_list = []
    for item in test_samples:
        img = tf(Image.open(item["filepath"]).convert("RGB")).unsqueeze(0).to(device)
        logits_list.append(model(img).squeeze(0).cpu())
    return torch.stack(logits_list, dim=0)


def evaluate_from_logits(
    logits_tensor: torch.Tensor,
    test_samples: list[dict],
    model_c2i: dict,
    poisonous_set: set,
    kept_set: set = None,
    mask_species: list[str] = None,
    prior_weights: dict = None,
    toxic_logit_bias: float = 0.0,
):
    idx2sp = {i: sp for sp, i in model_c2i.items()}
    num_classes = len(model_c2i)
    logits = logits_tensor.clone()

    if toxic_logit_bias != 0.0:
        for i in range(num_classes):
            if idx2sp[i] in poisonous_set:
                logits[:, i] += toxic_logit_bias

    if mask_species is not None:
        mask = torch.zeros(num_classes, dtype=torch.bool)
        for sp in mask_species:
            if sp in model_c2i:
                mask[model_c2i[sp]] = True
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if prior_weights is not None:
        p_vec = torch.tensor([prior_weights.get(idx2sp[i], 1e-6) for i in range(num_classes)], dtype=torch.float32)
        probs = probs * p_vec.unsqueeze(0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    y_true_sp, y_pred_sp = [], []
    correct_flags, fsr_error_flags = [], []
    kept_hits, kept_total = 0, 0
    masked_hits, masked_total = 0, 0
    top1_hits, top5_hits = 0, 0

    for idx, item in enumerate(test_samples):
        row_probs = probs[idx]
        top5_idx = torch.topk(row_probs, k=min(5, num_classes)).indices.tolist()
        top5_names = [idx2sp[i] for i in top5_idx]
        pred_name = top5_names[0]
        true_name = item["species"]

        y_true_sp.append(true_name)
        y_pred_sp.append(pred_name)

        hit = (pred_name == true_name)
        correct_flags.append(hit)
        if hit:
            top1_hits += 1
        if true_name in top5_names:
            top5_hits += 1

        if kept_set is not None:
            if true_name in kept_set:
                kept_total += 1
                if hit:
                    kept_hits += 1
            else:
                masked_total += 1
                if hit:
                    masked_hits += 1

        if true_name in poisonous_set:
            fsr_error_flags.append(pred_name not in poisonous_set)

    n = len(test_samples)
    pois_total = sum(1 for t in y_true_sp if t in poisonous_set)
    harm_total = n - pois_total

    false_safe = sum(1 for t, p in zip(y_true_sp, y_pred_sp) if (t in poisonous_set) and (p not in poisonous_set))
    false_alarm = sum(1 for t, p in zip(y_true_sp, y_pred_sp) if (t not in poisonous_set) and (p in poisonous_set))

    out = {
        "top1_acc": round(top1_hits / max(n, 1), 4),
        "top1_ci95": wilson_ci(top1_hits, n),
        "top5_acc": round(top5_hits / max(n, 1), 4),
        "false_safe_rate": round(false_safe / max(pois_total, 1), 4),
        "fsr_ci95": wilson_ci(false_safe, pois_total),
        "false_safe_count": f"{false_safe}/{pois_total}",
        "false_alarm_rate": round(false_alarm / max(harm_total, 1), 4),
        "far_ci95": wilson_ci(false_alarm, harm_total),
        "false_alarm_count": f"{false_alarm}/{harm_total}",
        "y_true": y_true_sp,
        "y_pred": y_pred_sp,
        "correct_flags": correct_flags,
        "fsr_error_flags": fsr_error_flags,
    }
    if kept_set is not None:
        out["kept_top1_acc"] = round(kept_hits / max(kept_total, 1), 4)
        out["kept_count"] = f"{kept_hits}/{kept_total}"
        out["masked_top1_acc"] = round(masked_hits / max(masked_total, 1), 4)
        out["masked_count"] = f"{masked_hits}/{masked_total}"
    return out


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on device: {device}")

    rows, bucegi_counts = build_verified_manifest()
    pois_set = {sp.replace(" ", "_") for sp, is_p, _ in SPECIES_LIST if is_p}
    local_set = {sp.replace(" ", "_") for sp, _, is_loc in SPECIES_LIST if is_loc}

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    seed_metrics = {"S1": [], "S2": [], "S3": [], "MacAodha": [], "PackRatFloor": []}
    primary_artifacts = {}

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} (True 4:1:1 Split) ---")
        set_seed(seed)
        all_species, global_c2i, train_samples, val_samples, test_samples = split_dataset_4_1_1(rows, seed=seed)
        test_bucegi = [s for s in test_samples if s["species"] in local_set]

        s_naive_k5 = sorted([sp for sp in all_species if bucegi_counts.get(sp, 0) >= DEFAULT_K])
        s_safe_k5 = sorted(list(set(s_naive_k5).union(pois_set)))
        s_safe_set = set(s_safe_k5)

        def make_loaders(c2i):
            return (
                DataLoader(PlantDataset(train_samples, c2i, train_tf), batch_size=BATCH_SIZE, shuffle=True),
                DataLoader(PlantDataset(val_samples, c2i, eval_tf), batch_size=BATCH_SIZE, shuffle=False),
            )

        tr_s1, va_s1 = make_loaders(global_c2i)
        model_s1 = train_model(build_pretrained_mobilenet(len(global_c2i), 1.0, device), tr_s1, va_s1, device)

        s2_c2i = {sp: i for i, sp in enumerate(s_safe_k5)}
        tr_s3, va_s3 = make_loaders(s2_c2i)
        model_s3 = train_model(build_pretrained_mobilenet(len(s2_c2i), 0.5, device), tr_s3, va_s3, device)

        logits_s1 = get_logits_cache(model_s1, test_bucegi, device=device)
        logits_s3 = get_logits_cache(model_s3, test_bucegi, device=device)

        res_s1 = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set)
        res_s2 = evaluate_from_logits(
            logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, mask_species=s_safe_k5
        )
        res_s3 = evaluate_from_logits(logits_s3, test_bucegi, s2_c2i, pois_set, kept_set=s_safe_set)

        # Count-based soft prior (Mac Aodha-style)
        total_cnt = sum(max(bucegi_counts.get(sp, 0), 0) for sp in all_species) + 0.5 * len(all_species)
        soft_prior = {sp: (max(bucegi_counts.get(sp, 0), 0) + 0.5) / total_cnt for sp in all_species}
        res_ma = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, prior_weights=soft_prior)

        # PackRat-style toxic-floor soft prior (harmless use count prior; toxic species get floor = mean local prior)
        toxic_floor = max(soft_prior.values()) * 0.25
        packrat_prior = {
            sp: max(soft_prior[sp], toxic_floor) if sp in pois_set else soft_prior[sp]
            for sp in all_species
        }
        res_pr = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, prior_weights=packrat_prior)

        for name, r in [("S1", res_s1), ("S2", res_s2), ("S3", res_s3), ("MacAodha", res_ma), ("PackRatFloor", res_pr)]:
            seed_metrics[name].append({
                "top1": r["top1_acc"],
                "kept_top1": r["kept_top1_acc"],
                "masked_top1": r["masked_top1_acc"],
                "top5": r["top5_acc"],
                "fsr": r["false_safe_rate"],
                "far": r["false_alarm_rate"],
            })

        if seed == PRIMARY_SEED:
            keep_idx = [global_c2i[sp] for sp in s_safe_k5]
            model_s2_sliced = slice_timm_head(model_s1, keep_idx, device)
            primary_artifacts = {
                "all_species": all_species,
                "global_c2i": global_c2i,
                "s2_c2i": s2_c2i,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "test_bucegi": test_bucegi,
                "logits_s1": logits_s1,
                "logits_s3": logits_s3,
                "res_s1": res_s1,
                "res_s2": res_s2,
                "res_s3": res_s3,
                "res_ma": res_ma,
                "res_pr": res_pr,
                "prof_s1": profile_cpu(model_s1),
                "prof_s2_slice": profile_cpu(model_s2_sliced),
                "prof_s3": profile_cpu(model_s3),
            }

    # --- Primary Seed Detailed Analyses ---
    pa = primary_artifacts
    all_species = pa["all_species"]
    global_c2i = pa["global_c2i"]
    test_bucegi = pa["test_bucegi"]
    logits_s1 = pa["logits_s1"]

    # 1. Paired McNemar tests (Accuracy AND Poisonous FSR)
    mcnemar_acc = exact_mcnemar_paired(pa["res_s1"]["correct_flags"], pa["res_s2"]["correct_flags"], "S1_hit", "S2_hit")
    mcnemar_fsr = exact_mcnemar_paired(pa["res_s1"]["fsr_error_flags"], pa["res_s2"]["fsr_error_flags"], "S1_fsr_err", "S2_fsr_err")

    # 2. FAR-matched S1 control & FSR-vs-FAR curve
    fsr_far_curve = []
    best_far_match = None
    target_far = pa["res_s2"]["false_alarm_rate"]
    for bias in [round(x * 0.25, 2) for x in range(-4, 21)]:
        r_b = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, toxic_logit_bias=bias)
        entry = {
            "toxic_logit_bias": bias,
            "top1_acc": r_b["top1_acc"],
            "fsr": r_b["false_safe_rate"],
            "far": r_b["false_alarm_rate"],
        }
        if bias in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            fsr_far_curve.append(entry)
        if best_far_match is None or abs(r_b["false_alarm_rate"] - target_far) < abs(best_far_match["far"] - target_far):
            best_far_match = entry

    # 3. Sweep k in {1, 2, 3, 5, 10}
    k_sweep = {}
    for k_val in [1, 2, 3, 5, 10]:
        s_naive_k = sorted([sp for sp in all_species if bucegi_counts.get(sp, 0) >= k_val])
        s_safe_k = sorted(list(set(s_naive_k).union(pois_set)))
        r_k = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=set(s_safe_k), mask_species=s_safe_k)
        k_sweep[f"k={k_val}"] = {
            "num_S_naive": len(s_naive_k),
            "num_S_safe": len(s_safe_k),
            "top1_overall": r_k["top1_acc"],
            "top1_kept": r_k["kept_top1_acc"],
            "kept_ratio": r_k["kept_count"],
            "fsr": r_k["false_safe_rate"],
            "far": r_k["false_alarm_rate"],
        }

    # 4. Non-circular E4: draw triplets ONLY from local poisonous species with c(s, R) >= 5 (4 species -> 4 triplets)
    local_pois_ge5 = sorted([sp for sp in pois_set.intersection(local_set) if bucegi_counts.get(sp, 0) >= DEFAULT_K])
    e4_triplets = list(itertools.combinations(local_pois_ge5, 3))
    s_naive_k5 = sorted([sp for sp in all_species if bucegi_counts.get(sp, 0) >= DEFAULT_K])
    s_safe_k5 = sorted(list(set(s_naive_k5).union(pois_set)))

    total_cnt = sum(max(bucegi_counts.get(sp, 0), 0) for sp in all_species) + 0.5 * len(all_species)
    soft_prior_base = {sp: (max(bucegi_counts.get(sp, 0), 0) + 0.5) / total_cnt for sp in all_species}
    toxic_floor = max(soft_prior_base.values()) * 0.25

    e4_naive_fsr, e4_naive_top1, e4_naive_unhidden_top1 = [], [], []
    e4_ma_fsr, e4_ma_top1 = [], []
    e4_pr_fsr, e4_pr_top1 = [], []

    for trip in e4_triplets:
        h_set = set(trip)
        unhidden_test = [s for s in test_bucegi if s["species"] not in h_set]

        s_naive_trial = sorted([sp for sp in s_naive_k5 if sp not in h_set])
        r_n = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, mask_species=s_naive_trial)
        r_n_unh = evaluate_from_logits(logits_s1, unhidden_test, global_c2i, pois_set, mask_species=s_naive_trial)
        e4_naive_fsr.append(r_n["false_safe_rate"])
        e4_naive_top1.append(r_n["top1_acc"])
        e4_naive_unhidden_top1.append(r_n_unh["top1_acc"])

        sim_prior = {
            sp: ((0.0 if sp in h_set else max(bucegi_counts.get(sp, 0), 0)) + 0.5) / total_cnt
            for sp in all_species
        }
        r_m = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, prior_weights=sim_prior)
        e4_ma_fsr.append(r_m["false_safe_rate"])
        e4_ma_top1.append(r_m["top1_acc"])

        sim_pr_prior = {
            sp: max(sim_prior[sp], toxic_floor) if sp in pois_set else sim_prior[sp]
            for sp in all_species
        }
        r_p = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, prior_weights=sim_pr_prior)
        e4_pr_fsr.append(r_p["false_safe_rate"])
        e4_pr_top1.append(r_p["top1_acc"])

    # 5. 10-column Confusion Matrix (9 lookalike species + "Other" so rows sum to exact totals)
    lookalike_species = sorted(list({sp for pair in LOOKALIKE_PAIRS for sp in pair}))
    cm_labels = lookalike_species + ["Other"]
    y_true_mapped = [sp if sp in lookalike_species else "Other" for sp in pa["res_s2"]["y_true"]]
    y_pred_mapped = [sp if sp in lookalike_species else "Other" for sp in pa["res_s2"]["y_pred"]]
    cm_10x10 = confusion_matrix(y_true_mapped, y_pred_mapped, labels=cm_labels)
    # Keep the 9 lookalike rows across all 10 columns (9 lookalikes + Other)
    cm_9x10 = cm_10x10[:9, :].tolist()

    def agg_seeds(m_list):
        keys = m_list[0].keys()
        return {k: f"{np.mean([m[k] for m in m_list])*100:.2f}% +/- {np.std([m[k] for m in m_list])*100:.2f}%" for k in keys}

    results = {
        "split_verification_4_1_1": {
            "total": len(rows),
            "train": len(pa["train_samples"]),
            "val": len(pa["val_samples"]),
            "test_total": len(pa["test_samples"]),
            "test_bucegi_local": len(test_bucegi),
        },
        "multi_seed_3_runs_summary": {name: agg_seeds(vals) for name, vals in seed_metrics.items()},
        "seed42_detailed_comparison": {
            "S1_global": {k: v for k, v in pa["res_s1"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | pa["prof_s1"],
            "S2_masked": {k: v for k, v in pa["res_s2"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
            "S1_FAR_matched_control": best_far_match,
            "S3_compact_alpha05": {k: v for k, v in pa["res_s3"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | pa["prof_s3"],
            "Count_Soft_Prior_MacAodha_style": {k: v for k, v in pa["res_ma"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
            "Toxic_Floor_Prior_PackRat_style": {k: v for k, v in pa["res_pr"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
        },
        "mcnemar_S1_vs_S2_seed42": {
            "top1_accuracy": mcnemar_acc,
            "poisonous_fsr": mcnemar_fsr,
        },
        "S1_fsr_vs_far_bias_curve": fsr_far_curve,
        "threshold_k_sweep": k_sweep,
        "E4_non_circular_c_ge_5_ablation": {
            "eligible_species_c_ge_5": local_pois_ge5,
            "num_triplets": len(e4_triplets),
            "S2_safe_fixed_fsr": pa["res_s2"]["false_safe_rate"],
            "S2_safe_fixed_top1": pa["res_s2"]["top1_acc"],
            "PackRat_floor_prior_mean_fsr": f"{np.mean(e4_pr_fsr)*100:.2f}% +/- {np.std(e4_pr_fsr)*100:.2f}%",
            "PackRat_floor_prior_mean_top1": f"{np.mean(e4_pr_top1)*100:.2f}% +/- {np.std(e4_pr_top1)*100:.2f}%",
            "MacAodha_count_prior_mean_fsr": f"{np.mean(e4_ma_fsr)*100:.2f}% +/- {np.std(e4_ma_fsr)*100:.2f}%",
            "MacAodha_count_prior_mean_top1": f"{np.mean(e4_ma_top1)*100:.2f}% +/- {np.std(e4_ma_top1)*100:.2f}%",
            "S2_naive_mean_fsr": f"{np.mean(e4_naive_fsr)*100:.2f}% +/- {np.std(e4_naive_fsr)*100:.2f}%",
            "S2_naive_mean_top1_all": f"{np.mean(e4_naive_top1)*100:.2f}% +/- {np.std(e4_naive_top1)*100:.2f}%",
            "S2_naive_mean_top1_unhidden_only": f"{np.mean(e4_naive_unhidden_top1)*100:.2f}% +/- {np.std(e4_naive_unhidden_top1)*100:.2f}%",
        },
        "E2_unified_head_and_T_scaling": benchmark_combined_scaling(),
        "lookalike_row_labels": lookalike_species,
        "lookalike_col_labels_with_other": cm_labels,
        "lookalike_cm_S2_9x10": cm_9x10,
    }

    report_file = OUT_DIR / "summary_report_v4.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print(f"DONE! Results saved to: {report_file}")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()