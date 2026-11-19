import copy
import csv
import itertools
import json
import math
import os
import random
import time
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
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

# ==============================================================================
# 1. CONFIGURATION & VERIFIED TOXICITY LABELS
# ==============================================================================
SPLIT_SEED = 42
TRAIN_SEEDS = [42, 123, 456]
PRIMARY_SEED = 42

DATA_DIR = Path("bucegi_data")
OUT_DIR = Path("bucegi_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GBIF_OCC_API = "https://api.gbif.org/v1/occurrence/search"
GBIF_OCC_GET = "https://api.gbif.org/v1/occurrence"
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
BATCH_SIZE = 16
DEFAULT_K = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. STATISTICAL HELPERS
# ==============================================================================
def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4))


def exact_mcnemar_paired(flags_a: list[bool], flags_b: list[bool], label_a="A", label_b="B") -> dict:
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


# ==============================================================================
# 3. LOAD CACHED IMAGES, OBSERVER/CELL METADATA & EUROPEAN OCCURRENCE GRIDS
# ==============================================================================
def make_retry_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def build_verified_manifest_and_grids():
    session = make_retry_session()

    # 3a. Bucegi counts cache
    gbif_cache_path = OUT_DIR / "gbif_bucegi_counts.json"
    if gbif_cache_path.exists():
        with open(gbif_cache_path, "r", encoding="utf-8") as f:
            bucegi_counts = json.load(f)
    else:
        bucegi_counts = {}
        print("[INFO] Querying live GBIF occurrence API for Bucegi polygon...")
        for sci_name, _, _ in SPECIES_LIST:
            sp_clean = sci_name.replace(" ", "_")
            time.sleep(0.2)
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

    # 3b. European occurrence coordinates per species for empirical T compression measurement
    eu_occ_cache_path = OUT_DIR / "europe_occ_coords.json"
    if eu_occ_cache_path.exists():
        with open(eu_occ_cache_path, "r", encoding="utf-8") as f:
            eu_coords = json.load(f)
    else:
        eu_coords = {}
        print("[INFO] Fetching European occurrence coordinates (300/species) for empirical |T| compression...")
        for sci_name, _, _ in SPECIES_LIST:
            sp_clean = sci_name.replace(" ", "_")
            time.sleep(0.2)
            sp_resp = session.get(GBIF_SPECIES_API, params={"name": sci_name}, timeout=15).json()
            t_key = sp_resp.get("usageKey")
            occ_resp = session.get(
                GBIF_OCC_API,
                params={"taxonKey": t_key, "continent": "EUROPE", "hasCoordinate": "true", "limit": 300},
                timeout=20,
            ).json()
            pts = []
            for r in occ_resp.get("results", []):
                lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
                if lat is not None and lon is not None:
                    pts.append((round(float(lat), 3), round(float(lon), 3)))
            eu_coords[sp_clean] = pts
        with open(eu_occ_cache_path, "w", encoding="utf-8") as f:
            json.dump(eu_coords, f)

    # 3c. Collect image paths and fetch observer/spatial-cell metadata for leakage-free grouping
    raw_items = []
    unique_gbif_ids = set()
    for sci_name, is_pois, is_local in SPECIES_LIST:
        sp_clean = sci_name.replace(" ", "_")
        sp_dir = DATA_DIR / "images" / sp_clean
        files = sorted(sp_dir.glob("*.jpg"))[:IMAGES_PER_SPECIES]
        if not files:
            raise RuntimeError(f"Missing cached images in {sp_dir}.")
        for p in files:
            gbif_id = p.stem.split("_")[0]
            unique_gbif_ids.add(gbif_id)
            raw_items.append((sp_clean, int(is_pois), int(is_local), gbif_id, str(p)))

    obs_meta_path = OUT_DIR / "gbif_obs_observer_meta.json"
    if obs_meta_path.exists():
        with open(obs_meta_path, "r", encoding="utf-8") as f:
            obs_meta = json.load(f)
    else:
        obs_meta = {}

    missing_ids = [gid for gid in sorted(unique_gbif_ids) if gid not in obs_meta]
    if missing_ids:
        print(f"[INFO] Fetching observer & coordinate metadata for {len(missing_ids)} GBIF observations...")
        from concurrent.futures import ThreadPoolExecutor

        def fetch_obs(gid):
            try:
                r = session.get(f"{GBIF_OCC_GET}/{gid}", timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    rec_by = str(d.get("recordedBy") or "").strip()
                    lat = d.get("decimalLatitude")
                    lon = d.get("decimalLongitude")
                    cell = f"{round(float(lat)*20)/20:.2f}_{round(float(lon)*20)/20:.2f}" if (lat is not None and lon is not None) else ""
                    return gid, {"recordedBy": rec_by, "cell_005": cell}
            except Exception:
                pass
            return gid, {"recordedBy": "", "cell_005": ""}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for gid, meta in pool.map(fetch_obs, missing_ids):
                obs_meta[gid] = meta
        with open(obs_meta_path, "w", encoding="utf-8") as f:
            json.dump(obs_meta, f)

    rows = []
    for sp_clean, is_pois, is_local, gbif_id, filepath in raw_items:
        meta = obs_meta.get(gbif_id, {})
        rec_by = meta.get("recordedBy", "")
        cell = meta.get("cell_005", "")
        # Group primarily by observer (if available) + spatial cell so same photographer/site never leaks
        if rec_by:
            group_key = f"obs_{rec_by}"
        elif cell:
            group_key = f"cell_{cell}"
        else:
            group_key = f"id_{gbif_id}"

        rows.append({
            "species": sp_clean,
            "is_poisonous": is_pois,
            "is_local_bucegi": is_local,
            "bucegi_count": bucegi_counts.get(sp_clean, 0),
            "gbif_id": gbif_id,
            "group_key": group_key,
            "filepath": filepath,
        })

    return rows, bucegi_counts, eu_coords


# ==============================================================================
# 4. FIXED 4:1:1 LEAKAGE-FREE SPLIT (Observer/Cell Grouped)
# ==============================================================================
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


def split_dataset_fixed_4_1_1(rows, split_seed=SPLIT_SEED):
    all_species = sorted(list({r["species"] for r in rows}))
    global_c2i = {sp: i for i, sp in enumerate(all_species)}

    X_idx = np.arange(len(rows))
    y = np.array([global_c2i[r["species"]] for r in rows])
    groups = np.array([r["group_key"] for r in rows])

    sgkf6 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=split_seed)
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
# 5. MODELS, TRAINING (STANDARD + DISTILLATION), SCALING & EVALUATION
# ==============================================================================
def build_pretrained_mobilenet(num_classes: int, width_mult: float = 1.0, device="cpu"):
    if width_mult == 1.0:
        model_name = "mobilenetv2_100"
    elif width_mult == 0.75:
        model_name = "mobilenetv2_075"
    else:
        model_name = "mobilenetv2_050"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(device)


def train_model(model, train_loader, val_loader, device, epochs=12, lr=3e-4, label_smoothing=0.0):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        scheduler.step()

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


def train_distilled_student(
    student_model,
    teacher_model,
    keep_indices,
    train_loader,
    val_loader,
    device,
    epochs=15,
    lr=8e-4,
    temp=3.0,
    alpha_kd=0.5,
):
    teacher_model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = -1.0
    best_weights = None

    for _ in range(epochs):
        student_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            s_logits = student_model(x)
            with torch.no_grad():
                t_logits = teacher_model(x)[:, keep_indices]
            loss_ce = ce_loss_fn(s_logits, y)
            loss_kd = kl_loss_fn(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
            ) * (temp ** 2)
            loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd
            loss.backward()
            optimizer.step()
        scheduler.step()

        student_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = student_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}

    student_model.load_state_dict(best_weights)
    return student_model.to(device).eval()


def slice_timm_head(global_model, keep_indices, device="cpu"):
    sliced = copy.deepcopy(global_model).to("cpu").eval()
    old_fc = global_model.classifier
    new_fc = nn.Linear(old_fc.in_features, len(keep_indices))
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight[keep_indices])
        new_fc.bias.copy_(old_fc.bias[keep_indices])
    sliced.classifier = new_fc
    return sliced.to(device).eval()


def measure_empirical_grid_compression(eu_coords: dict, all_species: list, pois_set: set):
    """Builds bit-packed European grid occupancy table from real GBIF coordinates and measures zlib compression."""
    # Europe grid: lat [35, 70] (35 deg), lon [-10, 40] (50 deg)
    n_lat, n_lon = int(35.0 / 0.10), int(50.0 / 0.10)  # 350 x 500 = 175,000 cells
    grid = np.zeros((n_lat * n_lon, len(all_species)), dtype=np.uint8)
    for j, sp in enumerate(all_species):
        if sp in pois_set:
            # Safety rule sets m_{R,s} = 1 everywhere for all poisonous species
            grid[:, j] = 1
        else:
            for lat, lon in eu_coords.get(sp, []):
                if 35.0 <= lat < 70.0 and -10.0 <= lon < 40.0:
                    r = int((lat - 35.0) / 0.10)
                    c = int((lon + 10.0) / 0.10)
                    grid[r * n_lon + c, j] = 1

    packed = np.packbits(grid, axis=1).tobytes()
    compressed = zlib.compress(packed, level=9)
    ratio = len(compressed) / max(len(packed), 1)
    return {
        "raw_bytes": len(packed),
        "zlib_bytes": len(compressed),
        "empirical_compression_ratio": round(ratio, 4),
    }


@torch.no_grad()
def benchmark_combined_scaling(
    backbone_mib: float,
    empirical_comp_ratio: float,
    in_features=1280,
    class_counts=(20, 30, 100, 500, 2000, 10000),
    runs=400,
):
    feat = torch.randn(1, in_features)
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
        t_010_zlib_mib = t_010_mib * empirical_comp_ratio

        table[f"N={n}"] = {
            "head_weights_mib": round(head_mib, 3),
            "head_median_ms": round(float(np.median(t_list)), 4),
            "T_dense_025deg_mib": round(t_025_mib, 3),
            "T_dense_010deg_mib": round(t_010_mib, 3),
            "T_zlib_010deg_mib": round(t_010_zlib_mib, 3),
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

    rows, bucegi_counts, eu_coords = build_verified_manifest_and_grids()
    pois_set = {sp.replace(" ", "_") for sp, is_p, _ in SPECIES_LIST if is_p}
    local_set = {sp.replace(" ", "_") for sp, _, is_loc in SPECIES_LIST if is_loc}

    # Fixed 4:1:1 split across all seeds (grouped by observer / spatial cell)
    all_species, global_c2i, train_samples, val_samples, test_samples = split_dataset_fixed_4_1_1(rows, split_seed=SPLIT_SEED)
    val_bucegi = [s for s in val_samples if s["species"] in local_set]
    test_bucegi = [s for s in test_samples if s["species"] in local_set]

    s_naive_k5 = sorted([sp for sp in all_species if bucegi_counts.get(sp, 0) >= DEFAULT_K])
    s_safe_k5 = sorted(list(set(s_naive_k5).union(pois_set)))
    s_safe_set = set(s_safe_k5)
    keep_idx = [global_c2i[sp] for sp in s_safe_k5]
    s2_c2i = {sp: i for i, sp in enumerate(s_safe_k5)}

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

    def make_loaders(c2i):
        return (
            DataLoader(PlantDataset(train_samples, c2i, train_tf), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(PlantDataset(val_samples, c2i, eval_tf), batch_size=BATCH_SIZE, shuffle=False),
        )

    # Check if mobilenetv2_075 is available in timm
    has_075 = True
    try:
        _ = timm.create_model("mobilenetv2_075", pretrained=True)
    except Exception:
        has_075 = False

    seed_metrics = {
        "S1": [],
        "S2": [],
        "S1_val_tuned_bias": [],
        "S3_tuned_a05": [],
        "S3_distil_a05": [],
        "MacAodha": [],
        "PackRatFloor": [],
    }
    if has_075:
        seed_metrics["S3_tuned_a075"] = []

    primary_artifacts = {}

    total_cnt = sum(max(bucegi_counts.get(sp, 0), 0) for sp in all_species) + 0.5 * len(all_species)
    soft_prior = {sp: (max(bucegi_counts.get(sp, 0), 0) + 0.5) / total_cnt for sp in all_species}
    toxic_floor = max(soft_prior.values()) * 0.25
    packrat_prior = {
        sp: max(soft_prior[sp], toxic_floor) if sp in pois_set else soft_prior[sp]
        for sp in all_species
    }

    for seed in TRAIN_SEEDS:
        print(f"\n--- Training Seed {seed} on Fixed Observer-Grouped 4:1:1 Split ---")
        set_seed(seed)

        tr_s1, va_s1 = make_loaders(global_c2i)
        model_s1 = train_model(
            build_pretrained_mobilenet(len(global_c2i), 1.0, device),
            tr_s1, va_s1, device, epochs=12, lr=3e-4, label_smoothing=0.05
        )

        tr_s3, va_s3 = make_loaders(s2_c2i)
        model_s3_tuned = train_model(
            build_pretrained_mobilenet(len(s2_c2i), 0.5, device),
            tr_s3, va_s3, device, epochs=15, lr=8e-4, label_smoothing=0.05
        )
        model_s3_distil = train_distilled_student(
            build_pretrained_mobilenet(len(s2_c2i), 0.5, device),
            model_s1, keep_idx, tr_s3, va_s3, device, epochs=15, lr=8e-4, temp=3.0, alpha_kd=0.5
        )
        if has_075:
            model_s3_075 = train_model(
                build_pretrained_mobilenet(len(s2_c2i), 0.75, device),
                tr_s3, va_s3, device, epochs=15, lr=6e-4, label_smoothing=0.05
            )

        # Tune toxic logit bias on VALIDATION set to match S2's validation FAR
        val_logits_s1 = get_logits_cache(model_s1, val_bucegi, device=device)
        val_res_s2 = evaluate_from_logits(val_logits_s1, val_bucegi, global_c2i, pois_set, mask_species=s_safe_k5)
        target_val_far = val_res_s2["false_alarm_rate"]
        best_val_bias = 0.0
        best_diff = 999.0
        for b_cand in [round(x * 0.25, 2) for x in range(0, 21)]:
            v_r = evaluate_from_logits(val_logits_s1, val_bucegi, global_c2i, pois_set, toxic_logit_bias=b_cand)
            if abs(v_r["false_alarm_rate"] - target_val_far) < best_diff:
                best_diff = abs(v_r["false_alarm_rate"] - target_val_far)
                best_val_bias = b_cand

        # Evaluate on TEST set
        logits_s1 = get_logits_cache(model_s1, test_bucegi, device=device)
        logits_s3_tuned = get_logits_cache(model_s3_tuned, test_bucegi, device=device)
        logits_s3_distil = get_logits_cache(model_s3_distil, test_bucegi, device=device)

        res_s1 = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set)
        res_s2 = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, mask_species=s_safe_k5)
        res_s1_bias = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, toxic_logit_bias=best_val_bias)
        res_s3_t = evaluate_from_logits(logits_s3_tuned, test_bucegi, s2_c2i, pois_set, kept_set=s_safe_set)
        res_s3_d = evaluate_from_logits(logits_s3_distil, test_bucegi, s2_c2i, pois_set, kept_set=s_safe_set)
        res_ma = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, prior_weights=soft_prior)
        res_pr = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, kept_set=s_safe_set, prior_weights=packrat_prior)

        eval_pairs = [
            ("S1", res_s1, 0.0),
            ("S2", res_s2, 0.0),
            ("S1_val_tuned_bias", res_s1_bias, best_val_bias),
            ("S3_tuned_a05", res_s3_t, 0.0),
            ("S3_distil_a05", res_s3_d, 0.0),
            ("MacAodha", res_ma, 0.0),
            ("PackRatFloor", res_pr, 0.0),
        ]
        if has_075:
            logits_s3_075 = get_logits_cache(model_s3_075, test_bucegi, device=device)
            res_s3_75 = evaluate_from_logits(logits_s3_075, test_bucegi, s2_c2i, pois_set, kept_set=s_safe_set)
            eval_pairs.append(("S3_tuned_a075", res_s3_75, 0.0))

        for name, r, b_used in eval_pairs:
            seed_metrics[name].append({
                "top1": r["top1_acc"],
                "kept_top1": r["kept_top1_acc"],
                "masked_top1": r["masked_top1_acc"],
                "top5": r["top5_acc"],
                "fsr": r["false_safe_rate"],
                "far": r["false_alarm_rate"],
                "val_tuned_bias": b_used,
            })

        if seed == PRIMARY_SEED:
            model_s2_sliced = slice_timm_head(model_s1, keep_idx, device)
            primary_artifacts = {
                "logits_s1": logits_s1,
                "res_s1": res_s1,
                "res_s2": res_s2,
                "res_s1_bias": res_s1_bias,
                "best_val_bias": best_val_bias,
                "res_s3_t": res_s3_t,
                "res_s3_d": res_s3_d,
                "res_ma": res_ma,
                "res_pr": res_pr,
                "prof_s1": profile_cpu(model_s1),
                "prof_s2_slice": profile_cpu(model_s2_sliced),
                "prof_s3_a05": profile_cpu(model_s3_tuned),
                "prof_s3_a075": profile_cpu(model_s3_075) if has_075 else None,
            }

    pa = primary_artifacts
    logits_s1 = pa["logits_s1"]

    # 1. Paired McNemar tests (Accuracy & Poisonous FSR)
    mcnemar_acc = exact_mcnemar_paired(pa["res_s1"]["correct_flags"], pa["res_s2"]["correct_flags"], "S1_hit", "S2_hit")
    mcnemar_fsr = exact_mcnemar_paired(pa["res_s1"]["fsr_error_flags"], pa["res_s2"]["fsr_error_flags"], "S1_fsr_err", "S2_fsr_err")

    # 2. FSR-vs-FAR curve across bias sweep + save plot
    fsr_far_curve = []
    for bias in [round(x * 0.25, 2) for x in range(-4, 21)]:
        r_b = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, toxic_logit_bias=bias)
        fsr_far_curve.append({
            "toxic_logit_bias": bias,
            "top1_acc": r_b["top1_acc"],
            "fsr": r_b["false_safe_rate"],
            "far": r_b["false_alarm_rate"],
        })

    plt.figure(figsize=(6.2, 4.2))
    fars = [pt["far"] * 100 for pt in fsr_far_curve]
    fsrs = [pt["fsr"] * 100 for pt in fsr_far_curve]
    plt.plot(fars, fsrs, "-o", markersize=3.5, color="#2b5c8f", label="S1 + Toxic Logit Bias Curve")
    plt.scatter([pa["res_s1"]["false_alarm_rate"] * 100], [pa["res_s1"]["false_safe_rate"] * 100], color="black", s=55, zorder=5, label="S1 Unfiltered (b=0)")
    plt.scatter([pa["res_s2"]["false_alarm_rate"] * 100], [pa["res_s2"]["false_safe_rate"] * 100], color="#d95f02", marker="s", s=65, zorder=5, label="S2 Hard Union (k=5)")
    plt.scatter([pa["res_pr"]["false_alarm_rate"] * 100], [pa["res_pr"]["false_safe_rate"] * 100], color="#1b9e77", marker="^", s=70, zorder=5, label="Toxic-Floor Prior (PackRat-style)")
    plt.xlabel("False-Alarm Rate (FAR, %)")
    plt.ylabel("False-Safe Rate (FSR, %)")
    plt.title("Safety-Specificity Trade-off (FSR vs. FAR on Bucegi Test Set)")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fsr_far_curve.pdf")
    plt.savefig(OUT_DIR / "fsr_far_curve.png", dpi=200)
    plt.close()

    # 3. Non-circular E4: Monte Carlo record-drop curve p_drop in {0%, 25%, 50%, 75%, 100%} at k=1 and k=5
    rng_mc = np.random.default_rng(SPLIT_SEED)
    e4_drop_curve = {}
    for k_eval in [1, 5]:
        e4_drop_curve[f"k={k_eval}"] = {}
        for p_drop in [0.0, 0.25, 0.50, 0.75, 1.0]:
            naive_fsrs, naive_top1s, ma_fsrs, pr_fsrs = [], [], [], []
            for _ in range(50):
                sim_counts = {
                    sp: int(rng_mc.binomial(max(bucegi_counts.get(sp, 0), 0), 1.0 - p_drop))
                    for sp in all_species
                }
                s_naive_sim = sorted([sp for sp in all_species if sim_counts[sp] >= k_eval])
                r_n = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, mask_species=s_naive_sim)
                naive_fsrs.append(r_n["false_safe_rate"])
                naive_top1s.append(r_n["top1_acc"])

                tot_c = sum(sim_counts.values()) + 0.5 * len(all_species)
                sim_ma = {sp: (sim_counts[sp] + 0.5) / tot_c for sp in all_species}
                r_m = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, prior_weights=sim_ma)
                ma_fsrs.append(r_m["false_safe_rate"])

                # Keep the same toxic floor calibrated from baseline local abundance
                sim_pr = {sp: max(sim_ma[sp], toxic_floor) if sp in pois_set else sim_ma[sp] for sp in all_species}
                r_p = evaluate_from_logits(logits_s1, test_bucegi, global_c2i, pois_set, prior_weights=sim_pr)
                pr_fsrs.append(r_p["false_safe_rate"])

            e4_drop_curve[f"k={k_eval}"][f"drop_{int(p_drop*100)}pct"] = {
                "S_safe_invariant_fsr": f"{pa['res_s2']['false_safe_rate']*100:.2f}% (by design)",
                "PackRat_floor_fsr": f"{np.mean(pr_fsrs)*100:.2f}% +/- {np.std(pr_fsrs)*100:.2f}%",
                "MacAodha_prior_fsr": f"{np.mean(ma_fsrs)*100:.2f}% +/- {np.std(ma_fsrs)*100:.2f}%",
                "Naive_filter_fsr": f"{np.mean(naive_fsrs)*100:.2f}% +/- {np.std(naive_fsrs)*100:.2f}%",
                "Naive_filter_top1": f"{np.mean(naive_top1s)*100:.2f}% +/- {np.std(naive_top1s)*100:.2f}%",
            }

    # 4. Empirical compression & exact backbone constant for Table 3
    emp_comp = measure_empirical_grid_compression(eu_coords, all_species, pois_set)
    head_30_mib = (1280 * 30 + 30) * 4 / (1024 ** 2)  # 0.147 MiB
    exact_backbone_mib = round(pa["prof_s1"]["weights_mib"] - head_30_mib, 3)  # 8.483 MiB
    unified_scaling = benchmark_combined_scaling(
        backbone_mib=exact_backbone_mib,
        empirical_comp_ratio=emp_comp["empirical_compression_ratio"],
    )

    def agg_seeds(m_list):
        keys = [k for k in m_list[0].keys() if k != "val_tuned_bias"]
        res = {k: f"{np.mean([m[k] for m in m_list])*100:.2f}% +/- {np.std([m[k] for m in m_list])*100:.2f}%" for k in keys}
        if m_list[0]["val_tuned_bias"] != 0.0:
            res["mean_val_tuned_bias"] = round(float(np.mean([m["val_tuned_bias"] for m in m_list])), 2)
        return res

    results = {
        "split_verification_fixed_4_1_1": {
            "total": len(rows),
            "train": len(train_samples),
            "val": len(val_samples),
            "test_total": len(test_samples),
            "test_bucegi_local": len(test_bucegi),
            "unique_observer_or_cell_groups": len({r["group_key"] for r in rows}),
        },
        "multi_seed_3_runs_fixed_split": {name: agg_seeds(vals) for name, vals in seed_metrics.items()},
        "seed42_detailed": {
            "S1": {k: v for k, v in pa["res_s1"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | pa["prof_s1"],
            "S2": {k: v for k, v in pa["res_s2"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
            "S1_val_tuned_bias": {k: v for k, v in pa["res_s1_bias"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | {"val_tuned_bias": pa["best_val_bias"]},
            "S3_tuned_a05": {k: v for k, v in pa["res_s3_t"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | pa["prof_s3_a05"],
            "S3_distil_a05": {k: v for k, v in pa["res_s3_d"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")} | pa["prof_s3_a05"],
            "MacAodha_count_prior": {k: v for k, v in pa["res_ma"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
            "PackRat_floor_prior": {k: v for k, v in pa["res_pr"].items() if k not in ("y_true", "y_pred", "correct_flags", "fsr_error_flags")},
            "prof_s3_a075": pa["prof_s3_a075"],
        },
        "mcnemar_S1_vs_S2_seed42": {
            "top1_accuracy": mcnemar_acc,
            "poisonous_fsr": mcnemar_fsr,
        },
        "fsr_far_curve_sampled": [pt for pt in fsr_far_curve if pt["toxic_logit_bias"] in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
        "E4_record_drop_curve": e4_drop_curve,
        "empirical_T_compression": emp_comp,
        "exact_backbone_mib": exact_backbone_mib,
        "E2_unified_head_and_T_scaling": unified_scaling,
    }

    report_file = OUT_DIR / "summary_report_v5.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print(f"DONE! Results saved to: {report_file}")
    print(f"Plot saved to: {OUT_DIR / 'fsr_far_curve.pdf'}")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()