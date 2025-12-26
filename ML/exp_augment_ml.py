# exp_augment_ml.py
import os, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from skimage import io, color, util, exposure
from skimage.transform import resize, rotate
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# --------- paths ----------
DATA_DIR = Path("paneldataset")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------- params ----------
IMG_SIZE     = (256, 256)
LBP_P, LBP_R = 8, 1
GLCM_DIST    = [1, 2]
GLCM_ANGLES  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
IMG_EXT      = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

RANDOM_STATE = 42
AUG_PER_IMG  = 2  

# --------- features ----------
def color_hist_feats(img_rgb, bins=8):
    h = []
    for c in range(3):
        hist, _ = np.histogram(img_rgb[..., c], bins=bins, range=(0,1), density=True)
        h.append(hist.astype(np.float32))
    return np.concatenate(h)

def glcm_feats(img_gray01):
    g = (img_gray01 * 255).astype(np.uint8)
    glcm = graycomatrix(g, distances=GLCM_DIST, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)
    props = ['contrast','homogeneity','ASM','correlation']
    feats = [graycoprops(glcm, p).ravel().astype(np.float32) for p in props]
    return np.concatenate(feats)

def lbp_feats(img_gray01):
    lbp = local_binary_pattern(img_gray01, P=LBP_P, R=LBP_R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_P+2, range=(0, LBP_P+2), density=True)
    return hist.astype(np.float32)

def extract_features_arr(img_rgb01):
    img_rgb01 = resize(img_rgb01, IMG_SIZE, anti_aliasing=True)
    img_gray = color.rgb2gray(img_rgb01)
    return np.concatenate([color_hist_feats(img_rgb01),
                           glcm_feats(img_gray),
                           lbp_feats(img_gray)]).astype(np.float32)

def read_to_rgb01(path: Path):
    img = io.imread(str(path))
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    else:
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img

# --------- augmentations ----------
rng = np.random.default_rng(RANDOM_STATE)

def aug_rotate_flip(img):
    ang = rng.uniform(-15, 15)
    out = rotate(img, ang, mode='edge', preserve_range=True)
    if rng.random() < 0.5: out = np.fliplr(out)
    if rng.random() < 0.5: out = np.flipud(out)
    return out

def aug_brightness(img):
    # gamma jitter (0.8..1.2) + slight intensity rescale
    g = rng.uniform(0.8, 1.2)
    out = exposure.adjust_gamma(img, g)
    out = exposure.rescale_intensity(out, in_range=(0,1), out_range=(0,1))
    return out

def aug_noise(img):
    sigma = rng.uniform(0.0, 0.02)
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img + n, 0, 1)
    return out

def aug_cutout(img, max_frac=0.25):
    h, w, _ = img.shape
    fh = int(h * rng.uniform(0.1, max_frac))
    fw = int(w * rng.uniform(0.1, max_frac))
    y = rng.integers(0, h - fh + 1)
    x = rng.integers(0, w - fw + 1)
    out = img.copy()
    out[y:y+fh, x:x+fw, :] = out.mean()  # fill with mean
    return out


SCENARIOS = {
    "baseline_none": [],
    "rot_flip": [aug_rotate_flip],
    "rot_flip_brightness": [aug_rotate_flip, aug_brightness],
    "rot_flip_brightness_noise_cutout": [aug_rotate_flip, aug_brightness, aug_noise, aug_cutout],
}

# --------- load dataset ----------
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
X_paths, y_labels = [], []
for yi, cls in enumerate(classes):
    for p in (DATA_DIR/cls).iterdir():
        if p.suffix.lower() in IMG_EXT:
            X_paths.append(p)
            y_labels.append(yi)
X_paths = np.array(X_paths)
y_labels = np.array(y_labels)
print("Classes:", classes, "| total images:", len(X_paths))


X_tr_p, X_te_p, y_tr, y_te = train_test_split(
    X_paths, y_labels, test_size=0.2, random_state=RANDOM_STATE, stratify=y_labels
)
print("Train/Test:", len(X_tr_p), len(X_te_p))

X_te = []
for p in X_te_p:
    img = read_to_rgb01(p)
    X_te.append(extract_features_arr(img))
X_te = np.asarray(X_te, dtype=np.float32)

# --------- run scenarios ----------
rows = []
for name, aug_list in SCENARIOS.items():
    print(f"\n=== Scenario: {name} ===")
  
    feats, labs = [], []
    for p, yi in zip(X_tr_p, y_tr):
        base = read_to_rgb01(p)
     
        feats.append(extract_features_arr(base))
        labs.append(yi)
     
        for _ in range(AUG_PER_IMG):
            img_aug = base.copy()
            for aug in aug_list:
                img_aug = aug(img_aug)
            feats.append(extract_features_arr(img_aug))
            labs.append(yi)

    X_tr = np.asarray(feats, dtype=np.float32)
    y_tr_aug = np.asarray(labs, dtype=np.int64)
    print("Train feats:", X_tr.shape, "| Test feats:", X_te.shape)

    # Scale + class weights
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)
    X_te_std = scaler.transform(X_te)

    uniq = np.unique(y_tr_aug)
    cw = compute_class_weight(class_weight='balanced', classes=uniq, y=y_tr_aug)
    cw_dict = {int(k): float(v) for k, v in zip(uniq, cw)}

    clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight=cw_dict, random_state=RANDOM_STATE)
    clf.fit(X_tr_std, y_tr_aug)

    y_pred = clf.predict(X_te_std)
    report = classification_report(y_te, y_pred, target_names=classes, digits=3, output_dict=True)
    cm = confusion_matrix(y_te, y_pred, labels=uniq)


    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({name})")
    plt.xticks(uniq, classes, rotation=45, ha='right'); plt.yticks(uniq, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    cm_path = OUT_DIR / f"cm_{name}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()


    dump({"scaler": scaler, "model": clf, "classes": classes}, OUT_DIR / f"svm_{name}.joblib")


    rows.append({
        "scenario": name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "train_samples": int(X_tr.shape[0]),
        "aug_per_img": AUG_PER_IMG,
    })


    with open(OUT_DIR / f"report_{name}.txt", "w") as f:
        from sklearn.metrics import classification_report as cr_plain
        f.write(cr_plain(y_te, y_pred, target_names=classes, digits=3))


df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
csv_path = OUT_DIR / "aug_results.csv"
df.to_csv(csv_path, index=False)
print("\nSaved:")
print(" -", csv_path)
for r in rows:
    print(f" - cm_{r['scenario']}.png, svm_{r['scenario']}.joblib, report_{r['scenario']}.txt")
