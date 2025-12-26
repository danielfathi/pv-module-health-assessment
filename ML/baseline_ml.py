
"""
Baseline ML for PV panel defect classification
- Features: Color histogram + GLCM (Haralick) + LBP
- Model: SVM (RBF) with class weights
- Outputs: confusion matrix (PNG), classification report (TXT), saved model (joblib)
Folder layout expected:
proje/
  ├─ paneldataset/            # one subfolder per class
  ├─ baseline_ml.py           # this file
  └─ outputs/                 # auto-created
"""

import os, glob, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.transform import resize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# -------------------- paths --------------------
DATA_DIR = Path("paneldataset")      
OUT_DIR  = Path("outputs")            
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- params -------------------
IMG_SIZE     = (256, 256)            
LBP_P, LBP_R = 8, 1                  
GLCM_DIST    = [1, 2]                
GLCM_ANGLES  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
IMG_EXT      = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

# ---------------- feature extractors -----------
def color_hist_feats(img_rgb, bins=8):
    """Normalized per-channel histograms (range 0..1) => 3*bins features."""
    h = []
    for c in range(3):
        hist, _ = np.histogram(img_rgb[..., c], bins=bins, range=(0,1), density=True)
        h.append(hist.astype(np.float32))
    return np.concatenate(h)

def glcm_feats(img_gray01):
    """Haralick-style props from graycomatrix (uint8) over multiple angles/distances."""
    g = (img_gray01 * 255).astype(np.uint8)
    glcm = graycomatrix(g, distances=GLCM_DIST, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'ASM', 'correlation']
    feats = [graycoprops(glcm, p).ravel().astype(np.float32) for p in props]
    return np.concatenate(feats)

def lbp_feats(img_gray01):
    """Uniform LBP histogram (P+2 bins)."""
    lbp = local_binary_pattern(img_gray01, P=LBP_P, R=LBP_R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_P+2, range=(0, LBP_P+2), density=True)
    return hist.astype(np.float32)

def extract_features(image_path: Path):
    """Read → resize → RGB/gray → concat features."""
    img = io.imread(str(image_path))
    if img.ndim == 2:                 # grayscale → RGB
        img_rgb = np.dstack([img, img, img])
    else:
        img_rgb = img[..., :3]
    img_rgb = img_rgb.astype(np.float32)
    if img_rgb.max() > 1.5:           # 0..255 → 0..1
        img_rgb /= 255.0
    img_rgb = resize(img_rgb, IMG_SIZE, anti_aliasing=True)
    img_gray = color.rgb2gray(img_rgb)  # float in 0..1

    f_color = color_hist_feats(img_rgb)
    f_glcm  = glcm_feats(img_gray)
    f_lbp   = lbp_feats(img_gray)
    return np.concatenate([f_color, f_glcm, f_lbp]).astype(np.float32)

# -------------------- load dataset -------------
assert DATA_DIR.exists(), f"Dataset folder not found: {DATA_DIR.resolve()}"
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print("Classes:", classes)
assert len(classes) > 1, "Need at least 2 classes under paneldataset/"

X, y = [], []
skipped = 0
for yi, cls in enumerate(classes):
    cls_dir = DATA_DIR / cls
    files = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    if len(files) == 0:
        print(f"[warn] no images in {cls_dir}")
    for fp in files:
        try:
            feats = extract_features(fp)
            if not np.isfinite(feats).all():
                raise ValueError("NaN/Inf in features")
            X.append(feats)
            y.append(yi)
        except Exception as e:
            skipped += 1
            print(f"[skip] {fp.name}: {e}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

print("Data shape:", X.shape, "| labels:", len(y), "| skipped files:", skipped)
if X.shape[0] == 0:
    raise SystemExit("No samples loaded. Check images/paths.")

# توزیع برچسب‌ها
uniq, counts = np.unique(y, return_counts=True)
print("Label counts:", {classes[i]: int(c) for i, c in zip(uniq, counts)})

# -------------------- split & scale ------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train/Test sizes:", X_tr.shape[0], X_te.shape[0])

scaler = StandardScaler()
X_tr_std = scaler.fit_transform(X_tr)
X_te_std = scaler.transform(X_te)

# -------------------- class weights ------------
class_w = compute_class_weight(class_weight='balanced', classes=uniq, y=y_tr)
class_w_dict = {int(k): float(v) for k, v in zip(uniq, class_w)}
print("Class weights:", {classes[k]: v for k, v in class_w_dict.items()})

# -------------------- train SVM ----------------
clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight=class_w_dict)
clf.fit(X_tr_std, y_tr)

# -------------------- evaluate -----------------
y_pred = clf.predict(X_te_std)
report = classification_report(y_te, y_pred, target_names=classes, digits=3)
print(report)

cm = confusion_matrix(y_te, y_pred, labels=uniq)
plt.figure(figsize=(7,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (SVM baseline)")
plt.xticks(uniq, classes, rotation=45, ha='right')
plt.yticks(uniq, classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
cm_path = OUT_DIR / "cm_baseline_svm.png"
plt.savefig(cm_path, dpi=200)
plt.show()

# -------------------- save artifacts -----------
model_path = OUT_DIR / "baseline_svm.joblib"
dump({"scaler": scaler, "model": clf, "classes": classes}, model_path)

with open(OUT_DIR / "baseline_report.txt", "w") as f:
    f.write(report)

print(f"Saved:\n - {model_path}\n - {cm_path}\n - {OUT_DIR/'baseline_report.txt'}")
