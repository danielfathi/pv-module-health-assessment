# rf_baseline_ml.py
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# ---------- paths ----------
DATA_DIR = Path("paneldataset")
OUT_DIR  = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- params ----------
IMG_SIZE     = (256, 256)
LBP_P, LBP_R = 8, 1
GLCM_DIST    = [1, 2]
GLCM_ANGLES  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
IMG_EXT      = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')
RANDOM_STATE = 42

# ---------- feature extractors ----------
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
    g = (img_gray01 * 255).astype(np.uint8)  #  skimage
    lbp = local_binary_pattern(g, P=LBP_P, R=LBP_R, method='uniform')
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
    if img.ndim == 2: img = np.dstack([img, img, img])
    else:             img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5: img /= 255.0
    return img

# ---------- load dataset ----------
assert DATA_DIR.exists(), f"Dataset not found: {DATA_DIR.resolve()}"
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
X_paths, y = [], []
for yi, cls in enumerate(classes):
    for p in (DATA_DIR/cls).iterdir():
        if p.suffix.lower() in IMG_EXT:
            X_paths.append(p); y.append(yi)
X_paths, y = np.array(X_paths), np.array(y, dtype=np.int64)
print("Classes:", classes, "| total images:", len(X_paths))

# split ثابت
X_tr_p, X_te_p, y_tr, y_te = train_test_split(
    X_paths, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

def paths_to_feats(paths):
    feats = []
    for p in paths:
        img = read_to_rgb01(p)
        feats.append(extract_features_arr(img))
    return np.asarray(feats, dtype=np.float32)

X_tr = paths_to_feats(X_tr_p)
X_te = paths_to_feats(X_te_p)
print("Train/Test shapes:", X_tr.shape, X_te.shape)

#  sample_weight  (RF  sample_weight )
uniq = np.unique(y_tr)
cw = compute_class_weight(class_weight='balanced', classes=uniq, y=y_tr)
cw_dict = {int(k): float(v) for k, v in zip(uniq, cw)}
w_tr = np.array([cw_dict[int(lbl)] for lbl in y_tr], dtype=np.float32)
print("Class weights:", {classes[k]: v for k, v in cw_dict.items()})

# ----------  1: RF  ----------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, max_features='sqrt',
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_tr, y_tr, sample_weight=w_tr)
y_pred = rf.predict(X_te)
rep_default = classification_report(y_te, y_pred, target_names=classes, digits=3)
cm_default = confusion_matrix(y_te, y_pred, labels=uniq)
print("\n=== RandomForest (default) ===\n", rep_default)


plt.figure(figsize=(7,6))
plt.imshow(cm_default, interpolation='nearest')
plt.title("Confusion Matrix (RF default)")
plt.xticks(uniq, classes, rotation=45, ha='right'); plt.yticks(uniq, classes)
for i in range(cm_default.shape[0]):
    for j in range(cm_default.shape[1]):
        plt.text(j, i, cm_default[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_DIR / "cm_rf_default.png", dpi=200)
plt.close()

with open(OUT_DIR / "report_rf_default.txt", "w") as f:
    f.write(rep_default)

dump({"model": rf, "classes": classes}, OUT_DIR / "rf_default.joblib")


param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 4]
}
rf2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
gs = GridSearchCV(rf2, param_grid, scoring="f1_macro", cv=3, n_jobs=-1)
gs.fit(X_tr, y_tr, **{"sample_weight": w_tr})  
best = gs.best_estimator_
print("\nBest params:", gs.best_params_)

y_pred2 = best.predict(X_te)
rep_gs = classification_report(y_te, y_pred2, target_names=classes, digits=3)
cm_gs = confusion_matrix(y_te, y_pred2, labels=uniq)
print("\n=== RandomForest (grid) ===\n", rep_gs)

plt.figure(figsize=(7,6))
plt.imshow(cm_gs, interpolation='nearest')
plt.title("Confusion Matrix (RF grid)")
plt.xticks(uniq, classes, rotation=45, ha='right'); plt.yticks(uniq, classes)
for i in range(cm_gs.shape[0]):
    for j in range(cm_gs.shape[1]):
        plt.text(j, i, cm_gs[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_DIR / "cm_rf_grid.png", dpi=200); plt.close()

with open(OUT_DIR / "report_rf_grid.txt", "w") as f:
    f.write(rep_gs)

dump({"model": best, "classes": classes}, OUT_DIR / "rf_grid.joblib")

print("\nSaved:")
print(" - outputs/report_rf_default.txt, cm_rf_default.png, rf_default.joblib")
print(" - outputs/report_rf_grid.txt,    cm_rf_grid.png,    rf_grid.joblib")

