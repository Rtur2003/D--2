# ============================================================
# HEAD CT HEMORRHAGE — Google Colab Eğitim Scripti
# ============================================================
# Kullanım:
#   1. Google Drive'a yükle: head_ct/ ve data_v2/ klasörlerini
#   2. Bu dosyayı Colab'a yükle, hücreleri sırayla çalıştır
#   3. GPU: Runtime > Change runtime type > T4 GPU
# ============================================================

# %% [markdown]
# ## 1. Kurulum & Drive Bağlantısı

# %%
# !pip install timm -q

from google.colab import drive
drive.mount('/content/drive')

# Veri klasörlerinizin Drive'daki yolunu buraya yazın:
DRIVE_ROOT   = "/content/drive/MyDrive/Data"
HEAD_CT_DIR  = f"{DRIVE_ROOT}/head_ct"    # 200 görüntü  (normal/ hemorrhage/)
DATA_V2_DIR  = f"{DRIVE_ROOT}/data_v2"   # ~6800 görüntü (normal/ hemorrhage/)
SAVE_DIR     = f"{DRIVE_ROOT}/ct_models" # Modeller buraya kaydedilir

import os
os.makedirs(SAVE_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Import & Sabitler

# %%
import glob, random, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
SEED       = 42
CLASSES    = ["Normal", "Hemorrhage"]

print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# %% [markdown]
# ## 3. Dataset & Augmentation

# %%
def get_transforms(mean, std, is_train=False):
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if is_train:
        t += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.85,1.15)),
            transforms.RandomApply([transforms.GaussianBlur(3,(0.1,2.0))], p=0.3),
            transforms.RandomAutocontrast(p=0.2),
        ]
    t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if is_train:
        t.append(transforms.RandomErasing(p=0.3, scale=(0.02,0.25), value=0))
    return transforms.Compose(t)

class CTDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.tf = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]

def load_folder(folder):
    paths, labels = [], []
    for lbl, name in [(0,"normal"),(1,"hemorrhage")]:
        d = os.path.join(folder, name)
        if not os.path.isdir(d): continue
        for ext in ("*.jpg","*.jpeg","*.png"):
            for p in glob.glob(os.path.join(d, ext)):
                paths.append(p); labels.append(lbl)
    return paths, labels

def compute_stats(paths, n=500):
    sample = random.sample(paths, min(n, len(paths)))
    tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor()])
    tensors = [tf(Image.open(p).convert("RGB")) for p in sample]
    t = torch.stack(tensors)
    mean = t.mean([0,2,3]).tolist()
    std  = t.std([0,2,3]).tolist()
    return mean, std

def make_loaders(paths, labels, mean, std, batch=16, augment=True):
    tr_tf = get_transforms(mean, std, is_train=augment)
    va_tf = get_transforms(mean, std, is_train=False)

    n = len(paths)
    idx = list(range(n)); random.shuffle(idx)
    n_tr = int(n*0.80); n_va = int(n*0.10)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    tr_p = [paths[i] for i in tr_idx]; tr_l = [labels[i] for i in tr_idx]
    va_p = [paths[i] for i in va_idx]; va_l = [labels[i] for i in va_idx]
    te_p = [paths[i] for i in te_idx]; te_l = [labels[i] for i in te_idx]

    # Weighted sampler for class imbalance
    n0 = tr_l.count(0); n1 = tr_l.count(1)
    w  = [1/n0 if l==0 else 1/n1 for l in tr_l]
    sampler = WeightedRandomSampler(w, len(w))

    tr_dl = DataLoader(CTDataset(tr_p, tr_l, tr_tf), batch, sampler=sampler, num_workers=2)
    va_dl = DataLoader(CTDataset(va_p, va_l, va_tf), batch, shuffle=False, num_workers=2)
    te_dl = DataLoader(CTDataset(te_p, te_l, va_tf), batch, shuffle=False, num_workers=2)

    print(f"  Train: {len(tr_p)} | Val: {len(va_p)} | Test: {len(te_p)}")
    print(f"  Normal(train): {n0}, Hemorrhage(train): {n1}")
    return tr_dl, va_dl, te_dl

# %% [markdown]
# ## 4. Model Mimarisi — Custom CNN v2

# %%
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0 or not self.training: return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        return x / keep * (torch.rand(shape, device=x.device).floor_() + keep)

class ChannelAtt(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        m = max(c//r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(c,m,False), nn.ReLU(True), nn.Linear(m,c,False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b,c,_,_ = x.shape
        return x * self.sig(self.fc(self.avg(x).view(b,c)) + self.fc(self.max(x).view(b,c))).view(b,c,1,1)

class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x,1,True); mx,_ = torch.max(x,1,True)
        return x * self.sig(self.conv(torch.cat([avg,mx],1)))

class CBAM(nn.Module):
    def __init__(self, c): super().__init__(); self.ca=ChannelAtt(c); self.sa=SpatialAtt()
    def forward(self, x): return self.sa(self.ca(x))

class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, dp=0.0, drop_path=0.0):
        super().__init__()
        self.c1  = nn.Conv2d(ic,oc,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(oc)
        self.c2  = nn.Conv2d(oc,oc,3,1,1,bias=False);      self.b2=nn.BatchNorm2d(oc)
        self.cbam = CBAM(oc)
        self.drop2d = nn.Dropout2d(dp)
        self.dp  = DropPath(drop_path)
        self.sc  = nn.Identity() if stride==1 and ic==oc else nn.Sequential(
            nn.Conv2d(ic,oc,1,stride,bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        o = F.gelu(self.b1(self.c1(x)))
        o = self.drop2d(self.cbam(self.b2(self.c2(o))))
        return F.gelu(self.dp(o) + self.sc(x))

class DilatedDS(nn.Module):
    def __init__(self, c, dp=0.15):
        super().__init__()
        self.dw = nn.Conv2d(c,c,3,padding=2,dilation=2,groups=c,bias=False)
        self.pw = nn.Conv2d(c,c,1,bias=False); self.bn=nn.BatchNorm2d(c)
        self.cbam=CBAM(c); self.dp=DropPath(dp)
    def forward(self, x):
        return self.dp(self.cbam(F.gelu(self.bn(self.pw(self.dw(x)))))) + x

class MultiScale(nn.Module):
    def __init__(self, ic, oc):
        super().__init__(); b=oc//3
        self.b1=nn.Sequential(nn.Conv2d(ic,b,1,bias=False),nn.BatchNorm2d(b),nn.GELU())
        self.b3=nn.Sequential(nn.Conv2d(ic,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.GELU())
        self.b5=nn.Sequential(nn.Conv2d(ic,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.GELU(),
                               nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.GELU())
        r=oc-3*b
        self.fuse=nn.Sequential(nn.Conv2d(3*b,oc,1,bias=False),nn.BatchNorm2d(oc),nn.GELU()) if r>0 else nn.Identity()
    def forward(self, x):
        o=torch.cat([self.b1(x),self.b3(x),self.b5(x)],1)
        return o if isinstance(self.fuse,nn.Identity) else self.fuse(o)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,32,7,2,3,bias=False),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(3,2,1))
        self.ms   = MultiScale(32,48)
        self.b1   = ResBlock(48,64,2,0.05,0.00)
        self.b2   = ResBlock(64,128,2,0.10,0.05)
        self.b3   = ResBlock(128,256,2,0.15,0.10)
        self.b4   = DilatedDS(256,0.15)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls  = nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode="fan_in",nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward(self, x):
        x=self.pool(self.b4(self.b3(self.b2(self.b1(self.ms(self.stem(x))))))).view(x.size(0),-1)
        return self.cls(x)

# %% [markdown]
# ## 5. Augmentation Teknikleri (Mixup + CutMix)

# %%
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1.0
    idx = torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0),device=x.device)
    ya, yb = y, y[idx]
    W,H = x.size(3),x.size(2)
    r = np.sqrt(1-lam); cw,ch = int(W*r),int(H*r)
    cx,cy = np.random.randint(W),np.random.randint(H)
    x1,x2 = max(0,cx-cw//2),min(W,cx+cw//2)
    y1,y2 = max(0,cy-ch//2),min(H,cy+ch//2)
    xn = x.clone(); xn[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    lam = 1-(x2-x1)*(y2-y1)/(W*H)
    return xn, ya, yb, lam

def mix_loss(crit, pred, ya, yb, lam):
    return lam*crit(pred,ya)+(1-lam)*crit(pred,yb)

# %% [markdown]
# ## 6. Eğitim Fonksiyonu

# %%
def train_model(model, tr_dl, va_dl, name, epochs=50, lr=1e-4, wd=1e-4,
                use_mix=True, alpha=0.2, label_smooth=0.1, patience=10):
    model = model.to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    best_val, best_state, wait = 0.0, None, 0
    history = {"tr_loss":[],"tr_acc":[],"va_loss":[],"va_acc":[]}

    for ep in range(1, epochs+1):
        # ── Train ──
        model.train(); tl=tc=tt=0
        for xb,yb in tr_dl:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            if use_mix:
                fn = mixup if random.random()<0.5 else cutmix
                xb,ya,yb2,lam = fn(xb,yb,alpha)
                out = model(xb); loss = mix_loss(crit,out,ya,yb2,lam)
                tc += out.argmax(1).eq(ya).sum().item()
            else:
                out = model(xb); loss = crit(out,yb)
                tc += out.argmax(1).eq(yb).sum().item()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tl += loss.item()*xb.size(0); tt += xb.size(0)
        sched.step(ep)

        # ── Val ──
        model.eval(); vl=vc=vt=0
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb = xb.to(DEVICE),yb.to(DEVICE)
                out = model(xb); vl+=crit(out,yb).item()*xb.size(0)
                vc+=out.argmax(1).eq(yb).sum().item(); vt+=xb.size(0)
        va_acc = vc/vt

        history["tr_loss"].append(tl/tt); history["tr_acc"].append(tc/tt)
        history["va_loss"].append(vl/vt); history["va_acc"].append(va_acc)

        print(f"[{name}] Ep {ep:3d}/{epochs} | tr_loss={tl/tt:.4f} tr_acc={tc/tt:.4f} | val_acc={va_acc:.4f} | lr={opt.param_groups[0]['lr']:.2e}")

        if va_acc > best_val:
            best_val = va_acc; best_state = copy.deepcopy(model.state_dict()); wait=0
            torch.save({"model_state_dict":best_state,"epoch":ep,"best_val":best_val},
                       os.path.join(SAVE_DIR, f"{name}_best.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"  [Early Stop] {ep} epoch"); break

    print(f"\n  Best val_acc: {best_val:.4f}")
    model.load_state_dict(best_state)
    return model, history

# %% [markdown]
# ## 7. Test & Görselleştirme

# %%
def evaluate(model, te_dl, name):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb,yb in te_dl:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(1).cpu().tolist())
            trues.extend(yb.tolist())
    acc = sum(p==t for p,t in zip(preds,trues))/len(trues)
    print(f"\n=== {name} Test Accuracy: {acc*100:.2f}% ===")
    print(classification_report(trues, preds, target_names=CLASSES))

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{name} Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_cm.png"), dpi=120)
    plt.show()
    return acc

def plot_history(history, name):
    fig, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
    a1.plot(history["tr_loss"], label="Train"); a1.plot(history["va_loss"], label="Val")
    a1.set_title(f"{name} Loss"); a1.legend(); a1.grid(True,alpha=0.3)
    a2.plot(history["tr_acc"], label="Train"); a2.plot(history["va_acc"], label="Val")
    a2.set_title(f"{name} Accuracy"); a2.legend(); a2.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_curves.png"), dpi=120)
    plt.show()

# %% [markdown]
# ## 8. Veri Yükleme

# %%
# Tüm görüntüleri birleştir (head_ct + data_v2)
print("=== Veri Yükleniyor ===")
paths, labels = [], []

if os.path.isdir(HEAD_CT_DIR):
    p, l = load_folder(HEAD_CT_DIR)
    paths+=p; labels+=l
    print(f"head_ct: {len(p)} görüntü")

if os.path.isdir(DATA_V2_DIR):
    p, l = load_folder(DATA_V2_DIR)
    paths+=p; labels+=l
    print(f"data_v2: {len(p)} görüntü")

print(f"\nToplam: {len(paths)} | Normal: {labels.count(0)} | Hemorrhage: {labels.count(1)}")

# Normalizasyon istatistikleri
print("\nNormalizasyon hesaplanıyor...")
mean, std = compute_stats(paths, n=1000)
print(f"Mean: {[round(m,4) for m in mean]}")
print(f"Std:  {[round(s,4) for s in std]}")

# DataLoader'lar
print("\nDataLoader'lar oluşturuluyor...")
tr_dl, va_dl, te_dl = make_loaders(paths, labels, mean, std, batch=32)

# %% [markdown]
# ## 9. Custom CNN v2 Eğitimi

# %%
print("\n" + "="*60)
print("Custom CNN v2 — CBAM + DropPath + DilatedDS")
print("="*60)

cnn = CustomCNN(num_classes=2)
total = sum(p.numel() for p in cnn.parameters())
print(f"Parametre: {total:,}")

cnn, cnn_hist = train_model(
    cnn, tr_dl, va_dl,
    name="custom_cnn",
    epochs=50,
    lr=1e-4,
    wd=1e-4,
    use_mix=True,
    alpha=0.2,
    label_smooth=0.1,
    patience=10,
)

plot_history(cnn_hist, "Custom CNN v2")
evaluate(cnn, te_dl, "Custom CNN v2")

# %% [markdown]
# ## 10. ConvNeXt-Tiny Eğitimi

# %%
print("\n" + "="*60)
print("ConvNeXt-Tiny — Transfer Learning")
print("="*60)

convnext = timm.create_model("convnext_tiny", pretrained=True, num_classes=2)
# Faz 1: Sadece head (5 epoch)
for p in convnext.parameters(): p.requires_grad = False
for p in convnext.head.parameters(): p.requires_grad = True

convnext, _ = train_model(
    convnext, tr_dl, va_dl,
    name="convnext_phase1",
    epochs=5, lr=1e-3, wd=1e-4,
    use_mix=False, label_smooth=0.1, patience=5,
)

# Faz 2: Tüm ağ (fine-tune)
for p in convnext.parameters(): p.requires_grad = True
convnext, cnxt_hist = train_model(
    convnext, tr_dl, va_dl,
    name="convnext_tiny",
    epochs=40, lr=5e-5, wd=1e-4,
    use_mix=True, alpha=0.2,
    label_smooth=0.1, patience=10,
)

plot_history(cnxt_hist, "ConvNeXt-Tiny")
evaluate(convnext, te_dl, "ConvNeXt-Tiny")

# %% [markdown]
# ## 11. Modelleri İndir

# %%
from google.colab import files

for fname in ["custom_cnn_best.pth", "convnext_tiny_best.pth"]:
    path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(path):
        files.download(path)
        print(f"İndiriliyor: {fname}")

print("\nTamamlandi! Modeller hem Drive'a kaydedildi hem indirildi.")
