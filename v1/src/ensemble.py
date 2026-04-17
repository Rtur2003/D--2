# =========================================================================
# ENSEMBLE MODEL - İki Modelin Birleştirilmesi
# =========================================================================
# Neden Ensemble?
# -> Tek model hata yapabilir, iki model birbirini tamamlar
# -> ConvNeXt (derin, genel özellikler) + Custom CNN (basit, yerel özellikler)
# -> Soft voting: Her iki modelin olasılık tahminlerinin ortalaması
#
# Medikal AI'da Ensemble neden kritik?
# -> Yanlış negatif (kaçırılan kanama) hayat tehlikesi
# -> Ensemble ile false negative oranı düşer
# -> Farklı mimari = farklı hata paterni = birbirini düzeltme
# =========================================================================

import json
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE, MODELS_DIR, CLASS_NAMES


class EnsembleModel:
    """
    Soft Voting Ensemble: İki modelin olasılık tahminlerinin ağırlıklı ortalaması.

    Formül:
        P_ensemble(sınıf) = w1 * P_model1(sınıf) + w2 * P_model2(sınıf)

    Varsayılan: Eşit ağırlık (w1 = w2 = 0.5)
    Alternatif: Validation performansına göre ağırlık
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        weight1: float = 0.5,
        weight2: float = 0.5
    ):
        self.model1 = model1.to(DEVICE).eval()
        self.model2 = model2.to(DEVICE).eval()
        self.weight1 = weight1
        self.weight2 = weight2

        print(f"[ENSEMBLE] Model 1 ağırlığı: {weight1:.2f}")
        print(f"[ENSEMBLE] Model 2 ağırlığı: {weight2:.2f}")

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble tahmin.

        Returns:
            predictions: Tahmin edilen sınıflar
            probabilities: Ensemble olasılıkları
        """
        images = images.to(DEVICE)

        # Her iki modelden olasılıklar
        logits1 = self.model1(images)
        logits2 = self.model2(images)

        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        # Ağırlıklı ortalama
        ensemble_probs = self.weight1 * probs1 + self.weight2 * probs2

        predictions = ensemble_probs.argmax(dim=1).cpu().numpy()
        probabilities = ensemble_probs.cpu().numpy()

        return predictions, probabilities

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tüm DataLoader üzerinde ensemble tahmin."""
        all_preds = []
        all_probs = []
        all_labels = []

        for images, labels in loader:
            preds, probs = self.predict(images)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    @torch.no_grad()
    def predict_single(self, input_tensor: torch.Tensor) -> Dict:
        """
        Tek görüntü için detaylı ensemble tahmin.
        Arayüz için kullanılır.
        """
        input_tensor = input_tensor.to(DEVICE)

        logits1 = self.model1(input_tensor)
        logits2 = self.model2(input_tensor)

        probs1 = F.softmax(logits1, dim=1)[0].cpu().numpy()
        probs2 = F.softmax(logits2, dim=1)[0].cpu().numpy()

        ensemble_probs = self.weight1 * probs1 + self.weight2 * probs2
        pred_class = ensemble_probs.argmax()

        return {
            "prediction": CLASS_NAMES[pred_class],
            "confidence": float(ensemble_probs[pred_class]),
            "ensemble_scores": {
                CLASS_NAMES[i]: float(ensemble_probs[i]) for i in range(len(CLASS_NAMES))
            },
            "model1_scores": {
                CLASS_NAMES[i]: float(probs1[i]) for i in range(len(CLASS_NAMES))
            },
            "model2_scores": {
                CLASS_NAMES[i]: float(probs2[i]) for i in range(len(CLASS_NAMES))
            }
        }


def compute_optimal_weights(
    model1: nn.Module,
    model2: nn.Module,
    val_loader: DataLoader
) -> Tuple[float, float]:
    """
    Validation setinden optimal ensemble ağırlıklarını hesapla.

    Strateji: 0.0 ile 1.0 arasında farklı ağırlıkları dene,
    en yüksek accuracy veren ağırlığı seç.
    """
    print("[ENSEMBLE] Optimal ağırlıklar hesaplanıyor...")

    best_acc = 0.0
    best_w1 = 0.5

    model1.eval().to(DEVICE)
    model2.eval().to(DEVICE)

    for w1 in np.arange(0.0, 1.05, 0.1):
        w2 = 1.0 - w1
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                p1 = F.softmax(model1(images), dim=1)
                p2 = F.softmax(model2(images), dim=1)
                ensemble = w1 * p1 + w2 * p2
                preds = ensemble.argmax(dim=1)
                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"  w1={w1:.1f}, w2={w2:.1f} -> Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_w1 = w1

    best_w2 = 1.0 - best_w1
    print(f"\n[ENSEMBLE] Optimal: w1={best_w1:.1f}, w2={best_w2:.1f} "
          f"(Val Acc: {best_acc:.4f})")

    return best_w1, best_w2
