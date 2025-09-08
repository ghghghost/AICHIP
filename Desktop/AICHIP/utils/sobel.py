import argparse
from glob import glob
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


# ---------------------------
# Model: Sobel as Conv2d
# ---------------------------
class Sobel(nn.Module):
    def __init__(self, constant: float):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False
        )
        Gx = torch.tensor([[constant, 0.0, -constant],
                           [2*constant, 0.0, -2*constant],
                           [constant, 0.0, -constant]], dtype=torch.float32)
        Gy = torch.tensor([[constant, 2*constant,  constant],
                           [0.0, 0.0, 0.0],
                           [-constant, -2*constant, -constant]], dtype=torch.float32)
        G = torch.stack([Gx, Gy], dim=0).unsqueeze(1)  # (2,1,3,3)
        with torch.no_grad():
            self.filter.weight.copy_(G)
        self.filter.requires_grad_(False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B,1,H,W), float32 in [0,255]
        x = self.filter(img)            # (B,2,H,W) -> [Gx, Gy]
        x = x * x
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + 1e-12)       # numerical stability
        return x


# ---------------------------
# Dataset
# ---------------------------
class GrayImageDataset(Dataset):
    def __init__(self, paths, resize: int):
        """
        paths: list of file paths
        resize: output size (square). Must be >0 for batching to have equal shapes.
        """
        self.paths = paths
        self.resize = resize
        if not self.paths:
            raise ValueError("입력 경로에 해당하는 이미지가 없습니다.")

        if self.resize is None or self.resize <= 0:
            raise ValueError("배치 처리를 위해 --resize 가 양수여야 합니다.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # 빈 더미를 주는 대신 에러를 던져 DataLoader가 잡게 하자
            raise FileNotFoundError(f"[SKIP] 이미지를 읽을 수 없습니다: {p}")
        gray = cv2.resize(gray, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
        # torch tensor [1,H,W], keep scale [0,255]
        t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0)
        return t, p  # for debug


# ---------------------------
# Utility
# ---------------------------
def safe_norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    return (x / m) if m > 0 else x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/workspace/val2017/*",
                   help="입력 이미지 글롭 패턴 (예: /workspace/val2017/*.jpg)")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda"], help="연산 장치 선택")
    p.add_argument("--constant", type=float, default=2.0,
                   help="Sobel 커널 스케일 계수")
    p.add_argument("--resize", type=int, default=224,
                   help="배치 처리를 위한 정사각 리사이즈 크기(필수, >0)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="배치 크기 (GPU 권장: 64~256)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--pin_memory", action="store_true",
                   help="DataLoader pin_memory 활성화 (GPU에서 권장)")
    p.add_argument("--debug", action="store_true",
                   help="중간 시각화")
    p.add_argument("--debug_limit", type=int, default=3,
                   help="디버그 시 최대 몇 배치/이미지까지 표시할지")
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Device pick
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 사용 불가.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Collect files
    paths = sorted(glob(args.root))
    if not paths:
        print("[ERROR] 잘못된 경로 또는 이미지 없음.")
        return

    # Dataset / DataLoader
    try:
        ds = GrayImageDataset(paths, resize=args.resize)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size if device.type == "cuda" else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and (device.type == "cuda"),
        drop_last=False,
    )

    sobel = Sobel(args.constant).to(device)
    total_torch = 0.0
    total_cv2 = 0.0
    shown = 0

    # Batch loop
    for img_batch_cpu, path_batch in loader:
        # img_batch_cpu: (B,1,H,W) float32 on CPU
        # ---- Torch (GPU/CPU) ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.inference_mode():
            img_batch = img_batch_cpu.to(device, non_blocking=True)
            out_batch = sobel(img_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_torch += (time.time() - t0)

        # ---- OpenCV baseline (CPU) in-batch ----
        t0 = time.time()
        imgs_np = img_batch_cpu.squeeze(1).numpy().astype(np.uint8)  # (B,H,W)
        for i in range(imgs_np.shape[0]):
            sobel_x = cv2.Sobel(imgs_np[i], cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imgs_np[i], cv2.CV_32F, 0, 1, ksize=3)
            _ = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        total_cv2 += (time.time() - t0)

        # ---- Debug visualize ----
        if args.debug and shown < args.debug_limit:
            gray_vis = imgs_np[0].astype(np.float32) / 255.0
            out0 = out_batch[0:1]  # (1,1,H,W)
            out0_np = out0.detach().cpu().squeeze().numpy()
            sx = cv2.Sobel(imgs_np[0], cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(imgs_np[0], cv2.CV_32F, 0, 1, ksize=3)
            cv2_vis = np.sqrt(np.square(sx) + np.square(sy))

            concat = np.concatenate(
                [gray_vis, safe_norm(out0_np), safe_norm(cv2_vis)], axis=1
            )
            plt.figure(figsize=(10, 4))
            plt.imshow(concat, cmap="gray")
            plt.title(f"Input | Torch Sobel | OpenCV Sobel\n{path_batch[0]}")
            plt.axis("off")
            plt.show()
            shown += 1

    print(f"[DONE] images: {len(paths)} | total torch: {total_torch:.6f}s | total cv2: {total_cv2:.6f}s")
    if len(paths) > 0:
        print(f"[MEAN] per-image torch: {total_torch/len(paths):.6f}s | per-image cv2: {total_cv2/len(paths):.6f}s")


if __name__ == "__main__":
    main()
