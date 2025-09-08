from matplotlib import pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torch
from glob import glob
import time
import cv2
import numpy as np
from utils.sobel import GrayImageDataset, Sobel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="val2017/*",
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
