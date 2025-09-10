from matplotlib import pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torch
from glob import glob
from pathlib import Path
import os
import time
import cv2
import numpy as np
from utils.sobel import GrayImageDataset, Sobel
import rebel


def safe_norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    return (x / m) if m > 0 else x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, default="train2017/*",
                   help="입력(디렉터리 또는 글롭 패턴). 예: /data/val2017/ 또는 /data/val2017/*.jpg")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda"], help="연산 장치 선택(기존 CPU/GPU)")
    p.add_argument("--constant", type=float, default=2.0,
                   help="Sobel 커널 스케일 계수")
    p.add_argument("--resize", type=int, default=224,
                   help="배치 처리를 위한 정사각 리사이즈 크기(필수, >0)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="배치 크기 (GPU 권장: 64~256)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers, For benchmark use 0")
    p.add_argument("--pin_memory", action="store_true",
                   help="DataLoader pin_memory 활성화 (GPU에서 권장)")
    p.add_argument("--debug", action="store_true",
                   help="중간 시각화")
    p.add_argument("--debug_limit", type=int, default=3,
                   help="디버그 시 최대 몇 배치/이미지까지 표시할지")
    p.add_argument("--accel", type=str, required=True, default='cpu',
                   choices=["cpu", "cuda", "rbln"],
                   help="실행 백엔드 선택: cpu/cuda/rbln(ATOM NPU)")
    return p.parse_args()


def collect_image_paths(root_arg: str):
    """디렉터리 또는 글롭 패턴을 받아 이미지 경로 리스트 반환."""
    root_arg = os.path.expanduser(root_arg)
    p = Path(root_arg)
    paths = []
    if p.is_dir():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for ext in exts:
            paths.extend(sorted(str(x) for x in p.rglob(ext)))
    else:
        # 글롭 패턴 (재귀 패턴 **/*.jpg 포함)
        paths = sorted(glob(root_arg, recursive=True))
    return paths


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # 가속기 분기: RBLN은 CPU 텐서를 그대로 넘기므로 CUDA 필요 없음
    if args.accel == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA 사용 불가. CPU로 대체합니다.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 파일 수집
    paths = collect_image_paths(args.root)
    if not paths:
        print(f"[ERROR] 이미지가 없습니다. --root에 디렉터리 또는 패턴을 주세요. "
              f"예) /data/val2017/*.jpg 혹은 /data/val2017/**/*.jpg")
        return

    # Dataset / DataLoader
    try:
        ds = GrayImageDataset(paths, resize=args.resize)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    if args.accel == "cuda":
        torch.backends.cudnn.benchmark = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and args.accel == "cuda"),
        drop_last=True,  # 정적 형태 유지: 마지막 작은 배치 제거
    )

    base = Sobel(args.constant).eval()

    if args.accel == "rbln":
        # 첫 호출 시 컴파일 → 이후부터 NPU 실행
        sobel = torch.compile(
            base,
            backend="rbln",
            options={"cache_dir": "./.rbln_cache"},
            dynamic=False,  # 정적 형태 권장
        )
    else:
        sobel = base.to(device)

    total_torch = 0.0
    total_cv2 = 0.0
    shown = 0
    warmed_up = False

    # Batch loop
    for img_batch_cpu, path_batch in loader:
        # ---- Torch (GPU/CPU/NPU) ----
        if args.accel == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.inference_mode():
            if args.accel == "rbln":
                # RBLN은 CPU 텐서를 그대로 넘깁니다 (런타임 내부에서 NPU로 처리)
                out_batch = sobel(img_batch_cpu)  # float32, [B,1,H,W]
            else:
                img_batch = img_batch_cpu.to(device, non_blocking=True)
                out_batch = sobel(img_batch)

        if args.accel == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0)

        # 첫 배치는 컴파일/캐시 생성시간 포함 → 평균 제외
        if args.accel == "rbln" and not warmed_up:
            warmed_up = True
        else:
            total_torch += dt

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
