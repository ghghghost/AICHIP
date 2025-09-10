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

# --- Furiosa Warboy SDK ---
from furiosa.runtime import sync as frt            # create_runner / Runner API
from furiosa.quantizer import Calibrator, CalibrationMethod, quantize
import onnx
from onnx import checker, shape_inference

try:
    from onnxsim import simplify  # 선택
    HAVE_ONNXSIM = True
except Exception:
    HAVE_ONNXSIM = False

# import rebel


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
                   help="배치 크기")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers, 벤치용 0 권장")
    p.add_argument("--pin_memory", action="store_true",
                   help="DataLoader pin_memory 활성화 (GPU에서 권장)")
    p.add_argument("--debug", action="store_true",
                   help="중간 시각화")
    p.add_argument("--debug_limit", type=int, default=3,
                   help="디버그 시 최대 몇 배치/이미지까지 표시할지")
    p.add_argument("--accel", type=str, required=True, default='cpu',
                   choices=["cpu", "cuda", "rbln", "warboy"],
                   help="실행 백엔드: cpu/cuda/rbln/warboy")
    p.add_argument("--skip_cv2", action="store_true",
                   help="OpenCV 기준선 측정을 생략 (NPU만 순수 측정)")
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
        paths = sorted(glob(root_arg, recursive=True))
    return paths


# ---------- Warboy 전용: ONNX export / INT8 양자화 / Runner ----------
def _sanitize_names(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """이름에 '/' 등이 섞여 있으면 '_'로 치환 (node/tensor/initializer/value_info/graph io 전부)"""
    def fix(s: str) -> str:
        return s.replace('/', '_') if isinstance(s, str) else s

    for n in onnx_model.graph.node:
        n.name = fix(n.name)
        n.input[:]  = [fix(x) for x in n.input]
        n.output[:] = [fix(x) for x in n.output]
    for vi in list(onnx_model.graph.value_info) + list(onnx_model.graph.input) + list(onnx_model.graph.output):
        vi.name = fix(vi.name)
    for init in onnx_model.graph.initializer:
        init.name = fix(init.name)
    return onnx_model


def export_sobel_to_onnx(model: torch.nn.Module, H: int, W: int, onnx_path: str):
    model = model.eval()
    dummy = torch.zeros(1, 1, H, W, dtype=torch.float32)

    # 1) 내보내기: PyTorch 새 ONNX exporter(dynamo=True) 시도, 실패 시 레거시 fallback
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["out"],
            opset_version=13, do_constant_folding=True,
            dynamic_axes=None, training=torch.onnx.TrainingMode.EVAL,
            dynamo=True  # 새 경로
        )
    except TypeError:
        # 오래된 torch면 dynamo 인자를 모를 수 있음 → 레거시 경로로
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["out"],
            opset_version=13, do_constant_folding=True,
            dynamic_axes=None, training=torch.onnx.TrainingMode.EVAL
        )

    # 2) 로드 → shape inference → 체크 → (선택) simplify → 이름 위생 → 저장
    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m)
    checker.check_model(m)

    if HAVE_ONNXSIM:
        m, ok = simplify(m)
        if ok:
            pass  # simplified 성공
    m = _sanitize_names(m)  # '/filter/...' 같은 이름 치환
    onnx.save(m, onnx_path)

    return onnx_path


def export_sobel_to_onnx(model: torch.nn.Module, H: int, W: int, onnx_path: str):
    model = model.eval()
    dummy = torch.zeros(1, 1, H, W, dtype=torch.float32)  # batch=1 정적
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["out"],
        opset_version=13,           # Warboy 권장 opset
        do_constant_folding=True,
        dynamic_axes=None           # 정적 형태
    )
    return onnx_path


def quantize_onnx_int8(onnx_path: str, calib_iter, q_path: str,
                       method=CalibrationMethod.MSE_ASYM):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    model = _sanitize_names(model)
    checker.check_model(model)

    calib = Calibrator(model, calibration_method=method, percentage=99.99)
    calib.collect_data(calib_iter)     # iterable of [np.ndarray]
    ranges = calib.compute_range(verbose=False)

    q_bytes = quantize(model, ranges)
    with open(q_path, "wb") as f:
        f.write(q_bytes)
    return q_path


def create_warboy_runner(model_path: str, batch_size: int, device_spec: str = None):
    if device_spec:
        os.environ["FURIOSA_DEVICES"] = device_spec  # 예: "warboy(2)*1"
    runner = frt.create_runner(model_path, batch_size=batch_size)
    return runner
# ---------------------------------------------------------------------


def main():
    args = parse_args()

    if args.accel == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA 사용 불가. CPU로 대체합니다.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    paths = collect_image_paths(args.root)
    if not paths:
        print(f"[ERROR] 이미지가 없습니다. --root에 디렉터리 또는 패턴을 주세요. "
              f"예) /data/val2017/*.jpg 혹은 /data/val2017/**/*.jpg")
        return

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
        drop_last=True,  # 정적 형태 유지(러너/컴파일 캐시 안정)
    )

    base = Sobel(args.constant).eval()

    sobel = None
    sobel_runner = None
    if args.accel == "warboy":
        # 1) ONNX 내보내기
        os.makedirs("./.warboy_cache", exist_ok=True)
        onnx_path = f"./.warboy_cache/sobel_b1_{args.resize}x{args.resize}.onnx"
        if not os.path.exists(onnx_path):
            export_sobel_to_onnx(base, args.resize, args.resize, onnx_path)  # :contentReference[oaicite:1]{index=1}

        # 2) 간단 캘리브레이션 후 INT8 양자화 (PTQ)  :contentReference[oaicite:2]{index=2}
        def calib_iter_gen(max_images=256):
            used = 0
            calib_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
            for t_cpu, _ in calib_loader:
                yield [t_cpu.numpy()]   # 입력은 numpy 배열 리스트
                used += 1
                if used >= max_images:
                    break

        q_onnx_path = f"./.warboy_cache/sobel_b1_{args.resize}x{args.resize}_i8.onnx"
        if not os.path.exists(q_onnx_path):
            quantize_onnx_int8(onnx_path, calib_iter_gen(), q_onnx_path)   # :contentReference[oaicite:3]{index=3}

        # 3) 러너 생성 (배치는 여기서 지정)  :contentReference[oaicite:4]{index=4}
        # 장치 지정이 필요하면 device_spec="warboy(2)*1" 처럼 넘기기
        sobel_runner = create_warboy_runner(q_onnx_path, batch_size=args.batch_size, device_spec=None)

    elif args.accel == "rbln":
        sobel = torch.compile(
            base,
            backend="rbln",
            options={"cache_dir": "./.rbln_cache"},
            dynamic=False,
        )
    else:
        sobel = base.to(device)

    total_torch = 0.0
    total_cv2 = 0.0
    shown = 0
    warmed_up = False  # rbln 첫 배치 제외

    for img_batch_cpu, path_batch in loader:
        if args.accel == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        if args.accel == "warboy":
            # 러너 생성 시 고정한 batch_size와 다르면 건너뜀(간단 처리)
            if img_batch_cpu.shape[0] != args.batch_size:
                continue
            # Warboy 입력/출력: list-of-ndarray
            out0 = sobel_runner.run([img_batch_cpu.numpy()])[0]  # (B,1,H,W), dtype는 양자화 설정에 따라 다를 수 있음
            out_batch = torch.from_numpy(out0)  # 아래 디버그 공통화를 위해 텐서로 캐스팅
            total_torch += (time.time() - t0)

        else:
            with torch.inference_mode():
                if args.accel == "rbln":
                    out_batch = sobel(img_batch_cpu)  # CPU 텐서 그대로
                else:
                    img_batch = img_batch_cpu.to(device, non_blocking=True)
                    out_batch = sobel(img_batch)

            if args.accel == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0)

            # rbln은 첫 배치에 컴파일/캐시가 포함될 수 있으므로 평균에서 제외
            if args.accel == "rbln" and not warmed_up:
                warmed_up = True
            else:
                total_torch += dt

        # ---- OpenCV baseline (CPU) ----
        if not args.skip_cv2:
            t0 = time.time()
            imgs_np = img_batch_cpu.squeeze(1).numpy().astype(np.uint8)
            for i in range(imgs_np.shape[0]):
                sobel_x = cv2.Sobel(imgs_np[i], cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(imgs_np[i], cv2.CV_32F, 0, 1, ksize=3)
                _ = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            total_cv2 += (time.time() - t0)

        # ---- Debug visualize ----
        if args.debug and shown < args.debug_limit:
            gray_vis = imgs_np[0].astype(np.float32) / 255.0
            out0_np = out_batch[0, 0].detach().cpu().numpy()
            sx = cv2.Sobel(imgs_np[0], cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(imgs_np[0], cv2.CV_32F, 0, 1, ksize=3)
            cv2_vis = np.sqrt(np.square(sx) + np.square(sy))

            concat = np.concatenate([gray_vis, safe_norm(out0_np), safe_norm(cv2_vis)], axis=1)
            plt.figure(figsize=(10, 4))
            title_backend = "Warboy" if args.accel == "warboy" else ("Rebellion" if args.accel == "rbln" else args.accel)
            plt.imshow(concat, cmap="gray")
            plt.title(f"Input | {title_backend} Sobel | OpenCV Sobel\n{path_batch[0]}")
            plt.axis("off")
            plt.show()
            shown += 1

    # 평균 산출 (rbln 첫 배치 제외 반영)
    eff_images = len(paths) - (args.batch_size if args.accel == "rbln" else 0)
    if eff_images <= 0:
        eff_images = len(paths)

    print(f"[DONE] images: {len(paths)} | total torch(npu): {total_torch:.6f}s | total cv2: {total_cv2:.6f}s")
    if len(paths) > 0:
        print(f"[MEAN] per-image torch(npu): {total_torch/eff_images:.6f}s | per-image cv2: {total_cv2/len(paths):.6f}s")


if __name__ == "__main__":
    main()
