"""
YOLO 서비스 (선택 과제 A)

input/ 폴더의 사진마다
  - 객체 탐지(yolo26n.pt): 클래스·점수·박스 좌표
  - 이미지 분할(yolo26n-seg.pt): 클래스·점수·박스·마스크
를 conf=0.25 와 conf=0.50 두 임계값으로 실행하고 outputs/yolo/ 에 저장한다.

실행:  python yolo_service.py            (input/ 의 모든 jpg/png)
      python yolo_service.py 사진.jpg   (특정 파일만)
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "input"
OUT_DIR = ROOT / "outputs" / "yolo"

IMGSZ = 640
DEVICE = "cpu"
CONFS = [0.10, 0.25, 0.50]  # 0.25·0.50 = 과제 비교용, 0.10 = 임계값 아래 후보를 보기 위한 참고용
MODELS = {"detect": "yolo26n.pt", "segment": "yolo26n-seg.pt"}


def box_record(names, cls, conf, xyxy, w, h):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return {
        "class_id": int(cls),
        "class_name": names[int(cls)],
        "confidence": round(float(conf), 4),
        # 원본 이미지 픽셀 좌표 (왼쪽 위 x1,y1 / 오른쪽 아래 x2,y2)
        "xyxy_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        # 원본 크기로 나눈 0~1 정규화 좌표
        "xyxy_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
    }


def run_one(models, img_path: Path):
    stem = img_path.stem
    img_out = OUT_DIR / stem
    img_out.mkdir(parents=True, exist_ok=True)
    summary = {"image": img_path.name, "runs": {}}

    for task, model in models.items():
        for conf in CONFS:
            tag = f"{task}_conf{conf:.2f}"
            r = model.predict(str(img_path), imgsz=IMGSZ, device=DEVICE, conf=conf, verbose=False)[0]
            h, w = r.orig_shape
            names = r.names

            # 1) 결과 시각화 이미지 (박스 / 마스크를 원본 위에 겹침)
            cv2.imwrite(str(img_out / f"{tag}.jpg"), r.plot())

            # 2) 박스 좌표 JSON
            dets = []
            if r.boxes is not None:
                for i, (c, s, b) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)):
                    rec = box_record(names, c, s, b, w, h)
                    rec["id"] = i
                    dets.append(rec)

            # 3) 분할: 마스크 PNG (객체별) + 마스크 면적(px)
            if task == "segment" and r.masks is not None:
                mask_dir = img_out / f"{tag}_masks"
                mask_dir.mkdir(exist_ok=True)
                # masks.data 는 모델 입력 크기 → 원본 크기로 되돌려 저장
                for i, m in enumerate(r.masks.data.cpu().numpy()):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                    fname = f"{i:02d}_{dets[i]['class_name']}.png"
                    cv2.imwrite(str(mask_dir / fname), (m * 255).astype(np.uint8))
                    x1, y1, x2, y2 = dets[i]["xyxy_px"]
                    box_area = max(0.0, (x2 - x1) * (y2 - y1))
                    dets[i]["mask_file"] = f"{tag}_masks/{fname}"
                    dets[i]["mask_area_px"] = int(m.sum())
                    dets[i]["box_area_px"] = int(box_area)
                    dets[i]["mask_to_box_ratio"] = round(m.sum() / box_area, 3) if box_area else None

            payload = {
                "image": img_path.name, "image_size": [w, h], "task": task,
                "model": MODELS[task], "imgsz": IMGSZ, "device": DEVICE, "conf": conf,
                "count": len(dets), "detections": dets,
            }
            (img_out / f"{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["runs"][tag] = {
                "count": len(dets),
                "by_class": {n: sum(d["class_name"] == n for d in dets) for n in sorted({d["class_name"] for d in dets})},
            }

        # 4) 임계값 비교: 0.25 에서는 있었는데 0.50 에서 사라진 후보
        lo = json.loads((img_out / f"{task}_conf0.25.json").read_text(encoding="utf-8"))["detections"]
        dropped = [d for d in lo if d["confidence"] < 0.50]
        summary["runs"][f"{task}_dropped_between_0.25_and_0.50"] = [
            {"class_name": d["class_name"], "confidence": d["confidence"], "xyxy_px": d["xyxy_px"]} for d in dropped
        ]

    (img_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        p for p in IN_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {t: YOLO(str(ROOT / f)) if (ROOT / f).exists() else YOLO(f) for t, f in MODELS.items()}

    summaries = [run_one(models, p) for p in targets]
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False, indent=2))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [{"file": f"input/{p.name}"} for p in targets],
        "models": MODELS,
        "settings": {"imgsz": IMGSZ, "device": DEVICE, "conf": CONFS},
        "versions": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
        "run_location": {
            "where": "Claude(Cowork) 클라우드 샌드박스 — 리눅스 컨테이너, GPU 없음(CPU)",
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "outputs": "outputs/yolo/<사진이름>/ (결과 이미지 .jpg, 좌표 .json, 마스크 폴더, summary.json)",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
