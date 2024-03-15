import shutil
from pathlib import Path
from ultralytics import YOLO

from dabox_research.env import DEFAULT_OUTPUT_DIR

def move_file(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src_path, dst_path)

def main():
    model_names = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    input_size = (640, 480)
    for model_name in model_names:
        model = YOLO(f"{model_name}.pt")
        model.export(format='onnx',imgsz=[input_size[1], input_size[0]], half=True, device=0, simplify=True)

        move_file(f"{model_name}.pt", DEFAULT_OUTPUT_DIR / "pt" / f"{model_name}.pt")
        move_file(f"{model_name}.onnx", DEFAULT_OUTPUT_DIR / "onnx" / f"{model_name}.onnx")

if __name__ == "__main__":
    main()
