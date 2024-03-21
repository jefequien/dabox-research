from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image
from tqdm import tqdm

from dabox_research.env import DEFAULT_OUTPUT_DIR, DEMO_DIR
from dabox_research.util.drawing import draw_detections


def save_image(image: Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    model_names = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    input_size = (640, 480)

    for model_name in model_names:
        export_dir = DEFAULT_OUTPUT_DIR / "export" / model_name
        export_onnx_path = export_dir / f"dabox_{model_name}.onnx"

        providers = onnxruntime.get_available_providers()
        # Disable Tensorrt because it is slow to startup
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        session = onnxruntime.InferenceSession(export_onnx_path, providers=providers)
        input_names = [model_input.name for model_input in session.get_inputs()]
        output_names = [model_output.name for model_output in session.get_outputs()]
        print(input_names, output_names)

        image = Image.open(DEMO_DIR / "image0.png").convert("RGB")
        image = np.array(image.resize(input_size))
        for idx in tqdm(range(1000)):
            inputs = {
                input_name: input_tensor
                for input_name, input_tensor in zip(input_names, [image])
            }
            output_tensors = session.run(output_names, inputs)
            outputs = {
                output_name: output_tensor
                for output_name, output_tensor in zip(output_names, output_tensors)
            }

            if idx == 0:
                det_bboxes = outputs["det_bboxes"][0]
                det_scores = outputs["det_scores"][0]
                det_classes = outputs["det_classes"][0]

                image_vis = draw_detections(image, det_bboxes, det_scores, det_classes)
                image_vis = Image.fromarray(image_vis)
                save_image(
                    image_vis, DEFAULT_OUTPUT_DIR / "infer" / f"{model_name}.png"
                )


if __name__ == "__main__":
    main()
