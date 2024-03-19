
import numpy as np
import onnx
import onnxruntime
from PIL import Image
from pathlib import Path

from dabox_research.env import DEMO_DIR, DEFAULT_OUTPUT_DIR
from dabox.util.drawing import draw_detections
import torch
import torchvision.transforms as transforms

class TransposeResizeNormalize(torch.nn.Module):
    def __init__(
        self, 
        resize, 
        mean_values=(0, 0, 0), 
        scale_factor=[255, 255, 255]
    ):
        super(TransposeResizeNormalize, self).__init__()
        self.resize = transforms.Resize(
            resize,
            antialias=True,
            interpolation=transforms.InterpolationMode.NEAREST)
        self.normalize = transforms.Normalize(
            mean=mean_values,
            std=scale_factor)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        x = self.normalize(x)
        return x

def preproc_onnx():
    model_prep = TransposeResizeNormalize(resize=(480, 640))

    dummy_input = torch.randn(1, 720, 1280, 3)

    dynamic = {'input': {0: 'batch', 1: 'height', 2: 'width'},
                'output': {0 : 'batch'}}

    path_export_model_prep = 'prep.onnx'

    torch.onnx.export(model_prep,
                    dummy_input,
                    path_export_model_prep,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic,
                    verbose=True)

def edit_onnx_model(model_path: Path) -> Path:
    preproc_onnx()
    prep = onnx.load('prep.onnx')
    model = onnx.load(model_path)

    # add prefix, resolve names conflits
    prep_with_prefix = onnx.compose.add_prefix(prep, prefix="prep_")

    model_prep = onnx.compose.merge_models(
        prep_with_prefix,
        model,    
        io_map=[('prep_output', # output prep model
                'images')])     # input yolov8 model

    onnx_path = DEFAULT_OUTPUT_DIR / "model+preproc.onnx"
    onnx.save(model_prep, onnx_path)
    return onnx_path
    

def main():
    model_path = DEFAULT_OUTPUT_DIR / "onnx" / "yolov8n.onnx"
    onnx_path = edit_onnx_model(model_path)

    providers = onnxruntime.get_available_providers()
    # Disable Tensorrt because it is slow to startup
    if "TensorrtExecutionProvider" in providers:
        providers.remove("TensorrtExecutionProvider")
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_names = [model_input.name for model_input in session.get_inputs()]
    output_names = [model_output.name for model_output in session.get_outputs()]
    print(input_names, output_names)

    image = Image.open(DEMO_DIR/ "image0.png").convert("RGB")
    input_tensor = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
    print(input_tensor.shape, input_tensor.dtype)

    inputs = {input_name: input_tensor for input_name, input_tensor in zip(input_names, [input_tensor])}
    output_tensors = session.run(output_names, inputs)
    outputs = {output_name: output_tensor for output_name, output_tensor in zip(output_names, output_tensors)}

    predictions = np.squeeze(outputs["output0"]).T
    conf_threshold = 0.5

    boxes = []
    scores = []
    class_ids = []

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]
    boxes = xywh2xyxy(boxes)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # indices = nms(boxes, scores, self.iou_threshold)
    indices = multiclass_nms(boxes, scores, class_ids)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    image_vis = draw_detections(np.array(image), boxes, scores, class_ids)
    image_vis = Image.fromarray(image_vis)
    image_vis.save(DEFAULT_OUTPUT_DIR / "image0_vis.png")

def multiclass_nms(boxes, scores, class_ids, iou_threshold=0.5):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    return keep_boxes


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

    

if __name__ == "__main__":
    main()