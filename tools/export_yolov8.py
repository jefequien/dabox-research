from pathlib import Path

import onnx
import onnxsim
import torch
import torchvision
from onnx import TensorProto
from torchvision import transforms
from ultralytics import YOLO

from dabox_research.env import DEFAULT_OUTPUT_DIR
from dabox_research.util.file_io import move_file


class Preprocess(torch.nn.Module):
    def forward(self, x):
        x = transforms.functional.convert_image_dtype(x)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x


class Postprocess(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int]) -> None:
        super().__init__()
        self.input_size = input_size

    def forward(self, idxTensor, boxes, scores):
        bbox_result = self.gather(boxes, idxTensor)
        score_intermediate_result = self.gather(scores, idxTensor).max(axis=-1)
        score_result = score_intermediate_result.values
        classes_result = score_intermediate_result.indices

        bbox_result = torchvision.ops.box_convert(
            bbox_result, in_fmt="cxcywh", out_fmt="xyxy"
        )
        bbox_result[..., 0::2] /= self.input_size[0]
        bbox_result[..., 1::2] /= self.input_size[1]
        return (bbox_result, score_result, classes_result)

    def gather(self, target, idxTensor):
        pick_indices = idxTensor[:, -1:].repeat(1, target.shape[1]).unsqueeze(0)
        return torch.gather(target.permute(0, 2, 1), 1, pick_indices)


def make_preproc_onnx(input_size: tuple[int, int], export_dir: Path) -> Path:
    model_preproc = Preprocess()
    dummy_input = torch.randint(
        255, (input_size[1], input_size[0], 3), dtype=torch.uint8
    )

    onnx_path = export_dir / "preproc.onnx"
    torch.onnx.export(
        model_preproc,
        dummy_input,
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return onnx_path


def postproc_onnx(input_size: tuple[int, int], export_dir: Path) -> Path:
    model_postproc = Postprocess(input_size=input_size)

    torch_boxes = torch.tensor(
        [
            [91.0, 2, 3, 4, 5, 6],
            [11, 12, 13, 14, 15, 16],
            [21, 22, 23, 24, 25, 26],
            [31, 32, 33, 34, 35, 36],
        ]
    ).unsqueeze(0)

    torch_scores = torch.tensor(
        [
            [0.1, 0.82, 0.3, 0.6, 0.55, 0.6],
            [0.9, 0.18, 0.7, 0.4, 0.45, 0.4],
        ]
    ).unsqueeze(0)

    torch_indices = torch.tensor([[0, 0, 0], [0, 0, 2], [0, 0, 1]])
    onnx_path = export_dir / "postproc.onnx"
    torch.onnx.export(
        model_postproc,
        (torch_indices, torch_boxes, torch_scores),
        onnx_path,
        input_names=["selected_indices", "boxes", "scores"],
        output_names=["det_bboxes", "det_scores", "det_classes"],
        dynamic_axes={
            "boxes": {0: "batch", 1: "boxes", 2: "num_anchors"},
            "scores": {0: "batch", 1: "classes", 2: "num_anchors"},
            "selected_indices": {0: "num_results"},
            "det_bboxes": {1: "num_results"},
            "det_scores": {1: "num_results"},
            "det_classes": {1: "num_results"},
        },
    )
    return onnx_path


def add_preprocessing_to_onnx(
    onnx_path: Path, input_size: tuple[int, int], export_dir: Path
) -> Path:
    preproc_onnx_path = make_preproc_onnx(input_size, export_dir)
    preproc_model = onnx.load(preproc_onnx_path)
    onnx_model = onnx.load(onnx_path)

    # add prefix, resolve names conflits
    preproc_model_with_prefix = onnx.compose.add_prefix(
        preproc_model, prefix="preproc_"
    )
    onnx_model = onnx.compose.merge_models(
        preproc_model_with_prefix, onnx_model, io_map=[("preproc_output", "images")]
    )
    onnx_path = export_dir / "preproc+model.onnx"
    onnx.save(onnx_model, onnx_path)
    return onnx_path


def add_postprocessing_to_onnx(
    onnx_path: Path, input_size: tuple[int, int], export_dir: Path
) -> Path:
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    # operation to transpose bbox before pass to NMS node
    transpose_bboxes_node = onnx.helper.make_node(
        "Transpose",
        inputs=["/model.22/Mul_2_output_0"],
        outputs=["bboxes"],
        perm=(0, 2, 1),
    )
    graph.node.append(transpose_bboxes_node)

    # make constant tensors for nms
    score_threshold = onnx.helper.make_tensor(
        "score_threshold", TensorProto.FLOAT, [1], [0.5]
    )
    iou_threshold = onnx.helper.make_tensor(
        "iou_threshold", TensorProto.FLOAT, [1], [0.5]
    )
    max_output_boxes_per_class = onnx.helper.make_tensor(
        "max_output_boxes_per_class", TensorProto.INT64, [1], [200]
    )

    # create the NMS node
    nms_node = onnx.helper.make_node(
        "NonMaxSuppression",
        inputs=[
            "bboxes",
            "/model.22/Sigmoid_output_0",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        outputs=["selected_indices"],
        center_point_box=1,
    )
    graph.node.append(nms_node)

    # add to initializers - without this, onnx will not know where these came from, and complain that
    # they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated
    # as constants needed for the NMS op
    graph.initializer.append(score_threshold)
    graph.initializer.append(iou_threshold)
    graph.initializer.append(max_output_boxes_per_class)

    # remove the unused nodes
    last_concat_node = [
        node for node in onnx_model.graph.node if node.name == "/model.22/Concat_5"
    ][0]
    graph.node.remove(last_concat_node)
    output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
    graph.output.remove(output0)

    # append to the output
    output_value_info = onnx.helper.make_tensor_value_info(
        "selected_indices", TensorProto.INT64, shape=["num_results", 3]
    )
    graph.output.append(output_value_info)
    output_value_info = onnx.helper.make_tensor_value_info(
        "/model.22/Mul_2_output_0",
        TensorProto.FLOAT,
        shape=["batch", "boxes", "num_anchors"],
    )
    graph.output.append(output_value_info)
    output_value_info = onnx.helper.make_tensor_value_info(
        "/model.22/Sigmoid_output_0",
        TensorProto.FLOAT,
        shape=["batch", "classes", "num_anchors"],
    )
    graph.output.append(output_value_info)

    # check that it works and re-save
    onnx.checker.check_model(onnx_model)
    onnx_path = export_dir / "preproc+model+nms.onnx"
    onnx.save(onnx_model, onnx_path)

    # Compose with postproc onnx
    postproc_onnx_path = postproc_onnx(input_size, export_dir)
    postproc_onnx_model = onnx.load_model(postproc_onnx_path)
    onnx_model = onnx.compose.merge_models(
        onnx_model,
        postproc_onnx_model,
        io_map=[
            ("/model.22/Mul_2_output_0", "boxes"),
            ("/model.22/Sigmoid_output_0", "scores"),
            ("selected_indices", "selected_indices"),
        ],
    )

    onnx.checker.check_model(onnx_model)
    onnx_path = export_dir / "preproc+model+nms+postproc.onnx"
    onnx.save(onnx_model, onnx_path)
    return onnx_path


def main():
    model_names = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    input_size = (640, 480)

    for model_name in model_names:
        export_dir = DEFAULT_OUTPUT_DIR / "export" / model_name
        export_onnx_path = export_dir / f"dabox_{model_name}.onnx"
        
        model = YOLO(f"{model_name}.pt")
        model.export(format="onnx", imgsz=[input_size[1], input_size[0]])
        pt_path = export_dir / f"{model_name}.pt"
        onnx_path = export_dir / f"{model_name}.onnx"
        move_file(f"{model_name}.pt", pt_path)
        move_file(f"{model_name}.onnx", onnx_path)

        onnx_path = add_preprocessing_to_onnx(onnx_path, input_size, export_dir)
        onnx_path = add_postprocessing_to_onnx(onnx_path, input_size, export_dir)

        # Simplify and convert model to float16
        sim_model, check = onnxsim.simplify(onnx.load_model(onnx_path))
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(sim_model, export_onnx_path)


if __name__ == "__main__":
    main()
