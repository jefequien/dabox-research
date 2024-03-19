
import numpy as np
import onnx
from onnx import TensorProto

import onnxsim
import onnxruntime
from PIL import Image
from pathlib import Path
import time

from dabox_research.env import DEMO_DIR, DEFAULT_OUTPUT_DIR
from dabox_research.util.drawing import draw_detections
import torch
import torchvision
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

class Transform(torch.nn.Module):
    def forward(self, idxTensor, boxes, scores):
        bbox_result = self.gather(boxes, idxTensor)
        score_intermediate_result = self.gather(scores, idxTensor).max(axis=-1)
        score_result = score_intermediate_result.values
        classes_result = score_intermediate_result.indices
        num_dets = torch.tensor(score_result.shape[-1])

        bbox_result = torchvision.ops.box_convert(bbox_result, in_fmt = "cxcywh", out_fmt = "xyxy")
        bbox_result[..., 0::2] /= 640
        bbox_result[..., 1::2] /= 480
        return (bbox_result, score_result,  classes_result, num_dets)

    def gather(self, target, idxTensor):
        '''
        Input:
        boxes: [bs=1, 4, 8400]
        indices: [N, 3]

        expect output
        '''
        pick_indices = idxTensor[:,-1:].repeat(1,target.shape[1]).unsqueeze(0)
        return torch.gather(target.permute(0,2,1),1,pick_indices)

def make_preproc_onnx(export_dir: Path) -> Path:
    model_prep = TransposeResizeNormalize(resize=(480, 640))

    dummy_input = torch.randn(1, 720, 1280, 3)

    dynamic = {
        'input': {0: 'batch', 1: 'height', 2: 'width'},
        'output': {0 : 'batch'}
    }

    onnx_path = export_dir /'preproc.onnx'
    torch.onnx.export(model_prep,
                    dummy_input,
                    onnx_path,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic,
                    verbose=True)
    return onnx_path

def postproc_onnx(export_dir: Path) -> Path:
    torch_boxes = torch.tensor([
    [91.0,2,3,4,5,6],
    [11,12,13,14,15,16],
    [21,22,23,24,25,26],
    [31,32,33,34,35,36],
    ]).unsqueeze(0)

    torch_scores = torch.tensor([
    [0.1,0.82,0.3,0.6,0.55,0.6],
    [0.9,0.18,0.7,0.4,0.45,0.4],
    ]).unsqueeze(0)

    torch_indices = torch.tensor([[0,0,0], [0,0,2], [0,0,1]])
    t_model = Transform()
    onnx_path = export_dir / "postproc.onnx"
    torch.onnx.export(t_model, (torch_indices, torch_boxes, torch_scores), onnx_path,
                    input_names=["selected_indices", "boxes", "scores"], 
                    output_names=["det_bboxes", "det_scores", "det_classes", "num_dets"], 
                    dynamic_axes={
                        "boxes":{0:"batch",1:"boxes",2:"num_anchors"},
                        "scores":{0:"batch",1:"classes",2:"num_anchors"},
                        "selected_indices":{0:"num_results"},
                        "det_bboxes":{1:"num_results"},
                        "det_scores":{1:"num_results"},
                        "det_classes":{1:"num_results"},
                    })
    return onnx_path


def add_preprocessing_to_onnx(model_path: Path, export_dir: Path) -> Path:
    preproc_onnx_path = make_preproc_onnx(export_dir)
    preproc_model = onnx.load(preproc_onnx_path)
    model = onnx.load(model_path)

    # add prefix, resolve names conflits
    prep_with_prefix = onnx.compose.add_prefix(preproc_model, prefix="prep_")

    model_prep = onnx.compose.merge_models(
        prep_with_prefix,
        model,    
        io_map=[('prep_output', # output prep model
                'images')])     # input yolov8 model

    onnx_path = export_dir / "preproc+model.onnx"
    onnx.save(model_prep, onnx_path)
    return onnx_path

def add_postprocessing_to_onnx(onnx_path: Path, export_dir: Path) -> Path:
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    # operation to transpose bbox before pass to NMS node
    transpose_bboxes_node = onnx.helper.make_node("Transpose",inputs=["/model.22/Mul_2_output_0"],outputs=["bboxes"],perm=(0,2,1))
    graph.node.append(transpose_bboxes_node)

    # make constant tensors for nms
    score_threshold = onnx.helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.5])
    iou_threshold = onnx.helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5])
    max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [200])

    # create the NMS node
    inputs=['bboxes', "/model.22/Sigmoid_output_0", 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
    outputs = ["selected_indices"]
    nms_node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs,
        ["selected_indices"],
        # center_point_box=1 is very important, PyTorch model's output is 
        #  [x_center, y_center, width, height], but default NMS expect
        #  [x_min, y_min, x_max, y_max]
        center_point_box=1, 
    )

    # add NMS node to the list of graph nodes
    graph.node.append(nms_node)

    # append to the output (now the outputs would be scores, bboxes, selected_indices)
    output_value_info = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape=["num_results",3])
    graph.output.append(output_value_info)

    # add to initializers - without this, onnx will not know where these came from, and complain that 
    # they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated 
    # as constants needed for the NMS op
    graph.initializer.append(score_threshold)
    graph.initializer.append(iou_threshold)
    graph.initializer.append(max_output_boxes_per_class)

    # remove the unused concat node
    last_concat_node = [node for node in onnx_model.graph.node if node.name == "/model.22/Concat_5"][0]
    graph.node.remove(last_concat_node)

    # remove the original output0
    output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
    graph.output.remove(output0)

    # append to the output
    output_value_info = onnx.helper.make_tensor_value_info("/model.22/Mul_2_output_0", TensorProto.FLOAT, shape=["batch","boxes", "num_anchors"])
    graph.output.append(output_value_info)
    output_value_info = onnx.helper.make_tensor_value_info("/model.22/Sigmoid_output_0", TensorProto.FLOAT, shape=["batch","classes", "num_anchors"])
    graph.output.append(output_value_info)

    # check that it works and re-save
    onnx.checker.check_model(onnx_model)
    onnx_path = export_dir / "preproc+model+nms.onnx"
    onnx.save(onnx_model, onnx_path)

    # Compose with postproc onnx
    postproc_onnx_path = postproc_onnx(export_dir)
    postproc_onnx_model = onnx.load_model(postproc_onnx_path)
    # nms_postprocess_onnx_model_sim, check = onnxsim.simplify(nms_postprocess_onnx_model)
    # onnx.save(nms_postprocess_onnx_model_sim, "nms_sim.onnx")
    onnx_model = onnx.compose.merge_models(onnx_model, postproc_onnx_model, io_map=[
        ('/model.22/Mul_2_output_0', 'boxes'), 
        ('/model.22/Sigmoid_output_0', 'scores'),
        ('selected_indices', 'selected_indices')
    ])

    onnx.checker.check_model(onnx_model)
    onnx_path = export_dir / "preproc+model+nms+postproc.onnx"
    onnx.save(onnx_model, onnx_path)
    return onnx_path
    

def main():
    onnx_path = DEFAULT_OUTPUT_DIR / "onnx" / "yolov8n.onnx"
    export_dir = DEFAULT_OUTPUT_DIR / "export"
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = add_preprocessing_to_onnx(onnx_path, export_dir)
    onnx_path = add_postprocessing_to_onnx(onnx_path, export_dir)

    providers = onnxruntime.get_available_providers()
    # Disable Tensorrt because it is slow to startup
    if "TensorrtExecutionProvider" in providers:
        providers.remove("TensorrtExecutionProvider")
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_names = [model_input.name for model_input in session.get_inputs()]
    output_names = [model_output.name for model_output in session.get_outputs()]
    print(input_names, output_names)

    image = Image.open(DEMO_DIR/ "image0.png").convert("RGB")
    image = np.array(image)
    input_tensor = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
    print(input_tensor.shape, input_tensor.dtype)

    t0 = time.time()
    inputs = {input_name: input_tensor for input_name, input_tensor in zip(input_names, [input_tensor])}
    output_tensors = session.run(output_names, inputs)
    outputs = {output_name: output_tensor for output_name, output_tensor in zip(output_names, output_tensors)}
    t1 = time.time()
    print("Runtime", t1 - t0)

    det_bboxes = outputs["det_bboxes"][0]
    det_scores = outputs["det_scores"][0]
    det_classes = outputs["det_classes"][0]

    image_vis = draw_detections(image, det_bboxes, det_scores, det_classes)
    image_vis = Image.fromarray(image_vis)
    image_vis.save(export_dir / "image0_vis.png")

    

if __name__ == "__main__":
    main()