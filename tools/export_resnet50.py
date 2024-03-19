import torch
import torchvision.models as models

def main():
    model = models.resnet50(pretrained=True)
    model.eval()

    input_dim = (1, 3, 224, 224)
    dummy_input = torch.randn(input_dim)

    onnx_path = 'resnet50.onnx'
    dynamic = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(model, 
                    dummy_input, 
                    onnx_path, 
                    verbose=True, 
                    input_names=['input'], 
                    output_names=['output'], 
                    dynamic_axes=dynamic,
                    opset_version=17)

if __name__ == '__main__':
    main()
    