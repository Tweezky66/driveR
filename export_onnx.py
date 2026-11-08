import torch
import onnx
from sim.depth.model import DepthNet
from visualization.icon_map import PROJECT_ROOT


def export_onnx(pt_path,  onnx_path):
    model =  DepthNet(freeze_encoder=False).cpu()

    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state, strict=True)

    model.eval()

    dummy_input = torch.randn(1, 3,  352,  640)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        opset_version=18,
        input_names=["input_rgb"],
        output_names=["output_log_depth"],
        dynamo=True,
        verify=True,
        external_data=False,
    )

    onnx.checker.check_model(str(onnx_path))
    print(f"Onnx export completed succesfully to {onnx_path}")

if __name__ == "__main__":
    export_onnx(
        pt_path=PROJECT_ROOT / "new_best_depth.pt",
        onnx_path=PROJECT_ROOT /  "depth_model.onnx"
    )