import torch

import utils
from onnxexport.model_onnx import SynthesizerTrn
import onnxruntime as ort
import numpy as np


def onnxruntime_test(onnx_model_path, test_input):
    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Prepare input data as a dictionary
    input_dict = {}
    for i, inp in enumerate(ort_session.get_inputs()):
        print(inp.name)
        input_dict[inp.name] = test_input[
            i
        ]  # Assuming test_input[i] is already a numpy array

    # Run ONNX inference
    predictions = ort_session.run(None, input_dict)

    return predictions


def main(NetExport):
    if NetExport:
        device = torch.device("cpu")
        hps = utils.get_hparams_from_file(
            f"/home/alexander/Projekte/so-vits-svc/logs/22k/config.json"
        )
        SVCVITS = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )
        _ = utils.load_checkpoint(
            f"/home/alexander/Projekte/so-vits-svc/logs/22k/G_216000.pth", SVCVITS, None
        )
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False

        n_frame = 120
        test_hidden_unit = torch.rand(1, n_frame, 768)
        test_pitch = torch.rand(1, n_frame)
        test_mel2ph = torch.arange(0, n_frame, dtype=torch.int64)[
            None
        ]  # torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
        test_uv = torch.ones(1, n_frame, dtype=torch.float32)
        test_noise = torch.FloatTensor([0.5])
        test_sid = torch.randn(1, 512)
        input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
        output_names = [
            "audio",
        ]

        torch.onnx.export(
            SVCVITS,
            (
                test_hidden_unit.to(device),
                test_pitch.to(device),
                test_mel2ph.to(device),
                test_uv.to(device),
                test_noise.to(device),
                test_sid.to(device),
            ),
            "/home/alexander/Projekte/so-vits-svc/logs/22k/model.onnx",
            dynamic_axes={
                "c": [0, 1],
                "f0": [1],
                "mel2ph": [1],
                "uv": [1],
            },
            do_constant_folding=False,
            opset_version=16,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )

        # validate onnx model
        import onnx

        onnx_model_path = "/home/alexander/Projekte/so-vits-svc/logs/22k/model.onnx"
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        # Adding ONNX runtime test
        test_input = [
            torch.rand(1, 120, 768),
            torch.rand(1, 120),
            torch.arange(0, 120, dtype=torch.int64)[None],
            torch.ones(1, 120, dtype=torch.float32),
            torch.FloatTensor([0.4]),
            torch.randn(1, 512),
        ]

        # Replace torch tensors with numpy ndarrays for ONNX runtime
        test_input_np = [tensor.numpy() for tensor in test_input]

        # Test
        predictions = onnxruntime_test(onnx_model_path, test_input_np)
        print(predictions[0].shape)


if __name__ == "__main__":
    main(True)
