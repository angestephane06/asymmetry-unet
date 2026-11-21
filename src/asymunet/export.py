"""ONNX export, static INT8 quantization, and FP32 vs INT8 comparison on CPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from asymunet.metrics import summarize
from asymunet.model import UNet
from asymunet.train import load, normalise


class _Calibration(CalibrationDataReader):
    def __init__(self, x: np.ndarray, name: str) -> None:
        self._it = iter([{name: x[i : i + 1]} for i in range(len(x))])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._it, None)


def _run(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    logits = np.concatenate(
        [session.run(None, {name: x[i : i + 64]})[0] for i in range(0, len(x), 64)]
    )
    return 1 / (1 + np.exp(-logits[:, 0]))


def _latency_ms(session: ort.InferenceSession, x: np.ndarray, runs: int = 300) -> float:
    name = session.get_inputs()[0].name
    one = x[:1]
    for _ in range(20):
        session.run(None, {name: one})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, {name: one})
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def export_and_quantize(data_dir: Path, out_dir: Path) -> dict[str, object]:
    model = UNet()
    model.load_state_dict(torch.load(out_dir / "unet.pt", map_location="cpu"))
    model.eval()
    fp32 = out_dir / "unet_fp32.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 64, 64),),
        str(fp32),
        input_names=["maps"],
        output_names=["logits"],
        dynamic_axes={"maps": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    va_x, _ = load(data_dir / "val.npz")
    te_x, te_y = load(data_dir / "test.npz")
    va_xn, te_xn = normalise(va_x), normalise(te_x)
    prep = out_dir / "unet_prep.onnx"
    quant_pre_process(str(fp32), str(prep))
    int8 = out_dir / "unet_int8.onnx"
    quantize_static(
        str(prep),
        str(int8),
        _Calibration(va_xn[:200], "maps"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prep.unlink()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # one thread: a fair, machine-independent comparison
    valid = te_x[:, 2] > 0
    report: dict[str, object] = {}
    with torch.no_grad():
        torch_prob = torch.sigmoid(model(torch.from_numpy(te_xn[:64]))).numpy()[:, 0]
    for label, path in (("fp32", fp32), ("int8", int8)):
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        prob = _run(session, te_xn)
        report[label] = {
            **summarize(prob > 0.5, te_y > 0.5, valid),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "cpu_latency_ms_batch1_1thread": round(_latency_ms(session, te_xn), 3),
            "max_abs_diff_vs_pytorch": float(np.abs(prob[:64] - torch_prob).max()),
        }
    (out_dir / "onnx_results.json").write_text(json.dumps(report, indent=2))
    return report
