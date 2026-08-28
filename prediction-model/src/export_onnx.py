"""
ONNX Export Pipeline for Liquid Neural Network.
Exports the PyTorch LNN model to standard ONNX format for zero-overhead
serverless execution in Node.js / Python without heavy framework dependencies.
"""

import os
import torch
from model import WeatherWaterLNN


def export_lnn_to_onnx(
    checkpoint_path: str = "lnn_weather_water.pt",
    onnx_output_path: str = "lnn_model.onnx"
):
    print("=" * 60)
    print("Exporting LNN Model to ONNX format...")
    print("=" * 60)

    model = WeatherWaterLNN(input_dim=4, hidden_dim=32)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained checkpoint: {checkpoint_path}")
    else:
        print("Checkpoint not found, initializing with default initial weights.")

    model.eval()

    # Dummy inputs: batch=1, sequence=24 steps, 4 features
    dummy_telemetry = torch.randn(1, 24, 4, dtype=torch.float32)
    dummy_dt = torch.ones(1, 24, 1, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_telemetry, dummy_dt),
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["telemetry_sequence", "dt_sequence"],
        output_names=["rain_probability", "precipitation_mm", "predicted_water_level"],
        dynamic_axes={
            "telemetry_sequence": {0: "batch_size", 1: "sequence_length"},
            "dt_sequence": {0: "batch_size", 1: "sequence_length"},
            "rain_probability": {0: "batch_size", 1: "sequence_length"},
            "precipitation_mm": {0: "batch_size", 1: "sequence_length"},
            "predicted_water_level": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"Successfully exported ONNX model to: {os.path.abspath(onnx_output_path)}")
    print("Inputs:")
    print("  - telemetry_sequence: [batch, seq_len, 4] (temp, heat_index, wind, pressure)")
    print("  - dt_sequence: [batch, seq_len, 1] (time delta in hours)")
    print("Outputs:")
    print("  - rain_probability: [batch, seq_len, 1] (0.0 to 1.0)")
    print("  - precipitation_mm: [batch, seq_len, 1] (accumulated mm)")
    print("  - predicted_water_level: [batch, seq_len, 1] (meters)")
    print("=" * 60)


if __name__ == "__main__":
    export_lnn_to_onnx()
