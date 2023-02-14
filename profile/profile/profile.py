import ctypes
import time
from tempfile import TemporaryDirectory

import tensorrt as trt
import torch
from typeo import scriptify
from hermes.quiver import ModelRepository, Platform

from profile.architecture import ResNet


def export_model(
    batch_size: int,
    num_channels: int,
    frame_length: int,
    batch_norm: bool = False,
    use_fp16: bool = False
):
    nn = ResNet(
        num_ifos=num_channels,
        layers=[2, 2, 2, 2],
        kernel_size=7,
        norm_groups=0 if batch_norm else 8
    )
    with TemporaryDirectory() as tmpdir:
        repo = ModelRepository(tmpdir)
        model = repo.add("model", platform=Platform.TENSORRT)
        binary_path = model.export_version(
            nn,
            input_shapes={"input": [batch_size, num_channels, frame_length]},
            output_names=["output"],
            use_fp16=use_fp16
        )
        with open(binary_path, "rb") as f:
            return f.read()


def benchmark(
    engine: trt.ICudaEngine,
    batch_size: int,
    num_channels: int,
    frame_length: int,
    N: int
) -> None:
    """Use a TensorRT engine to perform inference on a test input"""
    with engine.create_execution_context() as context:
        stream = torch.cuda.Stream()
        cuda_stream = stream.cuda_stream
        stream_ptr = ctypes.c_void_p(cuda_stream).value

        input = torch.randn(batch_size, num_channels, frame_length).to("cuda")
        output = torch.zeros((128, 1), dtype=torch.float32, device="cuda")
        context.set_tensor_address("input", input.data_ptr())
        context.set_tensor_address("output", output.data_ptr())

        # execute inference on device
        start_time = time.time()
        for i in range(N):
            context.execute_async_v3(stream_ptr)

        # synchronize the execution stream
        stream.synchronize()
        end_time = time.time()
        throughput = batch_size * N / (end_time - start_time)
        return throughput


@scriptify
def main(
    batch_size: int = 128,
    num_channels: int = 2,
    frame_length: int = 2048,
    N: int = 10000
) -> None:
    print(f"Using TensorRT version {trt.__version__}")

    trt.init_libnvinfer_plugins(None, "")
    for batch_norm in [True, False]:
        for use_fp16 in [True, False]:
            print(
                "Benchmarking with batch_norm={} and use_fp16={}".format(
                    batch_norm, use_fp16
                )
            )
            print("\tCreating serialized engine")
            serialized_engine = export_model(
                batch_size, num_channels, frame_length, batch_norm, use_fp16
            )

            # build a TensorRT engine and do some inference
            # on our dummy input using it
            trt_logger = trt.Logger()
            with trt.Runtime(trt_logger) as runtime:
                engine = runtime.deserialize_cuda_engine(serialized_engine)
                print(f"\tRunning benchmark for {N} iterations")
                throughput = benchmark(
                    engine, batch_size, num_channels, frame_length, N
                )
                print(f"\tThroughput: {throughput} inf/s")


if __name__ == "__main__":
    main()