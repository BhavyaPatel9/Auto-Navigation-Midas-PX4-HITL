import tensorrt as trt
import os


def build_engine(onnx_file_path, engine_file_path):
    # 1. Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 2. Initialize the builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)

    # EXPLICIT_BATCH flag is required for modern ONNX models
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 3. Parse the ONNX model
    print(f"Parsing ONNX file: {onnx_file_path}")
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return False

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 4. Configure the builder
    config = builder.create_builder_config()

    # Set the maximum workspace size (e.g., 1GB).
    # Jetson Nano has 4GB shared memory; 1GB is usually safe.
    # Note: For TRT 8.4+, max_workspace_size is deprecated in favor of set_memory_pool_limit,
    # but Jetson Nano (JetPack 4.6.1) runs TRT 8.2.
    config.max_workspace_size = 1 << 30

    # Enable FP16 precision if supported (Jetson Nano supports this!)

    # 5. Build the serialized engine
    print("Building the TensorRT engine. This might take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return False

    # 6. Save the engine to a file
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Successfully saved engine to {engine_file_path}")
    return True


# --- Execution ---
if __name__ == "__main__":
    ONNX_PATH = "/home/swayaan/Build_Tensor_RT/onnx_model/phase_4_new_aug.onnx"  # Replace with your ONNX model path
    ENGINE_PATH = "/home/swayaan/Build_Tensor_RT/Build_Engine/phase_4_new_aug.engine"  # Replace with your desired engine output path

    build_engine(ONNX_PATH, ENGINE_PATH)