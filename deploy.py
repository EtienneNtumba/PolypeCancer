import tensorrt as trt

# Convert to TensorRT
trt_cmd = f"""
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --explicitBatch \
        --workspace=4096
"""
subprocess.run(rt_cmd, shell=True)

# Inference class
class EndoscopyAIAssistant:
    def process_frame(self, frame):
        preprocessed = self.transforms(frame)
        outputs = self.trt_engine.infer(preprocessed)
        return self.postprocess(outputs)
