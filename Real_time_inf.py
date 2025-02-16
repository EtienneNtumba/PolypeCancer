# inference.py
import cv2
import torch
import tensorrt as trt

class RealTimePolypDetector:
    def __init__(self, model_path="deploy_model.pt"):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # TensorRT Initialization
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        with open("polyp_detector.engine", 'rb') as f:
            self.trt_runtime = trt.Runtime(self.trt_logger)
            self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())
        
    def process_frame(self, frame):
        # Preprocessing
        frame = cv2.resize(frame, (512, 512))
        frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            masks, boxes = self.model(frame_tensor)
        
        # Post-processing
        return self._parse_outputs(masks, boxes)
    
    def video_analysis(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            results = self.process_frame(frame)
            yield self._visualize_results(frame, results)
