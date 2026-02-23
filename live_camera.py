import numpy as np
import cv2
import time
import psutil
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

HEF_PATH = "yolo26n.hef"
CLASSES = {0: 'person', 5: 'bus', 7: 'truck', 2: 'car', 3: 'motorcycle', 67: 'cell phone'}

def decode_predictions(output_buffers, threshold=0.5):
    boxes, scores, class_ids = [], [], []
    resolutions, strides = [80, 40, 20], [8, 16, 32]
    
    for res, stride in zip(resolutions, strides):
        box_node = [n for n, b in output_buffers.items() if b.shape == (res, res, 4)][0]
        score_node = [n for n, b in output_buffers.items() if b.shape == (res, res, 80)][0]
        
        box_data = output_buffers[box_node]
        score_data = np.clip(output_buffers[score_node], -50, 50) # Prevents overflow warnings
        prob = 1 / (1 + np.exp(-score_data))
        
        max_prob = np.max(prob, axis=-1)
        y_idx, x_idx = np.where(max_prob > threshold)
        
        for y, x in zip(y_idx, x_idx):
            cls_id = np.argmax(prob[y, x])
            score = prob[y, x, cls_id]
            
            l, t, r, b = box_data[y, x]
            cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
            
            x_min = int(cx - (l * stride))
            y_min = int(cy - (t * stride))
            width = int((r + l) * stride)
            height = int((b + t) * stride)
            
            boxes.append([x_min, y_min, width, height])
            scores.append(float(score))
            class_ids.append(int(cls_id))
            
    return boxes, scores, class_ids

# --- HEADLESS MAIN EXECUTION ---
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

print("Initializing NPU...")
with VDevice(params) as vdevice:
    infer_model = vdevice.create_infer_model(HEF_PATH)
    infer_model.set_batch_size(1)
    
    for output_name in infer_model.output_names:
        infer_model.output(output_name).set_format_type(FormatType.FLOAT32)
        
    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()
        
        output_buffers = {}
        for output_name in infer_model.output_names:
            out_shape = infer_model.output(output_name).shape
            output_buffer = np.empty(out_shape, dtype=np.float32)
            bindings.output(output_name).set_buffer(output_buffer)
            output_buffers[output_name] = output_buffer
            
        print("Warming up Pi Camera...")
        cap = cv2.VideoCapture(0)
        
        # Hardware ISP Resizing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        if not cap.isOpened():
            print("Error: Could not open the Pi Camera.")
            exit()
            
        # Setup Video Writer for Headless Saving
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('headless_output.avi', fourcc, 30.0, (640, 640))
            
        print("Recording live inference to 'headless_output.avi'...")
        print("Press Ctrl+C to stop recording and save the file.")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                loop_start = time.time()
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bindings.input(infer_model.input_names[0]).set_buffer(img_rgb)
                
                configured_infer_model.run([bindings], 1000)
                
                boxes, scores, class_ids = decode_predictions(output_buffers, threshold=0.45)
                
                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.5)
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        label = CLASSES.get(class_ids[i], f"ID {class_ids[i]}")
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {scores[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                loop_time = time.time() - loop_start
                fps = 1.0 / loop_time if loop_time > 0 else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f} | CPU: {psutil.cpu_percent()}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                out.write(frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames... (Current FPS: {fps:.1f})")

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")

        finally:
            cap.release()
            out.release()
            print("[SUCCESS] Camera closed. Video saved to 'headless_output.avi'.")