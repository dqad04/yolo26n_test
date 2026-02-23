import numpy as np
import cv2
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

HEF_PATH = "yolo26n.hef"
IMAGE_PATH = "test_image.jpg"
OUTPUT_PATH = "output_image.jpg"

# Standard COCO classes (just a few for the bus image)
CLASSES = {0: 'person', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}

def decode_predictions(output_buffers, threshold=0.5):
    boxes, scores, class_ids = [], [], []
    
    # We map the resolutions to their respective strides (640 / res)
    resolutions = [80, 40, 20]
    strides = [8, 16, 32]
    
    for res, stride in zip(resolutions, strides):
        # Find the specific nodes for this resolution scale
        box_node = [n for n, b in output_buffers.items() if b.shape == (res, res, 4)][0]
        score_node = [n for n, b in output_buffers.items() if b.shape == (res, res, 80)][0]
        
        box_data = output_buffers[box_node]
        score_data = output_buffers[score_node]
        
        # 1. Convert raw logits to 0-100% probabilities
        prob = 1 / (1 + np.exp(-score_data))
        
        # 2. Find any grid cells that have a score higher than our threshold
        max_prob = np.max(prob, axis=-1)
        y_idx, x_idx = np.where(max_prob > threshold)
        
        for y, x in zip(y_idx, x_idx):
            cls_id = np.argmax(prob[y, x])
            score = prob[y, x, cls_id]
            
            # 3. Decode the Bounding Box (Left, Top, Right, Bottom)
            l, t, r, b = box_data[y, x]
            
            # Calculate the exact center of this grid cell in the 640x640 image
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            
            # Calculate pixel boundaries
            x_min = int(cx - (l * stride))
            y_min = int(cy - (t * stride))
            width = int((r + l) * stride)
            height = int((b + t) * stride)
            
            boxes.append([x_min, y_min, width, height])
            scores.append(float(score))
            class_ids.append(int(cls_id))
            
    return boxes, scores, class_ids

# --- MAIN EXECUTION ---
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

with VDevice(params) as vdevice:
    infer_model = vdevice.create_infer_model(HEF_PATH)
    infer_model.set_batch_size(1)
    
    for output_name in infer_model.output_names:
        infer_model.output(output_name).set_format_type(FormatType.FLOAT32)
        
    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()
        
        # Load Image
        img = cv2.imread(IMAGE_PATH)
        original_h, original_w = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        bindings.input(infer_model.input_names[0]).set_buffer(img_rgb)
        
        # Setup Output Buffers
        output_buffers = {}
        for output_name in infer_model.output_names:
            out_shape = infer_model.output(output_name).shape
            output_buffer = np.empty(out_shape, dtype=np.float32)
            bindings.output(output_name).set_buffer(output_buffer)
            output_buffers[output_name] = output_buffer
            
        # Run NPU
        print("Running NPU Inference...")
        configured_infer_model.run([bindings], 1000)
        
        # Decode the results
        print("Decoding bounding boxes...")
        boxes, scores, class_ids = decode_predictions(output_buffers, threshold=0.4)
        
        if not boxes:
            print("No objects detected above the threshold.")
        else:
            # 4. Clean up overlapping boxes using Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.5)
            
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = CLASSES.get(class_ids[i], f"Class {class_ids[i]}")
                conf = scores[i]
                
                # Scale boxes back to the original image size
                x = int(x * (original_w / 640))
                y = int(y * (original_h / 640))
                w = int(w * (original_w / 640))
                h = int(h * (original_h / 640))
                
                # Draw on the original image!
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label}: {conf*100:.1f}%"
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Detected: {text}")
                
        # Save the result
        cv2.imwrite(OUTPUT_PATH, img)
        print(f"\n[SUCCESS] Image saved to {OUTPUT_PATH}")