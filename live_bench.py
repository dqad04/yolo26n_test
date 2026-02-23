import time
from picamera2 import Picamera2
from picamera2.devices import Hailo

# Force the monitor to see us
import os
os.environ["HAILO_MONITOR"] = "1"

picam2 = Picamera2()
# Use a lighter configuration for the main stream to save CPU
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 640)})
picam2.configure(config)

# Use the 'with' block for the NPU
with Hailo("/home/ahmed/yolo26_test/yolo26n.hef") as hailo:
    picam2.start()
    
    print("STARTING ASYNC BENCHMARK...")
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # capture_array(raw=True) is faster as it avoids some Python overhead
            frame = picam2.capture_array()
            
            # This is where the 26 TOPS of the Hailo-10H actually kick in
            hailo.run(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Pipeline: {frame_count / elapsed:.2f} FPS")
                
    except KeyboardInterrupt:
        picam2.stop()