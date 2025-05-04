# Portable AI Sign-Language Translator 
**Platform:** Arduino Nicla Vision  
**Course:** CP330 – Edge AI  
**Instructor:** Prof. Pandarasamy Arjunan  
**Contributors:** Mohd Moin Khan, Poornima, Biswadeep Debnath, Falak Fatima  

---

## Project Overview

The **Portable AI Sign-Language Translator** is a compact, offline, AI-powered solution that uses the **Arduino Nicla Vision** board to detect and translate hand signs into **real-time textual output**. The goal is to provide a low-power, accessible assistive tool that functions without any internet connection.

---

## Dataset

### Data Collection  
- Images captured using the Nicla Vision’s onboard camera  
- Hand signs performed by team members in various lighting/background conditions  
- Each image annotated with bounding boxes for object detection
- [Dataset Link](https://drive.google.com/file/d/1TB8K_ECHom2vD-sPCrrnw5WASi-BnePt/view?usp=sharing)

### Preprocessing  
- Resized to **96x96** or **128x128** (depending on model)  
- Normalized and stratified split into train/test sets  
- Converted to Edge Impulse compatible object detection format  

---

## Model Architecture & Training

We initially developed a custom-trained MobileNetV2-based object detection model. However, due to deployment limitations on the Nicla Vision, we later transitioned to using **Edge Impulse Studio**, which provided better optimization for embedded deployment. The complete Edge Impulse flow (including model architecture and deployment pipeline) is included in a separate folder within this repository.


### Custom Model Details
#### Base Model  
- **MobileNetV2 (α = 0.35)** backbone (ImageNet pretrained, top removed)  
- **Lightweight head:**  
  - Dense(128, ReLU) → Dropout(0.3) → Softmax(15 classes)

#### Training Strategy  
1. **Stage 1:** Freeze backbone, train head (5 epochs)  
2. **Stage 2:** Unfreeze last 30 layers, fine-tune whole model at lower LR (5 epochs)  

### Deployment

- Initial custom-trained model (MobileNetV2 with INT8 quantization) resulted in a model size of **~700 KB**, which exceeded the usable flash/RAM limits of the Nicla Vision board.
- Therefore, we migrated to **Edge Impulse Studio**, which allowed us to optimize and quantize the model further using **FOMO (Faster Objects, More Objects)**.
- The final **INT8 quantized model** from Edge Impulse was **~57 KB**, making it lightweight and suitable for real-time inference on the Nicla Vision.


---

## Demonstration

The final system:
- Detects sign gestures using real-time vision
- Draws bounding boxes over the detected gesture
- Displays the corresponding **translated text**

*A video demonstration or GIF can be added here.*

---

## Requirements

- [Arduino Nicla Vision](https://store.arduino.cc/products/nicla-vision)  
- Edge Impulse CLI or Studio  
- Arduino IDE with Nicla Vision libraries  
- USB-C cable for flashing and serial communication  

---

## How to Use

Note: For the final working model, refer to the instructions provided in the `Edge_Impulse_Studio/` folder. This folder includes all steps for training, optimization, and deployment using Edge Impulse.

1. Clone this repository  
2. Follow setup and model training steps in the `Edge_Impulse_Studio/` folder  
3. Export the model as a `.zip` file from Edge Impulse Studio  
4. Extract the `.zip` file, then:
   - Copy `object_detection.py` from this repository to the Nicla Vision and rename it as `main.py`
   - Copy `trained_model.tflite` and `labels.txt` to the root of the Nicla Vision storage  
5. Flash the Nicla Vision with the latest firmware if required  
6. Power the device via USB or battery  
7. Show one of the supported hand signs to the camera  
8. Observe live text translation output and bounding box detection overlay


## Additional Deployment: Motion JPEG Streamer (Nicla Vision via Wi-Fi)

1. Copy the following files to the Nicla Vision (running OpenMV firmware):
   - `main.py`
   - `labels.txt`
   - `trained_model.tflite`

2. Connect Nicla Vision to a laptop and open `main.py` using the OpenMV IDE.

3. Run the script once while connected to a smartphone hotspot (Wi-Fi).

4. After successful connection (IP printed in terminal), disconnect Nicla Vision from the laptop.

5. Power the device using an external power source (e.g., power bank).

6. On your smartphone, open a browser and go to `http://<first IP address shown in OpenMV>:8080`.
   - Example: `http://192.168.43.123:8080`



