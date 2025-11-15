# Mind Blowing Brain Hack - Levitate a Ball with your thoughts!



## **Overview**

While this project is neither brain surgery or rocket science, it involves using only your mind to get a ball to fly!

Jokes aside, the project demonstrates a physical biofeedback system where a user can levitate a ping pong ball by changing their mental state. A Muse EEG headset captures brain activity, which is processed through Edge Impulse spectral features and a custom three class model: calm, sleep and non calm. The output class determines the blower speed, and the ball rises or falls  in real time. 

The goal is to turn mental activity into physical movement in a direct and intuitive way. This creates a new form of feedback that is engaging, easy to understand and suitable for training focus or calmness.

---

# **Why This Matters**

## **1. A practical biofeedback tool**

Traditional neurofeedback relies on charts or sounds. This project replaces abstract feedback with a real physical object and a three level control scheme:

- sleep (0 %) pulls the ball down
- calm (50 %) keeps it near the center
- non calm (100 %) pushes it up

The user’s task is to keep the ball hovering close to the middle. This means the system does not only reward maximum relaxation or maximum activation. Instead it trains the ability to find and maintain a balanced mental state that is alert but not stressed, relaxed but not drowsy. That is a more realistic target for everyday focus and performance.

## **2. Supports stress reduction and focus training**

The system can help users:

* practice relaxation
* build sustained attention
* learn how internal states influence performance

This makes it relevant for wellness, mental health training, high performance environments and education.

## **3. Accessible EEG and ML education**

The project provides a clear demonstration of:

* how raw EEG signals are collected
* how Edge Impulse converts them into usable features
* how a lightweight ML model predicts mental states
* how microcontrollers turn predictions into physical action

This is useful for students, workshops, STEM classes and public demonstrations.

## **4. Foundation for hands free control**

The same architecture can scale to:

* assistive devices
* hands free interfaces
* robotics
* game control
* accessibility tools

This project shows that simple EEG patterns combined with edge ML can drive real world devices.

---

# **How It Works**

## **1. Signal acquisition**

A Muse EEG headband streams four channels of brainwave data through BrainFlow. The project collects short windows of data at fixed intervals.

## **2. Feature extraction**

Windows are processed through Edge Impulse spectral features. This uses FFT based features that capture energy distribution across frequency bands. The system supports both spectral features and raw feature testing.

## **3. Classification**

A three class model inside Edge Impulse separates mental states into:

* calm
* sleep (eyes closed and relaxed)
* non calm (blinking, tension, cognitive load)

The model is exported as a TFLite file.

## **4. Real time application**

The Python program performs inference every 200 ms. Predictions are smoothed with a class history buffer so that the blower reacts to stable mental states rather than single frame noise.

Predicted class values are mapped to a PWM value between 0 and 255. This is sent over WiFi to a Particle Photon 2 that controls a 12 V blower through a Grove MOSFET.

The airflow lifts or lowers a ping pong ball within a transparent tube.

---

# **Hardware**

* Muse 2 EEG headband
* PC running Python and BrainFlow
* Particle Photon 2
* WiFi communication with a custom protocol
* 12 V blower fan
* Grove MOSFET driver
* Clear acrylic tube and lightweight ball

---

# **Software Components**

### **1. EEG acquisition**

* BrainFlow
* Python serial
* Internal windowing and filtering

### **2. Edge Impulse**

* Spectral processing block
* Model training and evaluation
* TFLite model export
* Public project with documentation

### **3. Control logic**

* PWM mapping
* Safety timeouts
* WiFi command handling
* Smoothing of predictions
* Class history of last N predictions

---

# **Repository Structure**

```
/SRC
    EEG_ball_levitation.py
    processing-blocks/
    models/
    utils/
    README.md

/hardware
    wiring-diagram.jpg
    blower-test.ino
    photon2-firmware.cpp

/data
    raw-eeg-sessions
    processed-features
    model-metrics
```

---

# **Setup Instructions**

## **1. Clone the repository**

```
git clone https://github.com/yourrepo/mind-controlled-ball
```

## **2. Install Python dependencies**

Use the provided requirements file for BrainFlow and Edge Impulse processing blocks.

```
pip install -r requirements.txt
```

## **3. Connect the Muse headset**

Confirm that BrainFlow detects the device.

```
python check_muse.py
```

## **4. Run the ML inference program**

```
python EEG_ball_levitation.py
```

## **5. Flash the Photon 2**

Upload the included firmware and verify serial output.

---

# **Model Training**

The Edge Impulse project is public and includes:

* dataset
* preprocessing steps
* model parameters
* validation metrics
* README describing the workflow

Data was collected in three controlled mental states. The spectral feature extractor used 256 samples per window and log scaled energy values. The model uses a small fully connected network suitable for microcontrollers.

---

# **Video Demonstration**

A short demonstration video shows:

1. The EEG headset collecting data
2. The Edge Impulse model predicting calm, sleep and non calm
3. The blower responding in real time
4. The ping pong ball rising and falling based on mental state

---

# **Future Work**

* Expand to regression for continuous control
* Introduce servo movement or multiple actuators
* Add Bluetooth audio feedback for training
* Explore cognitive load patterns more deeply
* Package as a classroom ready demo kit

---

# **License**

This project is open source and can be reused or modified for research, education or personal development.

---


