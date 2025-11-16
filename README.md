# Mind Blowing Brain Hack - Levitate a Ball with your thoughts!


## Overview

While this project is neither *brain surgery* nor *rocket science*, it still involves using only your mind to get a ball flying!

Jokes aside, the project demonstrates a physical biofeedback system where a user can levitate a ping pong ball by changing their mental state. A Muse EEG headset captures brain activity, which is processed through Edge Impulse spectral features and a custom three class model: calm, sleep and non calm. The output class determines the blower speed, and the ball rises or falls  in real time. 

The goal is to turn mental activity into physical movement in a direct and intuitive way. This creates a new form of feedback that is engaging, easy to understand and suitable for training focus or calmness.

This idea as such is not novel - there are earlier projects using entry level EEG-devices that in some cases are marketed as toys. Personally I have also earlier published projects using the Muse EEG headband to control devices. What makes this project different though, is that it focuses on the biofeedback side with the aim for the user to target a balanced mental state. Furthermore, by using Edge Impulse, the model can easily be implemented and updated, even amended with more mental states.

---

## Why This Matters

### 1. A practical biofeedback tool

Traditional neurofeedback relies on charts or sounds. This project replaces abstract feedback with a real physical object and a three level control scheme:

- eyes closed (0 %) pulls the ball down
- calm (50 %) keeps it near the center
- non calm (100 %) pushes it up

The user’s task is to keep the ball hovering close to the middle. This means the system does not only reward maximum relaxation or maximum activation. Instead it trains the ability to find and maintain a balanced mental state that is alert but not stressed, relaxed but not drowsy. That is a more realistic target for everyday focus and performance.

### 2. Supports stress reduction and focus training

The system can help users to practice relaxation, build sustained attention, and learn how internal states influence performance. This makes it relevant for wellness, mental health training, high performance environments and education.

### 3. Foundation for hands free control

The same architecture can scale to for example: assistive devices, hands free interfaces, robotics, game control, accessibility tools, and many others.
This project shows that simple EEG patterns combined with edge ML can drive real world devices.

---

## How it works

#### 1. EEG input
A Muse headset streams four EEG channels via BrainFlow. The program collects short overlapping windows for processing.

#### 2. Feature generation
Each window is converted into spectral features using Edge Impulse spectral analysis. The project also supports raw feature testing for validation.

#### 3. ML prediction
An Edge Impulse model classifies the window into one of three mental states: calm, sleep or non calm. The model is exported as a TFLite or Keras .h5 file and loaded by the Python script.

#### 4. Real time control
Inference runs every 200 ms. A class history buffer smooths predictions so that brief spikes do not cause sudden blower changes.  
The predicted class maps to a PWM value between 0 and 255, sent over Wi-Fi to a Particle Photon 2 that drives a 12 V blower through a Grove MOSFET. The airflow lifts or lowers a ping pong ball in real time.



---

# Hardware

* [Muse 2 EEG headband](https://eu.choosemuse.com/products/muse-2)
* PC running Python and BrainFlow
* [Particle Photon 2](https://store.particle.io/products/photon-2?srsltid=AfmBOoqq1V3DIss33WjENd1w0_bLUDX-0jrmhQs3YINJkUIqyJ5eP8fq) (or any other MCU supporting Wi-Fi and/or serial connection)
* [Grove Shield for Particle Mesh](https://www.seeedstudio.com/Grove-Shield-for-Particle-Mesh-p-4080.html?srsltid=AfmBOootJoz0kHWhe1_mkOXSAPDkXbr2qgLGuEO5VhwbfaTIpUZ2rS7F) - this is if you want to use the Grove ecosystem and cables
* [Grove MOSFET for Arduino](https://www.seeedstudio.com/Grove-MOSFET.html) - can be substituted with another suitable MOSFET
* [12 V blower fan](https://www.sparkfun.com/blower-squirrel-cage-12v.html)
* Ping Pong ball
* 12 V power source (wall adapter or battery)
* Powerbank (only if you want the Photon 2 to be stand-alone and not connected to your computer)

In this project a PC is used as an edge device, but it can easily be replaced with e.g. a Raspberry Pi or any other BLE-equipped device running Python and supported by Brainflow. With a Raspberry Pi you don't even need the Photon 2 as long as you can connect a MOSFET to it. And, if Python is not your cup of tea, Brainflow supports almost any modern language like Julia, Rust, C#, Swift, TypeScript, etc. Even some game engines are supported! 

## Wiring

Wiring is extremely simple, no soldering needed if you use above hardware. Do note that you can choose to have your Photon 2 connected to your computer and communicate through a serial port, or you can have it as a standalone device and communicate through Wi-Fi.

### Photon 2
* Plug your Photon 2 into the Grove Shield. Power it via a micro-USB cable connected to your computer. 
* Follow Particle's [instructions](https://setup.particle.io/) to set up your MCU and connect to your Wi-Fi network. 

### Grove MOSFET and 12V power source

* Connect the MOSFET with a Grove cable to the A2 Grove port on the shield
* Connect a wire from + on your 12V source to + on Grove in
* Connect a wire from - on your 12V source to - on  Grove in
* Connect the blower's red (+) wire to + on Grove out
* Connect the blower's black (-) wire to GND on Grove out

### Muse 2
* No wiring needed, (or even possible!), but set up the device according to the instructions. Remember that it can only stream to one device at a time, so don't have it connected to your phone while streaming EEG-data for this project.


---

# Software Components

## Photon 2 program controlling the blower

The [Photon 2 program](/src/levitate-ball-v0-1.ino)
### 1. EEG acquisition

* BrainFlow Python library
* Python serial
* Internal windowing and filtering

### 2. Edge Impulse

* Spectral processing block
* Model training and evaluation
* TFLite model export
* Public project with documentation

### 3. Control logic

* PWM mapping
* Safety timeouts
* WiFi command handling
* Smoothing of predictions
* Class history of last N predictions

---

# Repository Structure

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

# Setup Instructions

## 1. Clone the repository

```
git clone https://github.com/yourrepo/mind-controlled-ball
```

## 2. Install Python dependencies

Use the provided requirements file for BrainFlow and Edge Impulse processing blocks.

```
pip install -r requirements.txt
```

## 3. Connect the Muse headset

Confirm that BrainFlow detects the device.

```
python check_muse.py
```

## 4. Run the ML inference program

```
python EEG_ball_levitation.py
```

## 5. Flash the Photon 2

Upload the included firmware and verify serial output.

---

# Model Training

The Edge Impulse project is public and includes:

* dataset
* preprocessing steps
* model parameters
* validation metrics
* README describing the workflow

Data was collected in three controlled mental states. The spectral feature extractor used 256 samples per window and log scaled energy values. The model uses a small fully connected network suitable for microcontrollers.

---

# Video Demonstration

A short demonstration video shows:

1. The EEG headset collecting data
2. The Edge Impulse model predicting calm, sleep and non calm
3. The blower responding in real time
4. The ping pong ball rising and falling based on mental state

---

# Future Work

* Expand to regression for continuous control
* Introduce servo movement or multiple actuators
* Add Bluetooth audio feedback for training
* Explore cognitive load patterns more deeply
* Package as a classroom ready demo kit

---

# License

This project is open source and can be reused or modified for research, education or personal development.

---


