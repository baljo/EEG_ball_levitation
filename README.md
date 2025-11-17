# Mind Blowing Brain Hack - Levitate a Ball with your thoughts!

![](/images/Ping-Pong-ball.jpg)

## Overview

While this project is neither *brain surgery* nor *rocket science*, it still involves using only your mind to get a ball flying into the air!

Jokes aside, the project demonstrates a physical biofeedback system where a user can levitate a ping pong ball by changing their mental state. A Muse EEG headset captures brain activity, which is processed through Edge Impulse spectral features and a custom three class model: calm, sleep and non calm. The output class determines the blower speed, and the ball rises or falls  in real time. 

The goal is to turn mental activity into physical movement in a direct and intuitive way. This creates a new form of feedback that is engaging, easy to understand and suitable for training focus or calmness.

This idea as such is not novel - there are earlier projects using entry level EEG-devices that in some cases even are marketed as toys. Personally I have also earlier published projects using the Muse EEG headband to control devices. What makes this project different though, is that it focuses on the biofeedback side with the aim for the user to target a balanced mental state. Furthermore, by using Edge Impulse, the model can easily be implemented and updated, even amended with more mental states.

The project combines low-tech components like a blower and ping pong ball with high-tech components like an EEG-device and the Edge Impulse platform.

![](/images/Final_video.gif)
---

## Why This Matters

### Mental awareness and mental fatigue

Mental fatigue is a state that develops after sustained cognitive effort and can reduce performance, slow reaction times and alter EEG activity. Research shows that this state is often accompanied by increased frontal theta power and reductions in alpha activity, which reflect a shift toward lower alertness and reduced executive control (Wascher et al. 2014). Mental fatigue also affects self-regulation and increases vulnerability to performance drops, as discussed by Boksem and Tops (2008) and Pageaux et al. (2015).

Neurofeedback approaches aim to give users real-time information about their internal brain state so they can actively correct course. Studies on slow cortical potential neurofeedback (Drechsler et al. 2007; Strehl 2017) show that people can learn to modulate neural activity with simple, continuous feedback.

This project applies the same principle in a more accessible way: the user’s mental state controls the position of a levitating ball. Because the feedback is physical and immediate, users can more easily notice when they drift into low alertness (sleep class) or high tension (non calm) and work their way back to a balanced state (calm). This makes the system a practical tool for developing mental awareness and managing everyday cognitive fatigue, even though it is not a clinical device.


---

## How it works conceptually

#### 1. EEG input
A Muse headset streams four EEG channels via BrainFlow. The program collects short overlapping windows for processing.

#### 2. Feature generation
Each window is converted into spectral features using Edge Impulse spectral analysis. The project also supports raw feature testing for validation.


EEG activity is often described in frequency bands: theta (4–8 Hz), alpha (8–12 Hz), beta (13–30 Hz), and gamma (>30 Hz). The Edge Impulse model does not label these bands explicitly, but its FFT-based spectral features capture them automatically.

* Sleep / eyes-closed → strong alpha increase (and some theta), which the model learns as the “sleep” class.
* Calm → stable moderate alpha with lower beta activity.
* Non-calm (blinks, facial movement, cognitive load) → spikes in beta/gamma or broadband energy increases.

These characteristic spectral patterns are what allow the classifier to separate the three mental states.


#### 3. ML prediction
An Edge Impulse model classifies the window into one of three mental states: calm, sleep or non calm. The model is exported as a TFLite or Keras .h5 file and loaded by the Python script.

#### 4. Real time control
Inference runs every 500 ms. A class history buffer smooths predictions so that brief spikes do not cause sudden blower changes.  
The predicted class maps to a PWM value between 0 and 255, sent over Wi-Fi to a Particle Photon 2 that drives a 12 V blower through a Grove MOSFET. The airflow lifts or lowers a ping pong ball in real time.

---


# HARDWARE NEEDED

* [Muse 2 EEG headband](https://eu.choosemuse.com/products/muse-2)
* PC running Python and BrainFlow
* [Particle Photon 2](https://store.particle.io/products/photon-2?srsltid=AfmBOoqq1V3DIss33WjENd1w0_bLUDX-0jrmhQs3YINJkUIqyJ5eP8fq) (or any other MCU supporting Wi-Fi and/or serial connection)
* [Grove Shield for Particle Mesh](https://www.seeedstudio.com/Grove-Shield-for-Particle-Mesh-p-4080.html?srsltid=AfmBOootJoz0kHWhe1_mkOXSAPDkXbr2qgLGuEO5VhwbfaTIpUZ2rS7F) - this is if you want to use the Grove ecosystem and cables
* [Grove MOSFET for Arduino](https://www.seeedstudio.com/Grove-MOSFET.html) - can be substituted with another suitable MOSFET
* [12 V blower fan](https://www.sparkfun.com/blower-squirrel-cage-12v.html)
* Ping Pong ball
* 12 V power source (wall adapter or battery)

Optional:
* Powerbank (only if you want the Photon 2 to be stand-alone and not connected to your computer)
* 3D-printer, if you want to print a protective case for the Grove Shield, or a stand for the blower

In this project a PC is used as an edge device, but it can easily be replaced with e.g. a Raspberry Pi or any other BLE-equipped device running Python and supported by Brainflow. With a Raspberry Pi you don't even need the Photon 2 as long as you can connect a MOSFET to it. And, if Python is not your cup of tea, Brainflow supports almost any modern language like Julia, Rust, C#, Swift, TypeScript, etc. Even some game engines are supported! 

# BUILD INSTRUCTIONS

In this section you'll learn how to collect data, train and deploy a ML-model, connect the devices, and finally, let the ping pong ball levitate.

## Install Python programs

You'll basically only need two Python programs, one for capturing data to be imported into Edge Impulse, another for inferencing and sending signals to the blower. While these programs of course could be combined, and controlled via parameters, or a menu, it's often easier to keep completely different modules separated.

Clone the repository, decide on if you want to install it directly on your device, or in a virtual environment. Then open a command prompt and run `pip install -r requirements.txt`. This will install all needed libraries to the selected environment.

## Set up your Muse EEG device

Start your Muse headset and wear it properly, it should **not** be connected to your phone or other device in this project.

## Capture EEG-data for Edge Impulse

In this chapter you'll learn how to collect data for Edge Impulse.

### Usage

Here you'll use the [Capture_EEG_data.py](/src/Capture_EEG_data.py) program. There are only three settings you need to know:

```
# ====== CONFIG ======
LABEL = "non_calm.high_load"    # change between runs: "calm", "non_calm", "sleep", etc.
DURATION_SEC = 20               # how long to record this label
OUTPUT_DIR = "data"             # folder for CSV files
```
- Set `LABEL` to the label you want to use in Edge Impulse. When importing data where the file name will represent the label, only the part before the first dot (.) will be used, so in above example the final label = `non_calm`. In the example below, I wanted to add a note indicating this file included data when I had a high cognitive load. This in case I want to use this explicit sample later on.
  - To avoid the need to rename files later, always add a dot (.) after the label, even if you don't add something else!
- Set `DURATION_SEC` to how long you want the sample to be. 20-30 seconds is good to start with. Only when having your eyes closed and relaxing, it's easy to have 1-3 minutes or so.
- `OUTPUT_DIR` is used if you want your data to reside in a subfolder (recommended).

#### Start collecting data

- Set the first label you want to record, e.g. `"sleep"` 
- Start the program from a command prompt `python Capture_EEG_data.py`
- Once you see that your Muse device has connected, data capture will very soon start
- "Perform" the action, feel free to experiment with different actions as long as they are distinctive. Here the ones I used:
  - *sleep* = keep your eyes closed and relax without moving (avoid falling asleep though :-D)
  - *calm* = eyes open, relax without moving, avoid blinking if possible
  - *non_calm* = eyes open, blink and moving ok. You can also experiment with high cognitive load in this state, e.g. count down from 100 to 0 with 7 (93, 86, 79...) 
- Keep same label, or change it when ready to move to next one, rinse and repeat.
  - Try to collect roughly same amount of data for each label.

## Build a model with Edge Impulse Studio

In this section you'll learn how to import the EEG-data, build, train, test, and deploy a ML-model. 
A prerequisite for the following steps is that you have created an EI account (free tier is more than enough for this project), and logged into it.

### Import data

This step is about creating a import model via the CSV Wizard, and importing the sample files. If you've not used the CSV Wizard before, why not take a look at the [documentation](https://docs.edgeimpulse.com/studio/projects/data-acquisition/csv-wizard#csv-wizard).

- Select `Data acquisition` from the menu
- Click on `CSV Wizard`, upload one of your recorded CSV-files, and use following settings:
  - Timeseries in rows
  - Timestamp in seconds
  - Length 2000 ms (= 2 seconds)
  - Frequency 256 Hz
- After this you can upload all files using the default options. Edge Impulse will use file names - everything until the first dot (.) - as label for each sample in the file

![](/images/EI_003.png)

Once the files are uploaded you'll see the balance between the labels as well as the split between training and test data. If there's a huge discrepancy between the labels, you should record more data for the misrepresented labels to get a good balance. 

![](/images/EI_004.png)

### Create an impulse

In this step you'll set up data processing and learning blocks.

- Select `Create impulse` from the menu
- Set up the time series data like in the picture
  - Window size = 2000 ms
  - Window increase = 500 ms
  - Frequency 256 Hz
  - Checkmark Zero-pad data (= shorter samples than the window size will be filled with zeroes at the end)
- Select `Spectral Analysis` as processing block and ensure all four axes are selected
- Select 'Classification` as learning block and ensure spectral features is selected
- You should now see all three labels as output features
- Click `Save Impulse`

![](/images/EI_006.png)

### Configure spectral features

Here you'll configure FFT-settings ([Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)), and generate features.

- Select `Spectral features` from the menu
- Fill in the settings like in the screenshot below
  - feel free to experiment with these later, it's however important to update the EEG-inferencing Python program accordingly
- Click `Save parameters`
- On next page, click `Generate features`
  - Depending on the data amount and server load, this step typically takes a couple of minutes or so

![](/images/EI_009.png)


### Train the model

Here you'll start to see some first results of your work!

- Select `Classifier` from the menu
- Try first with same settings as in the screenshot, this is though an iterative process, so feel free to change settings if you find the model is not performing well.
- Click `Save & train`
  - You'll see the progress in the upper right window
  - This is the step that typically consumes most time, from a few minutes up to several hours. With a few minutes of data, and with these settings, the training should be ready in under 10 minutes.
  - Note that everything happens on the server side, so if the training takes much longer than expected, you don't need to keep the browser session open if you need to shut down your computer.
- Once the training is complete, you'll see the accuracy. With EEG-data that can be messy and have lots of artefacts from blinking, muscle twitching etc., everything above 90% or so can be considered a success.
  - While 100% accuracy might at first glance be a goal to desire, there's a risk of "overfitting", meaning your model works perfect with the exact data samples you've used, but not with data it hasn't seen before. This often happens with smaller data amounts and too large neural networks. 

![](/images/EI_012.png)

**Accuracy low?** 

If the accuracy is very low, the root cause is most often on the data side, or on how you've built up your ML-model.
- The visual explorer in the bottom right corner is in these cases probably showing that the data clusters are mostly overlapping each other.
- First verify that you have roughly same amount of data for each label
- Try to verify that the uploaded data is really representing respective label.
    - If this would a computer vision project, you'd verify that photos using the label 'cat' indeed are of cats and not dogs or zebras.
    - With EEG-data it's however nearly impossible to look at the data to be able to confirm it. Only data samples with e.g. eye blinks, jaw clenches or similar are shown as spikes.
- Building ML-models is an iterative process, even more so for this type of EEG-project. Thus, I recommend that you initially collect a smaller amount of data, train and test the model. 
- If you already in the beginning notice poor accuracy, you might consider deleting or inactivating the uploaded samples and record new data. Once you have a good base, you can consider collecting more and more diverse data.


### Test the model

Here you'll verify how well your model performs on data it hasn't seen before, i.e. the approximately 20% of data put aside in the data upload.

- Select `Model testing` from the menu
- Click `Classify all`
- Similarly as after completion of the training step you'll get an accuracy-%. This is typically somewhat lower than in the training phase. If it's much lower, you might have overfitted your model, follow the steps under **Accuracy low?** above.

![](/images/EI_015.png)

### Deploy the model

Deploying the model for this project is very simple as you only need to download one file.

- Select `Dashboard` from the menu
- Scroll down until you find `TensorFlow lite (float32)`, and click on the download icon
- This downloads an optimized model
  - move it to the `src`-folder (where you have the Python-files)
  - I recommend you rename it to a shorter name
  - this is the file name you'll later need to update in your Python-program, as per [these](https://github.com/baljo/EEG_ball_levitation?tab=readme-ov-file#selected-parameters) instructions

![](/images/EI_018.png)

## Wiring

Wiring is extremely simple, no soldering needed if you use above hardware. Do note that you can choose to have your Photon 2 connected to your computer and communicate through a serial port, or you can have it as a standalone device and communicate through Wi-Fi.

### Photon 2
* Plug your Photon 2 into the Grove Shield. Power it via a micro-USB cable connected to your computer. 
* Follow Particle's [instructions](https://setup.particle.io/) to set up your MCU and connect to your Wi-Fi network. 

![](/images/Photon2.jpg)

Optional:
* From the `images`-folder you'll find a [3D-printable STL-file](/images/Grove%20Shield%20Feather%20Case.stl) as protective bottom for the Grove shield. 
* Print this in a flexible material like TPU so it's easy to bend it around the shield's PCB.
* The bottom has engraved text for each Grove port, this is to make them easier to find without the need of a magnifying glass.
* Orient the bottom, so the USB-port of your Photon 2 aligns with the engraved `USB`.


![](/images/Shield_bottom.png)

### Grove MOSFET and 12V power source

* Connect the MOSFET with a Grove cable to the A2 Grove port on the shield
* Connect a wire from + on your 12V source to + on Grove in
* Connect a wire from - on your 12V source to - on  Grove in
* Connect the blower's red (+) wire to + on Grove out
* Connect the blower's black (-) wire to GND on Grove out

![](/images/MOSFET.jpg)

### Muse 2
* No wiring needed, (or even possible!), but set up the device according to the instructions. Remember that it can only stream to one device at a time, so don't have it connected to your phone while streaming EEG-data for this project.


## Install the Photon 2 program controlling the blower
Upload the [program](/src/levitate-ball-v0-1.ino) to your Photon 2. As the program doesn't need external libraries, you can use Particle's Web IDE if you want.

### How it works

The program is very straightforward:
- It sets up a web server listening at port 9000
- It's waiting for a number between 0-255 (in ASCII-form) through the serial port, or via Wi-Fi.
- When a nubmer is received, it sets the blower speed accordingly. 
- If a number has not been received the latest 30 seconds, it stops the blower. 
  - This is to cover e.g. for situations when the sending program has been stopped, and you want the blower to stop without having to unpower the blower.
- If you connect the Photon 2 directly to your computer, you can, using a serial monitor, test the blower manually by transmitting a number 0-255.
- If you are going to use Wi-Fi, you need to know its IP-address to be able to connect to it. 
  - When you have connected the Photon 2 to your computer, the program is printing its IP-adress to the terminal, so store it somewhere for later use.

There are very few settings in the program: `MOSFET_PIN`, `PWM_MIN` and `PWM_MAX`, `SERVER_PORT`, and `COMMAND_TIMEOUT_MS`. Using the hardware in this project, you might though not need to change them.

```
const int MOSFET_PIN = A2;                 // Grove MOSFET SIG → A2
const int PWM_MIN    = 0;
const int PWM_MAX    = 255;

const int SERVER_PORT = 9000;             // TCP port for WiFi control
const unsigned long COMMAND_TIMEOUT_MS = 30000UL;  // 30 seconds (adjust if you want)
```


## Start the main program and levitate!

### Usage

Before you start the program the first time, you should take a look at the [Selected parameters](https://github.com/baljo/EEG_ball_levitation?tab=readme-ov-file#selected-parameters) below to understand which parameters you might need to change. These might be the file name of the ML-model you downloaded from the EI dashboard, window size, name of labels, number of FFT windows, etc. All these parameters are depending on your project setup in Edge Impulse Studio. 

You change the parameter values in the main Python program `EEG_ball_levitation_v0.4.3.py`, [direct link](/src/EEG_ball_levitation_v0.4.3.py).

### Start levitating

Once you are ready:
- Start your Photon 2 and provide the blower with power.
  - The really first time you might want to connect your computer to the MCU, once you see that it works you can use Wi-Fi if you want.
- Feel free to put a ping pong ball on your blower outlet!
- Start the program from a command prompt with `python EEG_ball_levitation_v0.4.3.py --wifi-host <Photon 2 IP-adress> --wifi-port 9000`. 

➨➤ If everything works alright, the blower should start reacting to your mental state!


### How it works

While the main program at first sight might look a bit involved, the functionality itself is actually quite straightforward. 

The program works like this:

- It connects to the Photon 2, either through serial, or through Wi-Fi.
- It opens the Keras .h5-file if it exists, otherwise the Tensorflow Lite file. Both are exported from Edge Impulse.
- Using the Brainflow library it connects to Muse EEG, and starts receiving signals.
- Using the [Spectral Analysis Python-library](https://github.com/edgeimpulse/processing-blocks/tree/master/spectral_analysis) from Edge Impulse, it processes features exactly as in EI Studio.
- It runs inference against these processed features.
- Finally, it sends a number 0-255 to the Photon 2. 
  - By default it averages the latest few inference results to provide a smoother user experience. Otherwise it might jump too frequently between the three states.
- It prints continuosly inference results for testing and possible troubleshooting needs.

#### Selected parameters

This section explains a few selected parameters that you might need to change according to your setup.


**ML-model:** Replace these with the file/files you export from Edge Impulse.
```
# ----------------- Model selection and setup -----------------
MODEL_TFLITE = "EEG_float32_FFT8_1.lite" # "EEG_float32.lite"
MODEL_H5 = "EEG_model_64.h5"
```

**Impulse settings:** These are from `Create impulse`in EI. The first two ones have to be same as in EI, but feel free to experiment with the stride if you want to. See [this documentation](https://docs.edgeimpulse.com/studio/projects/impulse-design#time-series-audio,-vibration,-movements), especially the sketch, how the stride (= window increase) works.
```
FS = 256.0  # Hz
WINDOW_SECONDS = 2.0  # window size in seconds
STRIDE_SECONDS = 0.250  # stride between decisions
```

**Smoothing and threshold settings:** These are used to average the inference results. The threshold 0.7 means that the ML-model needs to be at least 70% confident of a prediction for that prediction to be triggered.
```
STABILITY_WINDOWS = 10  # (tests) number of last decisions to require stable target
SMOOTH_WINDOWS = 5  # (tests) number of last probabilities to average
USE_MEDIAN_SMOOTH = True  # (tests) True=median smoothing, False=mean smoothing
TARGET_THRESHOLD = 0.7

# For live blower mapping we use class history instead:
CLASS_HISTORY_WINDOWS = 8  # number of last predicted classes to majority-vote
```

**Spectral features settings:** These needs to be identical as in the spectral features menu in EI. So, if you for example in EI Studio find out that a FFT length of 16 works better, you need to change `FFT_LENGTH`to 16 here. These parameters, together with the window size, ensure the input layer to your neural network will be same as in EI. It they aren't, you'll get a message like `...got 54, expected 36...`. 


```
AXES = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]

SCALE_AXES = 1.0
INPUT_DECIMATION_RATIO = 1
FILTER_TYPE = "none"
FILTER_CUTOFF = 0
FILTER_ORDER = 0
ANALYSIS_TYPE = "FFT"
FFT_LENGTH = 8
SPECTRAL_PEAKS_COUNT = 0
SPECTRAL_PEAKS_THRESHOLD = 0
SPECTRAL_POWER_EDGES = "0"
DO_LOG_IN_BLOCK = True  # let EI block take log of spectrum (matches training)
DO_FFT_OVERLAP = True
WAVELET_LEVEL = 1
WAVELET = ""
EXTRA_LOW_FREQ = False
```

**Labels:** These have to be same labels as you've used in EI.
```
LABELS = ["calm", "non_calm", "sleep"]

# Index of the "target" class for the test-modes (threshold logic)
TARGET_CLASS_INDEX = 1  # non_calm

# Class indices for blower mapping (must match LABELS)
CLASS_CALM = 0
CLASS_NON_CALM = 1
CLASS_SLEEP = 2
```

**Serial and Wi-Fi port settings:**

These are defalt settings, but can be overridden in the command prompt.
```
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200
DEFAULT_WIFI_PORT = 9000
```

**CLI-arguments available:**

These are optional, and to be used if you want to override the default settings.
* --serial-port COM3
* --serial-baud 115200
* --wifi-host aaa.bbb.ccc.ddd = IP-address of your Photon 2 if you want to use Wi-Fi
* --wifi-port xxxx = default is 9000, but can be overridden
* --no-output = disabling blower output entirely, if you e.g. want to just watch the results on your screen

For troubleshooting purposes only:
* --test-features "processed features" = copy **processed features** from EI **after** the spectral analysis block
* --test-raw "raw flattened data" = copy **raw flattened data** from EI **before** the spectral analysis block


## Other programs
There are a few other Python programs, versions, and files in the [src folder](/src/). Apart from the data capture program, these are not needed for normal operation, I have used them when building up the main program module by module, or for troubleshooting purposes. Touch them at your own risk!

# Further suggestions

While this project works as intended, there is of course room for adjustments or improvements.

## Physical aspects

### Ball not levitating very high?

**Note:** The video is shot in a lightbox, and as the ceiling is quite low, it makes the ball oscillating more, and a few centimeters lower than in free air!

Having said that, With 12V the blower is still a bit too weak to blow the ball very high. I have not tried to give it more oomph, but a reviewer mentioned they had run it with 20V. Not recommending it, but let me know if you've tried 😉

Another option is to find a blower with more power, or a lighter ball.

### Ball oscillating?

As you see in the top video, the ball is oscillating. It does this even in free air, though a bit less. To mitigate this, I used a short and light piece of yarn to keep the ball aligned better vertically.
I had also considered 3D-printing some type of funnel or similar to possibly increase the ball altitude of the ball, but this would need a lot of designing, printing, and testing.

![](/images/Final_video_yarn.gif)

## ML-model aspects

As with all ML-models, they can always be improved. Here I'll share a few experiences and thoughts.

### Can you use my brain signals?

Brains are different, and my EEG-signals **might** be different from yours, meaning you probably need to collect more data to get a good accuracy. As mentioned earlier, I recommend you do this iteratively, meaning you collect a bit more data, upload it and then train and test the model before continuing.

### EEG-signal data quality issues?

While testing the model, I had intermittent issues when the model simply did not perform as it should. It could e.g. always predict calm with 100% confidence, regardless of what I did, or the prediction seemed to be random.

I found two reasons for this behavior:
* As the winter has arrived, the air is quite dry. This means that the dry electrodes on the EEG-device don't make good contact with the skin during the first few minutes. It is actually recommended to wet them with water to speed up the initial connection.
* Brain signals are measured in milli-volts, and at least the Muse EEG-device is sensitive to 50 Hz interference from nearby power conduits or cables. So, unless you want to go out in the forest or desert, at least stay a meter or two from the nearest electricity cables.

## Use only frontal electrodes

There is some research (Krigolson et al, Sidelinger et al, Beiramwand et al, Zhang et al) indicating that frontal/prefrontal channels can capture meaningful cognitive signals such as alpha activity and workload-related change. This means in practice that you *might* get better performance by only using data from the channels eeg2 and eeg3 as these map to AF7 (left frontal) and AF8 (right frontal). Right now all four channels are used, possible providing data not of importance for measuring mental state.

To change this, you can simple unselect eeg1 and eeg4 in Edge Impulse and retrain. In addition you also need to change the Python program slightly to accommodate for fewer channels. Feel free to experiment! 

# License

This project is open source and can be reused or modified for research, education or personal development.

---


### References

- **Mental fatigue overview and costs**  
  Boksem, M. A. S., & Tops, M. (2008). Mental fatigue: Costs and benefits. *Brain Research Reviews, 59(1), 125–139.*  
  https://doi.org/10.1016/j.brainresrev.2008.07.001  

- **Mental fatigue, self-regulation, and performance**  
  Pageaux, B., Marcora, S. M., Rozand, V., & Lepers, R. (2015). Mental fatigue induced by prolonged self-regulation does not exacerbate central fatigue during subsequent whole-body endurance exercise. *Frontiers in Human Neuroscience, 9, 67.*  
  https://doi.org/10.3389/fnhum.2015.00067  

- **EEG markers of mental fatigue (frontal theta and alpha)**  
  Wascher, E., Rasch, B., Sänger, J., Hoffmann, S., Schneider, D., Rinkenauer, G., Gutberlet, I., & Getzmann, S. (2014). Frontal theta activity reflects distinct aspects of mental fatigue. *Biological Psychology, 96, 57–65.*  
  PubMed: https://pubmed.ncbi.nlm.nih.gov/24309160/  

- **Neurofeedback of slow cortical potentials (SCP) as a treatment and training method**  
  Drechsler, R., Straub, M., Doehnert, M., Heinrich, H., Steinhausen, H. C., & Brandeis, D. (2007). Controlled evaluation of a neurofeedback training of slow cortical potentials in children with attention-deficit/hyperactivity disorder (ADHD). *Behavioral and Brain Functions, 3, 35.*  
  Full text: https://behavioralandbrainfunctions.biomedcentral.com/articles/10.1186/1744-9081-3-35  

- **Review on SCP neurofeedback and mechanisms**  
  Strehl, U. (2017). Slow cortical potentials neurofeedback in attention deficit hyperactivity disorder. *Frontiers in Human Neuroscience, 11, 135.*  
  https://www.frontiersin.org/articles/10.3389/fnhum.2017.00135

- **Validation of Muse for ERP research**  
  Krigolson, O. E., Williams, C. C., Norton, A., Hassall, C. D., & Colino, F. L. (2017).  
  Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research.  
  *Frontiers in Neuroscience, 11, 109.*  
  https://doi.org/10.3389/fnins.2017.00109

- **Mobile EEG alpha-frequency reliability (Muse-derived IAF)**  
  Sidelinger, L., Zhang, M., Frohlich, F., & Daughters, S. B. (2023).  
  Day-to-day individual alpha frequency variability measured by a mobile EEG device relates to anxiety.  
  *European Journal of Neuroscience, 57(11), 1815–1833.*  
  https://doi.org/10.1111/ejn.16002

- **Mental workload classification using only two prefrontal EEG channels**  
  Beiramvand, M., Shahbakhti, M., Karttunen, N., Koivula, R., Turunen, J., & Lipping, T. (2024).  
  Assessment of mental workload using a transformer network and two prefrontal EEG channels: An unparameterized approach.  
  *IEEE Transactions on Instrumentation and Measurement, 73, 1–10.*  
  https://doi.org/10.1109/TIM.2024.3395312

- **Muse 2 performance in real-time cognitive workload detection**  
  Zhang, L., & Cui, H. (2022).  
  Reliability of MUSE 2 and Tobii Pro Nano at capturing mobile application users’ real-time cognitive workload changes.  
  *Frontiers in Neuroscience, 16, 1011475.*  
  https://doi.org/10.3389/fnins.2022.1011475
