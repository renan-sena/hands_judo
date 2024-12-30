# Judo Grip Tracking

This project uses **MediaPipe**, **OpenCV**, and the **Kalman Filter** to track the grips of two athletes in a judo video. It determines which athlete (wearing blue or white judogi) maintains the grip on their opponent for the longest duration.

---

## 📋 **Features**

1. **Hand Tracking with MediaPipe**  
   Detects and tracks athletes' hands in real-time using the MediaPipe library.

2. **Kalman Filter for Smoothing**  
   Applies the Kalman Filter to smooth detected hand positions and predict movements.

3. **Judogi Color Detection**  
   Identifies athletes based on the judogi color (blue or white) to assign the grip to the correct athlete.

4. **Grip Analysis**  
   Determines the winner of the grip if an athlete holds it for more than 7 consecutive seconds.

5. **Real-Time Visualization**  
   Displays grip statuses ("Active" or "Inactive") directly on the video and prints the winner in the terminal.

---

## 📜 **Dependencies**

- [OpenCV](https://opencv.org/)  
- [MediaPipe](https://mediapipe.dev/)  
- [NumPy](https://numpy.org/)  
- [Colorama](https://pypi.org/project/colorama/)

---

## 🛠️ **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/RenanLealSena/hands-judo.git
   cd judo-pegada-tracking
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python
   pip install mediapipe
   pip install numpy
   pip install colorama
   ```

3. Ensure a video is available at `video_path.mp4`, or update the `video_path` variable in the code.

---

## ▶️ **How to Run**

1. Ensure all dependencies are installed.
2. Run the script:
   ```bash
   python app.py
   ```
3. To exit the video display, press `q`.

---

## 📂 **Code Structure**

- **Hand Detection (MediaPipe)**  
  Tracks athletes' hands and maps their positions to identify grips.

- **Kalman Filter**  
  Used to smooth motion detection and predict future hand positions.

- **Grip Verification**  
  Analyzes the continuous grip duration and determines the winner based on a minimum duration of 7 seconds.

- **Color Detection (HSV)**  
  Identifies athletes based on the judogi color:  
  - Blue: `[100, 150, 50]` to `[140, 255, 255]`  
  - White: `[0, 0, 200]` to `[180, 30, 255]`  

---

## 🔧 **Configuration and Customization**

- **Adjust Minimum Grip Time**  
  Change the value `7` in the `verify_grip()` function to modify the time required to determine the winner.

- **Disappear Tolerance Configuration**  
  The `disappear_tolerance` variable defines the maximum time without detection before considering the grip lost.

- **Video Path**  
  Update the `video_path` variable with the path to your custom video.

---

# Running in Jupyter Notebook


## 📜 **Dependencies**

- [OpenCV](https://opencv.org/)  
- [MediaPipe](https://mediapipe.dev/)  
- [NumPy](https://numpy.org/)  
- [Ipython](https://ipython.org/)
- [Pillow](https://pypi.org/project/pillow/)
- [Colorama](https://pypi.org/project/colorama/)

---

## 🛠️ **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/RenanLealSena/hands-judo.git
   cd judo-pegada-tracking
   ```

2. Install the required packages:

   ```bash
   !pip install opencv-python 
   !pip install mediapipe 
   !pip install numpy 
   !pip install ipython 
   !pip install pillow 
   !pip install colorama
   ```

3. Ensure a video is available at `video_path.mp4`, or update the `video_path` variable in the code.

---

# 📈 **Future Improvements**

- Add support for more judogi colors.  
- Implement graphical visualizations of grip duration.  
- Optimize the Kalman Filter for multiple hands simultaneously.

---

## 🏆 **License**

This project is free to use and modify under the terms of the [MIT License](LICENSE).

---