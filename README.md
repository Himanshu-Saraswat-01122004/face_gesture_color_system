<div align="center">

# 🎨 Face Gesture Color System 👍👎

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.0+-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19.0+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-Educational_Use-yellow.svg)]()

*An intelligent system that recognizes faces and interprets hand gestures to provide personalized color recommendations*

</div>

---

## 🚀 Features

- 👤 **Facial Recognition** - Identifies users across sessions
- 👍 **Gesture Detection** - Recognizes thumbs up and thumbs down
- 🎨 **Color Recommendation** - Suggests colors based on user preferences
- 💾 **Preference Storage** - Saves your liked and disliked colors
- 📊 **Visual Feedback** - Provides real-time interaction

---

## 📋 Requirements

- 🐍 Python 3.6 or higher
- 📷 Webcam
- 📦 Required packages (included in `requirements.txt`):
  ```
  opencv-python>=4.5.0
  numpy>=1.19.0
  ```

---

## 🔧 Installation

### 1️⃣ Get the code
```bash
# Clone or download this repository
git clone https://github.com/Himanshu-Saraswat-01122004/face_gesture_color_system.git
cd face_gesture_color_system
```

### 2️⃣ Set up environment
```bash
# Create a virtual environment (recommended)
python -m venv env

# Activate it
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### Starting the Application
```bash
python face_gesture_system.py
```

### Interacting with the System

1. 🧍 Position yourself in front of the webcam so your face is visible
2. 🎨 The system will detect your face and show a color suggestion
3. 👍👎 Provide feedback using one of these methods:
   - Show a **thumbs up** gesture to like the current color
   - Show a **thumbs down** gesture to dislike and see the next color
   - Press the **'u'** key to like the current color
   - Press the **'d'** key to dislike the current color

---

## 👐 Gesture Detection Tips

For optimal gesture recognition:

| Tip | Description |
|-----|-------------|
| 📏 Position | Make gestures in the bottom half of the frame |
| 💡 Lighting | Ensure good lighting for clear hand visibility |
| 👋 Movement | Make clear, deliberate hand movements |
| ⏱️ Timing | Hold the gesture briefly for better detection |

---

## 💾 Saving Your Preferences

To create a permanent user profile:

1. Press the **'n'** key during the session
2. Enter your name when prompted
3. Your color preferences will be saved and associated with your face
4. The system will recognize you in future sessions

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **u** | 👍 Like the current color |
| **d** | 👎 Dislike the current color |
| **n** | 👤 Set your name to save preferences |
| **q** | 🚪 Quit the application |

---

## 🔍 Technical Details

- **Face Detection**: Uses Haar Cascades from OpenCV
- **Gesture Recognition**: Employs motion detection and contour analysis
- **Data Storage**: User preferences are stored in a JSON file
- **Logging**: System events are recorded in `gesture_system.log`

---

## ❓ Troubleshooting

If you experience issues:

- ✅ Ensure your webcam is properly connected
- ✅ Check that you have adequate lighting
- ✅ Verify no other applications are using your webcam
- ✅ Review `gesture_system.log` for detailed error information

---

## 📄 License

This project is available for educational and personal use.

---

<div align="center">

## 👨‍💻 Author

Created as part of a computer graphics course assignment.

Made with ❤️ by Yogi

</div>
