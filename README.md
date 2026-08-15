<div align="center">

# 👁️ DYNOVISION
### Autonomous Computer Vision Engine & Generative Ambient Acoustic Synthesizer

[![Live Demo](https://img.shields.io/badge/Live_Demo-dynovision.netlify.app-00DC82?style=for-the-badge&logo=netlify&logoColor=white)](https://dynovision.netlify.app/)
[![React 19](https://img.shields.io/badge/React-19.0.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8.2-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_Vision-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow-TFJS_4.22-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Web Audio API](https://img.shields.io/badge/Web_Audio-Tone.js_15-orange?style=for-the-badge&logo=audio&logoColor=white)](https://tonejs.github.io/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-Flash_Lite-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Tailwind CSS v4](https://img.shields.io/badge/Tailwind_CSS-v4.1-38B2AC?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

**[🌐 Experience Live Demo](https://dynovision.netlify.app/)** • **[📖 Architecture](#-system-architecture)** • **[🧠 AI Models](#-multi-model-computer-vision-deep-dive)** • **[🎵 Audio Engine](#-procedural-soundscape--acoustic-synthesis-engine)** • **[🔒 Privacy](#-privacy--zero-latency-edge-computing)**

---

</div>

## 🌐 About DynoVision

**DynoVision** is an edge-native, real-time AI perception engine and cybernetic soundscape synthesizer. It bridges the gap between **spatial computer vision**, **affective computing**, and **generative procedural acoustics**. 

By orchestrating parallel client-side neural networks directly inside the browser using WebAssembly and WebGL, DynoVision ingests a real-time webcam video feed to continuously quantify:
1. **Surrounding Objects & Entities** via object classification models.
2. **Fine-Grained Hand & Finger Kinematics** via 21 3D landmark points per hand.
3. **Full-Body Posture & Kinetic Momentum** via 33 3D skeletal landmark points.
4. **Affective Facial Expressions & Emotional State** via 52 facial blendshapes.

These spatial and biological telemetry vectors feed directly into a custom **Procedural Polyphonic Soundscape Engine** and an optional **Google Gemini LLM context bridge**, turning human physical motion, emotional affect, and physical surroundings into evolving, reactive ambient music in real time.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   Live Camera Stream   │
                                  │    (User Media API)    │
                                  └───────────┬────────────┘
                                              │
                                   Frame Ingestion Pipeline
                                              │
        ┌─────────────────────────────────────┼────────────────────────────────────┐
        │                                     │                                    │
        ▼                                     ▼                                    ▼
┌────────────────┐                   ┌────────────────┐                   ┌────────────────┐
│  MediaPipe     │                   │  MediaPipe     │                   │  TensorFlow.js │
│  FaceLandmarker│                   │  Pose & Hands  │                   │  COCO-SSD      │
│  (52 Shapes)   │                   │  (33 + 21 Pts) │                   │  (80 Classes)  │
└───────┬────────┘                   └────────┬───────┘                   └────────┬───────┘
        │                                     │                                    │
        ▼                                     ▼                                    ▼
┌────────────────┐                   ┌────────────────┐                   ┌────────────────┐
│ Affective      │                   │ Kinetic Energy │                   │ Object & Scene │
│ Emotion Metric │                   │ & Gestures     │                   │ Categorization │
└───────┬────────┘                   └────────┬───────┘                   └────────┬───────┘
        │                                     │                                    │
        └─────────────────────────────────────┼────────────────────────────────────┘
                                              │
                                  Telemetry Fusion Bus
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  Google Gemini 2.0 AI   │                       │ Deterministic Real-Time │
        │  Generative Vibe Prompt │                       │ Local Vibe Matrix Engine│
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Procedural Music Engine      │
                             │  • Multi-Harmonic Scale Mapper  │
                             │  • Kinetic Tempo Scaler (40-150)│
                             │  • Polyphonic Multi-Oscillators │
                             │  • Feedback Delay & Filter Mod  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Web Audio API / Tone.js Output  │
                             │ Dynamic Spatial Soundscape      │
                             └─────────────────────────────────┘
```

---

## 🧠 Multi-Model Computer Vision Deep Dive

DynoVision runs four neural network pipelines concurrently with hardware-accelerated WebAssembly (Wasm) and WebGL backends:

### 1. ✋ Hand Kinematics & Gesture Recognition (`MediaPipe HandLandmarker`)
* **Tracking Topology:** 21 three-dimensional coordinate landmarks per hand.
* **Pinch & Grip Metric:** Measures Euclidean distance between `Thumb Tip (4)` and `Index Tip (8)` to dynamically compute normalized pinch accuracy (0–100%).
* **Gesture Classifier:** Identifies dynamic gestures including *Open Palm*, *Precision Pinch*, *Index Pointing*, *Closed Fist*, and *Hand Waving*.
* **Velocity Tracking:** Quantifies hand traversal delta between consecutive frames to modulate high-frequency synthesizer arpeggios.

### 2. 🏃 Full-Body Kinetic Energy & Pose Tracking (`MediaPipe PoseLandmarker`)
* **Tracking Topology:** 33 full-body skeletal keypoints (shoulders, elbows, wrists, hips, knees, ankles).
* **Kinetic Momentum Engine:** Calculates rate of change across wrist and elbow vectors to yield a real-time **Kinetic Energy Score** ($0 - 100\%$).
* **Arm Elevation Analyzer:** Computes relative angle and elevation between wrists and shoulders (`Left/Right Hand Raised`).
* **Interactive Modulation:** Higher body kinetic energy drives tempo increases and transposes procedural synthesizer octaves up.

### 3. 😊 Affective Facial Biometrics (`MediaPipe FaceLandmarker`)
* **Tracking Topology:** Dense 468-point 3D mesh + 52 facial blendshape coefficients.
* **Affective State Machine:** Evaluates real-time combinations of blendshapes:
  * `smileScore`: `mouthSmileLeft` + `mouthSmileRight` $\rightarrow$ **Happy / Harmonic**
  * `frownScore`: `browDownLeft` + `browDownRight` + `mouthFrown` $\rightarrow$ **Melancholic / Minor**
  * `mouthOpenScore`: `jawOpen` $\rightarrow$ **Surprised / Drone**
  * `browRaiseScore`: `browInnerUp` $\rightarrow$ **Alert / Tension**
* **Scale Modulation:** Injects emotional coefficients directly into musical scale selection.

### 4. 📦 Spatial Object Classification (`TensorFlow.js COCO-SSD`)
* **Object Taxonomy:** Classifies 80 distinct real-world classes (Tech devices, dining tools, furniture, vehicles, animals).
* **Categorization Pipeline:** Maps detected bounding boxes into semantic zones (*Human, Tech/Electronics, Dining/Food, Fauna/Pet, Furniture/Interior, Vehicle, Media*).
* **Vibe Mapping:** Binds visible physical items to sound profiles (e.g., Laptops $\rightarrow$ Cyberpunk electronic, Books $\rightarrow$ Delicate piano, Plants $\rightarrow$ Ambient nature flute).

---

## 🎵 Procedural Soundscape & Acoustic Synthesis Engine

Rather than playing static audio loops, DynoVision implements a custom **Polyphonic Procedural Synthesizer** written directly over the browser's native `AudioContext` and `Tone.js`:

```typescript
// Dynamic harmonic scales selected according to visual & affective vectors
scales = {
  cyberpunk:   [0, 3, 7, 8, 10],   // Minor pentatonic with dark intervals
  tribal:      [0, 3, 5, 7, 10],   // Rhythmic modal progression
  pentatonic:  [0, 2, 4, 7, 9],    // Ethereal harmonious scale
  drone:       [0, 7],             // Root-fifth minimalist foundation
  melancholic: [0, 2, 3, 7, 8],    // Natural minor contemplative scale
  dissonant:   [0, 1, 6, 7, 11],   // High-tension tritone intervals
  major:       [0, 2, 4, 5, 7, 9, 11] // Uplifting acoustic harmonic series
}
```

### Acoustic Features:
* **Continuous Vibe Crossfading:** Seamless mathematical interpolation ($\sin / \cos$ curves) transitions between mood soundscapes without acoustic pops or discontinuities.
* **Kinetic Tempo Modulation:** Baseline tempo ($\approx 40\text{ BPM}$) scales dynamically up to $150\text{ BPM}$ in direct proportion to physical movement energy.
* **Quad-Oscillator Detuning:** Employs 4 detuned oscillators per note with lowpass biquad filters and feedback stereo delay ($0.33\text{s}$, $40\%$ feedback gain) for wide ambient space.
* **Sub-Bass Generator:** Injects octave-down pure sine foundation tones underneath melodic passages.

---

## 🤖 Google Gemini AI Context Bridge

DynoVision connects real-time vision telemetry to Google's Gemini multimodal models via `@google/genai`:
* **Model:** `gemini-flash-lite-latest`
* **Prompt Synthesis:** Serializes current facial emotion, active hand gesture, skeletal kinetic percentage, and visible objects into structured ambient generation prompts.
* **Zero-Latency Local Fallback:** If offline or if API quotas are exceeded, the deterministic **Local Vibe Matrix Engine** instantly calculates acoustic parameters with zero interruption to sound output.

---

## 🖥️ Cyberpunk HUD & Interface Controls

* **Multi-Layer Diagnostic Overlay:** Toggle between `All`, `Objects`, `Pose`, `Hands`, `Face`, or `Clean Feed`.
* **Telemetry Gauges:** Real-time FPS counters, kinetic energy percentage bars, hand pinch calibration indicators, and biometric emotion trackers.
* **Micro-Audio Feedback:** Built-in `Tone.Synth` spatial feedback triggers subtle sine chirps during button clicks and HUD interactions.
* **Responsive Motion UI:** Built with **Tailwind CSS v4** and **Motion** (`motion/react`) for silky $60\text{ FPS}$ hardware-accelerated animations.

---

## 🛠️ Complete Technology Stack

| Domain | Technology / Library | Role & Application |
| :--- | :--- | :--- |
| **Frontend Framework** | `React 19` + `TypeScript 5.8` | Component architecture, state synchronization, and lifecycle hooks |
| **Build & Bundler** | `Vite 6` + `@vitejs/plugin-react` | Hot Module Replacement, optimized bundling, and client deployment |
| **Styling & Animation** | `Tailwind CSS v4` + `Motion (Framer)` | Cyberpunk HUD aesthetics, glassmorphism, responsive diagnostic layouts |
| **Vision AI (Hands, Pose, Face)**| `@mediapipe/tasks-vision` | Wasm-accelerated 3D Landmark & blendshape extraction |
| **Vision AI (Objects)** | `@tensorflow/tfjs` + `@tensorflow-models/coco-ssd` | Real-time 80-class object bounding box detection |
| **Procedural Audio Synthesis** | Native `Web Audio API` + `Tone.js` | Polyphonic multi-oscillator synthesizer, delay lines, dynamic filters |
| **Generative Cloud AI** | `@google/genai` (`gemini-flash-lite-latest`) | Spatial-acoustic ambient prompt generation |
| **Production Server** | `Express 4` + `tsx` | Cloud-ready static distribution and production serving |
| **Iconography** | `lucide-react` | Diagnostic iconography and HUD telemetry symbols |

---

## 🔒 Privacy & Zero-Latency Edge Computing

* **100% Client-Side Computation:** Video frames captured from the user's camera are ingested directly into WebGL memory buffers and processed locally on the client machine.
* **Zero Data Transmission:** No image frames, video streams, or biometric data are transmitted to external servers or cloud storage.
* **Ephemeral Memory Model:** Video frames are immediately discarded following frame inference completion.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and build upon this engine for personal, research, or commercial applications.
