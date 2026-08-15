# 👁️ DynoVision

> **Interactive AI Computer Vision Engine detecting Objects, Hand Gestures, Body Poses, and Facial Biometrics to Synthesize Real-Time Adaptive Ambient Soundscapes.**

![DynoVision Banner](https://ai.google.dev/static/site-assets/images/share-ais-513315318.png)

---

## ✨ Features

- 👁️ **Multi-Model Local Computer Vision:**
  - **Object Detection ([COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)):** Scans surroundings in real time (laptops, phones, cups, pets, vehicles, stationery).
  - **Hand Gesture Recognition ([MediaPipe HandLandmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)):** 21 keypoints per hand tracking pinch gestures, pointing, fists, open palms, and velocity.
  - **Kinetic Body Tracking ([MediaPipe PoseLandmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)):** 33 3D body keypoints calculating posture, arm elevation, hand speed, and overall kinetic motion energy.
  - **Facial Emotion Biometrics ([MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)):** Facial blendshapes extraction (*Happy, Sad, Surprised, Neutral, Focused*).
- 🎵 **Adaptive Procedural Soundscape Engine:**
  - Dynamic scale & harmony modulation (*Cyberpunk, Tribal, Pentatonic, Drone, Melancholic, Dissonant, Major*).
  - Kinetic tempo scaling (40–150 BPM) and octave shifts based on real-time movement velocity.
  - Built with Web Audio API & [Tone.js](https://tonejs.github.io/) synthesizers.
- 🧠 **Google Gemini AI Integration:**
  - Uses `@google/genai` with `gemini-flash-lite-latest` to synthesize contextual 3–5 word ambient acoustic descriptions with offline fallback.
- 🔒 **Privacy-First Architecture:**
  - 100% of computer vision inference runs locally in-browser via WebGL/Wasm. No camera feeds are recorded or sent to remote servers.

---

## 🛠️ Tech Stack

- **Framework:** React 19, TypeScript, Vite 6
- **Styling & UI:** Tailwind CSS v4, Motion (Framer Motion), Lucide React
- **Computer Vision:** TensorFlow.js, COCO-SSD, MediaPipe Tasks Vision
- **Audio:** Web Audio API, Tone.js
- **AI/LLM:** Google Gen AI SDK (`@google/genai`)
- **Backend / Deployment:** Express, tsx, Node.js

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher)
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KaiX-Jr/dynovision.git
   cd dynovision
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment Variables:**
   Create a `.env` or `.env.local` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Start the Development Server:**
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser and allow camera permissions.

---

## 📜 Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Starts the Vite development server |
| `npm run build` | Builds the production bundle |
| `npm run preview` | Previews the production build locally |
| `npm run start` | Runs the Node.js / Express static server |

---

## 📄 License

MIT License
