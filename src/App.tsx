import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI } from '@google/genai';
import { 
  Camera, Square, Play, Music, Loader2, AlertCircle, 
  Activity, Cpu, ScanFace, Info, X, Accessibility, 
  Layers, Eye, EyeOff, Sparkles, Zap, Shield, Flame, Move, Hand
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'motion/react';
import * as Tone from 'tone';

let hoverSynth: Tone.Synth | null = null;

async function initAudio() {
  if (Tone.context.state !== 'running') {
    await Tone.start().catch(() => {});
  }
  if (!hoverSynth) {
    hoverSynth = new Tone.Synth({
      oscillator: { type: 'sine' },
      envelope: { attack: 0.01, decay: 0.1, sustain: 0, release: 0.01 }
    }).toDestination();
    hoverSynth.volume.value = -15;
  }
}

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
  category: string;
}

interface BodyMovementState {
  kineticEnergy: number; // 0 - 100
  leftHandSpeed: number; // 0 - 100
  rightHandSpeed: number; // 0 - 100
  armElevation: number; // 0 - 100
  activeGesture: string;
  bodyPoseDetected: boolean;
  leftHandRaised: boolean;
  rightHandRaised: boolean;
}

interface HandMovementState {
  handsCount: number;
  leftHandGesture: string;
  rightHandGesture: string;
  leftPinchScore: number;
  rightPinchScore: number;
  handSpeed: number;
  activeHandAction: string;
}

interface ConsoleState {
  emotion: string;
  objects: string[];
  objectDetails: { class: string; score: number; category: string }[];
  blendshapes: { smile: number; frown: number; mouthOpen: number; browRaise: number; eyeBlink: number; pucker: number };
  bodyMovement: BodyMovementState;
  handMovement: HandMovementState;
}

type OverlayMode = 'all' | 'objects' | 'pose' | 'hands' | 'face' | 'none';

class PCMPlayer {
  audioContext: AudioContext;
  nextStartTime: number;

  constructor(sampleRate: number = 48000) {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate });
    this.nextStartTime = this.audioContext.currentTime;
  }

  playChunk(base64Data: string) {
    const binaryString = atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // 16-bit PCM stereo
    const int16Array = new Int16Array(bytes.buffer);
    const numSamples = int16Array.length / 2;
    const leftChannel = new Float32Array(numSamples);
    const rightChannel = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
      leftChannel[i] = int16Array[i * 2] / 32768.0;
      rightChannel[i] = int16Array[i * 2 + 1] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(2, numSamples, this.audioContext.sampleRate);
    audioBuffer.getChannelData(0).set(leftChannel);
    audioBuffer.getChannelData(1).set(rightChannel);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const currentTime = this.audioContext.currentTime;
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime + 0.05;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;
  }

  stop() {
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }
}

class ProceduralMusicEngine {
  audioContext: AudioContext;
  isPlaying: boolean = false;
  currentVibe: string = 'minimalist ambient drone, quiet';
  targetVibe: string = 'minimalist ambient drone, quiet';
  vibeBlend: number = 1.0;
  nextNoteTime: number = 0;
  timerID: number | null = null;
  movementEnergy: number = 0; // 0 - 100
  
  // Scales (intervals from root)
  scales: Record<string, number[]> = {
    major: [0, 2, 4, 5, 7, 9, 11],
    minor: [0, 2, 3, 5, 7, 8, 10],
    pentatonic: [0, 2, 4, 7, 9],
    cyberpunk: [0, 3, 7, 8, 10],
    drone: [0, 7],
    melancholic: [0, 2, 3, 7, 8],
    dissonant: [0, 1, 6, 7, 11],
    tribal: [0, 3, 5, 7, 10]
  };

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  }

  setVibe(vibe: string) {
    if (this.targetVibe !== vibe) {
      if (this.vibeBlend >= 1.0) {
        this.currentVibe = this.targetVibe;
      }
      this.targetVibe = vibe;
      this.vibeBlend = 0.0;
    }
  }

  setMovementEnergy(energy: number) {
    this.movementEnergy = energy;
  }

  start() {
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    this.isPlaying = true;
    this.nextNoteTime = this.audioContext.currentTime + 0.1;
    this.scheduleNext();
  }

  stop() {
    this.isPlaying = false;
    if (this.timerID !== null) {
      clearTimeout(this.timerID);
      this.timerID = null;
    }
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }

  playNote(freq: number, type: OscillatorType, duration: number, vol: number, attack: number, time: number) {
    if (this.audioContext.state === 'closed') return;
    
    const numOscs = 4;
    const masterGain = this.audioContext.createGain();
    masterGain.connect(this.audioContext.destination);
    
    const now = time;
    masterGain.gain.setValueAtTime(0, now);
    masterGain.gain.linearRampToValueAtTime(vol, now + attack);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    const delay = this.audioContext.createDelay();
    delay.delayTime.value = 0.33;
    const feedback = this.audioContext.createGain();
    feedback.gain.value = 0.4;
    delay.connect(feedback);
    feedback.connect(delay);
    delay.connect(masterGain);

    for (let i = 0; i < numOscs; i++) {
      const osc = this.audioContext.createOscillator();
      const filter = this.audioContext.createBiquadFilter();
      
      osc.type = i % 2 === 0 ? type : 'sine';
      osc.frequency.value = freq * (1 + (i * 0.008));
      
      filter.type = 'lowpass';
      filter.frequency.value = freq * 2;
      filter.frequency.linearRampToValueAtTime(freq * 6, now + attack);
      filter.frequency.linearRampToValueAtTime(freq * 1.5, now + duration);
      
      osc.connect(filter);
      filter.connect(masterGain);
      filter.connect(delay);
      
      osc.start(now);
      osc.stop(now + duration);
    }
  }

  getTempoForVibe(vibe: string): number {
    let base = 40;
    if (vibe.includes('tribal') || vibe.includes('rhythmic')) base = 100;
    else if (vibe.includes('cyberpunk') || vibe.includes('electronic')) base = 65;
    else if (vibe.includes('kinetic') || vibe.includes('energetic')) base = 90;
    
    // Scale tempo up with kinetic energy (up to +35 BPM)
    const energyBonus = (this.movementEnergy / 100) * 35;
    return Math.min(150, base + energyBonus);
  }

  scheduleNext() {
    if (!this.isPlaying) return;
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    while (this.nextNoteTime < this.audioContext.currentTime + 0.5) {
      if (this.vibeBlend < 1.0) {
        this.vibeBlend += 0.02;
        if (this.vibeBlend > 1.0) this.vibeBlend = 1.0;
      }

      if (this.vibeBlend < 1.0) {
        const currentWeight = Math.cos(this.vibeBlend * 0.5 * Math.PI);
        const targetWeight = Math.sin(this.vibeBlend * 0.5 * Math.PI);
        this.generateTickForVibe(this.currentVibe, currentWeight, this.nextNoteTime);
        this.generateTickForVibe(this.targetVibe, targetWeight, this.nextNoteTime);
      } else {
        this.generateTickForVibe(this.targetVibe, 1.0, this.nextNoteTime);
      }
      
      const currentTempo = this.getTempoForVibe(this.currentVibe);
      const targetTempo = this.getTempoForVibe(this.targetVibe);
      const tempo = currentTempo * (1 - this.vibeBlend) + targetTempo * this.vibeBlend;
      
      const secondsPerBeat = 60.0 / tempo;
      this.nextNoteTime += secondsPerBeat;
    }
    
    this.timerID = window.setTimeout(() => this.scheduleNext(), 50);
  }

  generateTickForVibe(vibe: string, weight: number, time: number) {
    if (weight <= 0.01) return;
    
    const isCyberpunk = vibe.includes('cyberpunk') || vibe.includes('electronic');
    const isTribal = vibe.includes('tribal') || vibe.includes('rhythmic') || vibe.includes('happy');
    const isAcoustic = vibe.includes('acoustic') || vibe.includes('guitar');
    const isAmbient = vibe.includes('ambient') || vibe.includes('drone');
    const isSad = vibe.includes('sad') || vibe.includes('melancholy');
    const isTense = vibe.includes('angry') || vibe.includes('fear') || vibe.includes('disgust');
    const isEnergetic = vibe.includes('energetic') || vibe.includes('kinetic') || this.movementEnergy > 50;
    
    let scale = this.scales.pentatonic;
    let baseNote = 48; // C3
    let oscType: OscillatorType = 'sine';
    let vol = 0.08;
    let duration = 6.0;
    let attack = 3.0;

    if (isEnergetic) {
      scale = this.scales.cyberpunk;
      baseNote = 52; // E3
      oscType = 'sawtooth';
      vol = 0.07;
      duration = 2.0;
      attack = 0.2;
    } else if (isCyberpunk) {
      scale = this.scales.cyberpunk;
      baseNote = 36;
      oscType = 'sawtooth';
      vol = 0.04;
      duration = 4.0;
      attack = 2.0;
    } else if (isTribal) {
      scale = this.scales.tribal;
      baseNote = 43;
      oscType = 'square';
      vol = 0.06;
      duration = 1.5;
      attack = 0.1;
    } else if (isSad) {
      scale = this.scales.melancholic;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 8.0;
      attack = 4.0;
    } else if (isTense) {
      scale = this.scales.dissonant;
      baseNote = 36;
      oscType = 'sawtooth';
      vol = 0.05;
      duration = 5.0;
      attack = 1.5;
    } else if (isAcoustic) {
      scale = this.scales.major;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 5.0;
      attack = 2.0;
    } else if (isAmbient) {
      scale = this.scales.drone;
      baseNote = 36;
      oscType = 'sine';
      vol = 0.12;
      duration = 10.0;
      attack = 5.0;
    }

    vol *= weight;

    if (Math.random() > 0.2) {
      const noteIndex = scale[Math.floor(Math.random() * scale.length)];
      // Shift octave up if movement energy is high
      const octaveShift = (this.movementEnergy > 60 && Math.random() > 0.5) ? 12 : 0;
      const freq = 440 * Math.pow(2, (baseNote + noteIndex + octaveShift - 69) / 12);
      this.playNote(freq, oscType, duration, vol, attack, time);
    }
    
    if (Math.random() > 0.5) {
      const bassFreq = 440 * Math.pow(2, (baseNote - 12 - 69) / 12);
      this.playNote(bassFreq, 'sine', duration * 2, vol * 1.5, attack * 2, time);
    }
  }
}

const VIBE_MAP: Record<string, string> = {
  person: "ethereal ambient drone, calm",
  'cell phone': "cyberpunk synthwave, electronic",
  mouse: "precision synth arpeggios, digital",
  keyboard: "tactile rhythmic pulse, electronic",
  headphones: "deep binaural ambient, immersive",
  pen: "focused acoustic piano, delicate",
  laptop: "cyberpunk synthwave, electronic",
  tv: "cyberpunk synthwave, electronic",
  cup: "coffee shop jazz, chill acoustic",
  bottle: "coffee shop jazz, chill acoustic",
  bowl: "coffee shop jazz, chill acoustic",
  cat: "playful acoustic guitar, happy melody",
  dog: "playful acoustic guitar, happy melody",
  bird: "playful acoustic guitar, happy melody",
  car: "driving rock beat, fast tempo",
  bus: "driving rock beat, fast tempo",
  truck: "driving rock beat, fast tempo",
  chair: "ambient drone, relaxing",
  couch: "ambient drone, relaxing",
  bed: "ambient drone, relaxing",
  'potted plant': "ethereal flute, ambient nature",
  book: "classical piano, focused",
};

function normalizeClassName(cls: string): string {
  const c = cls.toLowerCase();
  if (c === 'scissors') return 'pen / writing tool';
  if (c === 'cell phone' || c === 'mobile phone' || c === 'smartphone') return 'cell phone';
  if (c === 'mouse') return 'mouse';
  if (c === 'keyboard') return 'keyboard';
  if (c === 'tv') return 'monitor / screen';
  if (c === 'remote') return 'remote control';
  return cls;
}

function getObjectCategory(className: string): string {
  const c = className.toLowerCase();
  if (['person', 'human'].some(k => c.includes(k))) return 'Human';
  if (['cell phone', 'smartphone', 'mobile phone', 'phone', 'laptop', 'tv', 'monitor', 'mouse', 'computer mouse', 'keyboard', 'remote', 'headphones', 'headset', 'earbuds', 'smartwatch', 'tablet'].some(k => c.includes(k))) return 'Tech / Electronics';
  if (['pen', 'pencil', 'marker', 'scissors', 'writing tool', 'stylus', 'stationery'].some(k => c.includes(k))) return 'Stationery / Tool';
  if (['cup', 'mug', 'bottle', 'bowl', 'fork', 'knife', 'spoon', 'wine glass', 'banana', 'apple'].some(k => c.includes(k))) return 'Dining / Food';
  if (['cat', 'dog', 'bird', 'horse', 'sheep', 'cow'].some(k => c.includes(k))) return 'Fauna / Pet';
  if (['chair', 'couch', 'bed', 'dining table', 'potted plant', 'desk', 'mat'].some(k => c.includes(k))) return 'Furniture / Interior';
  if (['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'airplane'].some(k => c.includes(k))) return 'Vehicle';
  if (['book', 'notebook', 'paper'].some(k => c.includes(k))) return 'Media / Print';
  return 'Object / Item';
}

function getVibeForObjectsAndPose(objects: string[], gesture: string, handAction: string, kineticEnergy: number): string {
  if (objects.length === 0 && gesture === 'CALM POSTURE' && handAction === 'NO HANDS DETECTED') return "minimalist ambient drone, quiet";
  
  const vibes = new Set<string>();
  if (kineticEnergy > 55) {
    vibes.add("high kinetic electronic synthwave");
  } else if (handAction.includes('PINCH') || handAction.includes('POINTING')) {
    vibes.add("precision synth arpeggios");
  } else if (gesture.includes('RAISED') || gesture.includes('WAVING') || handAction.includes('WAVING')) {
    vibes.add("uplifting energetic ambient pulse");
  }

  for (const obj of objects) {
    if (VIBE_MAP[obj]) {
      vibes.add(VIBE_MAP[obj]);
    } else {
      vibes.add("chill lofi beat");
    }
  }
  
  return Array.from(vibes).slice(0, 2).join(", ");
}

const getVibeFromGemini = async (objects: string[], emotion: string, gesture: string, handAction: string, kinetic: number): Promise<string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY });
    const prompt = `You are a soundscape generator. Based on the scene, output a 3-5 word ambient soundscape description (e.g., 'kinetic rhythmic synth drone', 'cyberpunk electronic arpeggios', 'chill acoustic ambient' or 'uplifting ambient pulse'). Do not include any other text. Never output 'pop', 'upbeat', or 'energetic'. Everything must be ambient soundscapes, adapted to movement and emotion. Scene: Feeling ${emotion}, gesture '${gesture}', hand movement '${handAction}' (kinetic motion: ${kinetic}%), objects: ${objects.length > 0 ? objects.join(', ') : 'none'}.`;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-lite-latest',
      contents: prompt,
    });
    return response.text?.trim() || "ambient drone, relaxing";
  } catch (e: any) {
    console.warn("Gemini API error (falling back to local vibe map):", e.message || e);
    return getVibeForObjectsAndPose(objects, gesture, handAction, kinetic) + `, ${emotion} mood`;
  }
};

const POSE_CONNECTIONS = [
  // Torso & Shoulders
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left Arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right Arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left Leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right Leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
  // Head
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 11], [0, 12]
];

const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17], [0, 17]
];

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState('Loading Object, Pose & Hand Detection Models...');
  const [currentPrompt, setCurrentPrompt] = useState('Waiting for camera...');
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all');
  const [activeTab, setActiveTab] = useState<'pose' | 'hands' | 'face'>('hands');
  const [showHUDPanels, setShowHUDPanels] = useState(true);

  const [consoleState, setConsoleState] = useState<ConsoleState>({
    emotion: 'neutral',
    objects: [],
    objectDetails: [],
    blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 },
    bodyMovement: {
      kineticEnergy: 0,
      leftHandSpeed: 0,
      rightHandSpeed: 0,
      armElevation: 0,
      activeGesture: 'STATIONARY / CALM POSTURE',
      bodyPoseDetected: false,
      leftHandRaised: false,
      rightHandRaised: false
    },
    handMovement: {
      handsCount: 0,
      leftHandGesture: 'NONE',
      rightHandGesture: 'NONE',
      leftPinchScore: 0,
      rightPinchScore: 0,
      handSpeed: 0,
      activeHandAction: 'NO HANDS DETECTED'
    }
  });
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement>(null);
  const handCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);
  const playerRef = useRef<PCMPlayer | null>(null);
  
  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  
  const geminiBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const isGeminiScanningRef = useRef(false);
  const geminiScanIntervalRef = useRef<any>(null);

  // Performance optimization refs for non-blocking ML inference
  const isObjectDetectingRef = useRef(false);
  const isHandDetectingRef = useRef(false);
  const isPoseDetectingRef = useRef(false);
  const isFaceDetectingRef = useRef(false);

  const lastObjectDetectTimeRef = useRef(0);
  const lastHandDetectTimeRef = useRef(0);
  const lastPoseDetectTimeRef = useRef(0);
  const lastFaceDetectTimeRef = useRef(0);

  const latestObjectPredictionsRef = useRef<any[]>([]);
  const latestHandResultRef = useRef<any>(null);
  const latestPoseResultRef = useRef<any>(null);
  const latestFaceResultRef = useRef<any>(null);

  const isPlayingRef = useRef(false);
  const lastPromptRef = useRef<string>("");
  const lastStateRef = useRef<string>("");
  const pendingStateRef = useRef<string | null>(null);
  const vibeTimeoutRef = useRef<any>(null);
  const lastStateUpdateTimeRef = useRef<number>(0);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const smoothedBlendshapesRef = useRef({ smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 });

  // Pose movement tracking refs
  const prevPoseLandmarksRef = useRef<any[] | null>(null);
  const prevPoseTimeRef = useRef<number>(0);
  const smoothedMovementRef = useRef({
    kineticEnergy: 0,
    leftHandSpeed: 0,
    rightHandSpeed: 0,
    armElevation: 0
  });

  // Hand movement tracking refs
  const prevHandLandmarksRef = useRef<Record<string, any[]>>({});
  const smoothedHandStateRef = useRef({
    leftPinch: 0,
    rightPinch: 0,
    handSpeed: 0
  });

  const runGeminiVisionScan = async () => {
    if (!videoRef.current || isGeminiScanningRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2) return;

    try {
      isGeminiScanningRef.current = true;
      const offCanvas = document.createElement('canvas');
      offCanvas.width = 480;
      offCanvas.height = 270;
      const offCtx = offCanvas.getContext('2d');
      if (!offCtx) return;

      offCtx.drawImage(video, 0, 0, 480, 270);
      const dataUrl = offCanvas.toDataURL('image/jpeg', 0.5);
      const base64Data = dataUrl.split(',')[1];

      if (!base64Data) return;

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY || '' });
      const prompt = `Locate and identify all visible objects in this desk or camera view.
Pay special attention to detecting: mouse, keyboard, headphones, headset, pen, pencil, marker, cell phone, smartphone, laptop, smartwatch, book, cup, mug, bottle, glasses, earbuds, desk items.
Return ONLY a valid JSON array of detected items:
[
  {
    "class": "headphones",
    "score": 0.95,
    "box": [ymin, xmin, ymax, xmax]
  }
]
Box coordinates MUST be normalized numbers from 0 to 1000 representing box edges [ymin, xmin, ymax, xmax].
Do not include any markdown or commentary outside the JSON array.`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: base64Data
            }
          },
          prompt
        ]
      });

      const responseText = response.text || '';
      const cleanJson = responseText.replace(/```json/g, '').replace(/```/g, '').trim();

      if (cleanJson.startsWith('[') && cleanJson.endsWith(']')) {
        const items = JSON.parse(cleanJson);
        const newGeminiBoxes = new Map<string, SmoothedBox>();

        if (Array.isArray(items) && canvasRef.current) {
          const cWidth = canvasRef.current.width;
          const cHeight = canvasRef.current.height;

          items.forEach((item: any, idx: number) => {
            if (item.class && Array.isArray(item.box) && item.box.length === 4) {
              const [ymin, xmin, ymax, xmax] = item.box;
              const x = (xmin / 1000) * cWidth;
              const y = (ymin / 1000) * cHeight;
              const width = Math.max(30, ((xmax - xmin) / 1000) * cWidth);
              const height = Math.max(30, ((ymax - ymin) / 1000) * cHeight);
              const rawLabel = normalizeClassName(String(item.class).toLowerCase());
              const category = getObjectCategory(rawLabel);

              const id = `gemini-${rawLabel}-${idx}`;
              newGeminiBoxes.set(id, {
                x, y, width, height,
                class: rawLabel,
                score: item.score || 0.92,
                opacity: 1.0,
                labelX: Math.min(cWidth - 140, x + width + 10),
                labelY: Math.max(20, y - 10),
                category
              });
            }
          });
        }
        geminiBoxesRef.current = newGeminiBoxes;
      }
    } catch (err) {
      console.warn("Gemini Vision object scan error:", err);
    } finally {
      isGeminiScanningRef.current = false;
    }
  };

  const playHoverSound = () => {
    try {
      initAudio();
      if (!hoverSynth || Tone.context.state !== 'running') return;
      
      const now = Tone.now();
      hoverSynth.triggerAttackRelease(800, 0.1, now);
      hoverSynth.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
    } catch (e) {}
  };

  useEffect(() => {
    const handleInteraction = () => initAudio();
    window.addEventListener('click', handleInteraction, { once: true });
    window.addEventListener('touchstart', handleInteraction, { once: true });
    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('touchstart', handleInteraction);
    };
  }, []);

  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();
        const cocoModel = await cocoSsd.load({ base: 'mobilenet_v2' });
        objectModelRef.current = cocoModel;

        let vision;
        try {
          vision = await FilesetResolver.forVisionTasks("/wasm");
        } catch (e) {
          console.warn("Local WASM load failed, falling back to CDN:", e);
          vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
          );
        }
        
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        faceLandmarkerRef.current = faceLandmarker;

        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
          },
          runningMode: "VIDEO",
          numPoses: 1
        });
        poseLandmarkerRef.current = poseLandmarker;

        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        handLandmarkerRef.current = handLandmarker;

        setIsModelLoaded(true);
        setStatus('Idle');
      } catch (err: any) {
        console.error("Failed to load models:", err);
        setStatus('Error loading vision models');
        setErrorMsg(err.message || 'Vision model load failed');
      }
    };
    
    loadModels();
    
    return () => {
      stopSession();
    };
  }, []);

  const runDetection = () => {
    if (!isPlayingRef.current || !videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const now = performance.now();

      // Trigger background Gemini AI Vision scan periodically
      if (!geminiScanIntervalRef.current) {
        runGeminiVisionScan();
        geminiScanIntervalRef.current = setInterval(runGeminiVisionScan, 3000);
      }

      // -------------------------------------------------------------
      // ASYNC STAGGERED ML INFERENCES (Non-blocking background triggers)
      // -------------------------------------------------------------
      
      // 1. Object Detection Trigger (~160ms throttle = 6.25 FPS)
      if (objectModelRef.current && (overlayMode === 'all' || overlayMode === 'objects') && !isObjectDetectingRef.current && (now - lastObjectDetectTimeRef.current > 160)) {
        isObjectDetectingRef.current = true;
        lastObjectDetectTimeRef.current = now;
        
        objectModelRef.current.detect(video, 20, 0.18)
          .then(rawPredictions => {
            latestObjectPredictionsRef.current = rawPredictions.map(p => ({
              ...p,
              class: normalizeClassName(p.class)
            }));
          })
          .catch(() => {})
          .finally(() => {
            isObjectDetectingRef.current = false;
          });
      }

      // 2. Hand Detection Trigger (~33ms throttle = 30 FPS)
      if (handLandmarkerRef.current && (overlayMode === 'all' || overlayMode === 'hands') && !isHandDetectingRef.current && (now - lastHandDetectTimeRef.current > 33)) {
        isHandDetectingRef.current = true;
        lastHandDetectTimeRef.current = now;
        try {
          latestHandResultRef.current = handLandmarkerRef.current.detectForVideo(video, now);
        } catch (e) {}
        isHandDetectingRef.current = false;
      }

      // 3. Pose Detection Trigger (~50ms throttle = 20 FPS)
      if (poseLandmarkerRef.current && (overlayMode === 'all' || overlayMode === 'pose') && !isPoseDetectingRef.current && (now - lastPoseDetectTimeRef.current > 50)) {
        isPoseDetectingRef.current = true;
        lastPoseDetectTimeRef.current = now;
        try {
          latestPoseResultRef.current = poseLandmarkerRef.current.detectForVideo(video, now);
        } catch (e) {}
        isPoseDetectingRef.current = false;
      }

      // 4. Face Detection Trigger (~50ms throttle = 20 FPS)
      if (faceLandmarkerRef.current && (overlayMode === 'all' || overlayMode === 'face') && !isFaceDetectingRef.current && (now - lastFaceDetectTimeRef.current > 50)) {
        isFaceDetectingRef.current = true;
        lastFaceDetectTimeRef.current = now;
        try {
          latestFaceResultRef.current = faceLandmarkerRef.current.detectForVideo(video, now);
        } catch (e) {}
        isFaceDetectingRef.current = false;
      }

      // -------------------------------------------------------------
      // SYNCHRONOUS 60 FPS RENDER LOOP
      // -------------------------------------------------------------
      try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const detectedClasses = new Set<string>();
        const objectDetailsList: { class: string; score: number; category: string }[] = [];

        // 1. HIGH-SENSITIVITY OBJECT DETECTION (COCO-SSD + GEMINI AI)
        if (overlayMode === 'all' || overlayMode === 'objects') {
          const predictions = latestObjectPredictionsRef.current;
          const newSmoothedBoxes = new Map<string, SmoothedBox>();
          const unassignedPredictions = [...predictions];

          smoothedBoxesRef.current.forEach((box, id) => {
            let closestIdx = -1;
            let minDist = Infinity;
            unassignedPredictions.forEach((pred, idx) => {
              if (pred.class === box.class) {
                const [px, py, pw, ph] = pred.bbox;
                const dist = Math.hypot(px + pw/2 - (box.x + box.width/2), py + ph/2 - (box.y + box.height/2));
                if (dist < 150) {
                  if (dist < minDist) {
                    minDist = dist;
                    closestIdx = idx;
                  }
                }
              }
            });

            if (closestIdx !== -1) {
              const pred = unassignedPredictions[closestIdx];
              const [px, py, pw, ph] = pred.bbox;
              const lerp = 0.15;
              box.x += (px - box.x) * lerp;
              box.y += (py - box.y) * lerp;
              box.width += (pw - box.width) * lerp;
              box.height += (ph - box.height) * lerp;
              box.opacity = Math.min(1, box.opacity + 0.1);
              box.score = pred.score;
              
              const targetLabelX = box.x + box.width + 20;
              const targetLabelY = box.y - 20;
              box.labelX += (targetLabelX - box.labelX) * lerp;
              box.labelY += (targetLabelY - box.labelY) * lerp;

              newSmoothedBoxes.set(id, box);
              unassignedPredictions.splice(closestIdx, 1);
              detectedClasses.add(box.class);
              objectDetailsList.push({ class: box.class, score: box.score, category: box.category });
            } else {
              box.opacity -= 0.05;
              if (box.opacity > 0) {
                newSmoothedBoxes.set(id, box);
                detectedClasses.add(box.class);
                objectDetailsList.push({ class: box.class, score: box.score, category: box.category });
              }
            }
          });

          unassignedPredictions.forEach((pred) => {
            const id = Math.random().toString(36).substring(7);
            const [x, y, width, height] = pred.bbox;
            const category = getObjectCategory(pred.class);
            newSmoothedBoxes.set(id, {
              x, y, width, height, class: pred.class, score: pred.score, opacity: 0,
              labelX: x + width + 40, labelY: y - 40, category
            });
            detectedClasses.add(pred.class);
            objectDetailsList.push({ class: pred.class, score: pred.score, category });
          });

          smoothedBoxesRef.current = newSmoothedBoxes;

          // Draw Local COCO-SSD Object Bounding Boxes
          smoothedBoxesRef.current.forEach((box) => {
            const { x, y, width, height, opacity, labelX, labelY } = box;
            const text = `${box.class.toUpperCase()} (${Math.round(box.score * 100)}%)`;

            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.8})`;
            ctx.lineWidth = 1.5;

            // Corner brackets
            const cornerLength = Math.min(18, width / 4, height / 4);
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLength);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLength, y);
            
            ctx.moveTo(x + width - cornerLength, y);
            ctx.lineTo(x + width, y);
            ctx.lineTo(x + width, y + cornerLength);
            
            ctx.moveTo(x + width, y + height - cornerLength);
            ctx.lineTo(x + width, y + height);
            ctx.lineTo(x + width - cornerLength, y + height);
            
            ctx.moveTo(x + cornerLength, y + height);
            ctx.lineTo(x, y + height);
            ctx.lineTo(x, y + height - cornerLength);
            ctx.stroke();

            // Center target crosshair
            ctx.beginPath();
            ctx.moveTo(x + width / 2 - 6, y + height / 2);
            ctx.lineTo(x + width / 2 + 6, y + height / 2);
            ctx.moveTo(x + width / 2, y + height / 2 - 6);
            ctx.lineTo(x + width / 2, y + height / 2 + 6);
            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
            ctx.stroke();

            // Line to label
            ctx.beginPath();
            ctx.moveTo(x + width, y);
            ctx.lineTo(labelX, labelY + 16);
            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.5})`;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Minimalist Label Box
            ctx.font = '500 10px "JetBrains Mono", monospace';
            const textWidth = ctx.measureText(text).width;
            ctx.fillStyle = `rgba(0, 0, 0, ${opacity * 0.7})`;
            ctx.fillRect(labelX, labelY, textWidth + 12, 18);
            ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
            ctx.strokeRect(labelX, labelY, textWidth + 12, 18);
            
            ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
            ctx.fillText(text, labelX + 6, labelY + 12);
          });

          // Draw Gemini AI Open-Vocabulary Object Detections
          geminiBoxesRef.current.forEach((gBox) => {
            detectedClasses.add(gBox.class);
            objectDetailsList.push({ class: gBox.class, score: gBox.score, category: gBox.category });

            const { x, y, width, height, labelX, labelY } = gBox;
            const text = `${gBox.class.toUpperCase()} (${Math.round(gBox.score * 100)}%) [AI]`;

            ctx.strokeStyle = `rgba(255, 255, 255, 0.95)`;
            ctx.lineWidth = 1.5;

            const cornerLength = Math.min(18, width / 4, height / 4);
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLength);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLength, y);
            
            ctx.moveTo(x + width - cornerLength, y);
            ctx.lineTo(x + width, y);
            ctx.lineTo(x + width, y + cornerLength);
            
            ctx.moveTo(x + width, y + height - cornerLength);
            ctx.lineTo(x + width, y + height);
            ctx.lineTo(x + width - cornerLength, y + height);
            
            ctx.moveTo(x + cornerLength, y + height);
            ctx.lineTo(x, y + height);
            ctx.lineTo(x, y + height - cornerLength);
            ctx.stroke();

            // Center target crosshair
            ctx.beginPath();
            ctx.moveTo(x + width / 2 - 6, y + height / 2);
            ctx.lineTo(x + width / 2 + 6, y + height / 2);
            ctx.moveTo(x + width / 2, y + height / 2 - 6);
            ctx.lineTo(x + width / 2, y + height / 2 + 6);
            ctx.strokeStyle = `rgba(255, 255, 255, 0.6)`;
            ctx.stroke();

            // Line to label
            ctx.beginPath();
            ctx.moveTo(x + width, y);
            ctx.lineTo(labelX, labelY + 16);
            ctx.strokeStyle = `rgba(255, 255, 255, 0.6)`;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Minimalist Label Box
            ctx.font = '600 10px "JetBrains Mono", monospace';
            const textWidth = ctx.measureText(text).width;
            ctx.fillStyle = `rgba(0, 0, 0, 0.85)`;
            ctx.fillRect(labelX, labelY, textWidth + 12, 18);
            ctx.strokeStyle = `rgba(255, 255, 255, 0.8)`;
            ctx.strokeRect(labelX, labelY, textWidth + 12, 18);
            
            ctx.fillStyle = `#ffffff`;
            ctx.fillText(text, labelX + 6, labelY + 12);
          });
        }

        // -------------------------------------------------------------
        // 2. HAND MOVEMENT & GESTURE DETECTION (MEDIAPIPE HAND LANDMARKER)
        // -------------------------------------------------------------
        let currentHandState: HandMovementState = {
          handsCount: 0,
          leftHandGesture: 'NONE',
          rightHandGesture: 'NONE',
          leftPinchScore: 0,
          rightPinchScore: 0,
          handSpeed: 0,
          activeHandAction: 'NO HANDS DETECTED'
        };

        const nowTime = performance.now();

        if (overlayMode === 'all' || overlayMode === 'hands') {
          const handResult = latestHandResultRef.current;

          if (handResult.landmarks && handResult.landmarks.length > 0) {
            currentHandState.handsCount = handResult.landmarks.length;
            let totalHandSpeed = 0;

            handResult.landmarks.forEach((handPoints, handIdx) => {
              const handednessCategory = handResult.handedness[handIdx]?.[0];
              const handLabel = handednessCategory?.displayName === 'Left' ? 'Left' : 'Right';
              
              // 1. Calculate Pinch score (distance between Thumb Tip #4 and Index Tip #8)
              const thumbTip = handPoints[4];
              const indexTip = handPoints[8];
              const middleTip = handPoints[12];
              const ringTip = handPoints[16];
              const pinkyTip = handPoints[20];
              const wrist = handPoints[0];

              const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
              // Scale pinch score 0 - 100% (0.02 is tight pinch, 0.2 is wide open)
              const rawPinch = Math.max(0, Math.min(100, Math.round((1 - (pinchDist - 0.02) / 0.18) * 100)));

              if (handLabel === 'Left') currentHandState.leftPinchScore = rawPinch;
              else currentHandState.rightPinchScore = rawPinch;

              // 2. Gesture Classification
              // Distance of fingertips to wrist
              const indexWristDist = Math.hypot(indexTip.x - wrist.x, indexTip.y - wrist.y);
              const middleWristDist = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);
              const ringWristDist = Math.hypot(ringTip.x - wrist.x, ringTip.y - wrist.y);
              const pinkyWristDist = Math.hypot(pinkyTip.x - wrist.x, pinkyTip.y - wrist.y);

              let gesture = 'OPEN PALM';
              if (rawPinch > 75) {
                gesture = 'PINCH GESTURE';
              } else if (indexWristDist < 0.25 && middleWristDist < 0.25 && ringWristDist < 0.25 && pinkyWristDist < 0.25) {
                gesture = 'FIST / CLENCH';
              } else if (indexWristDist > 0.35 && middleWristDist > 0.35 && ringWristDist < 0.25 && pinkyWristDist < 0.25) {
                gesture = 'PEACE / V-SIGN';
              } else if (indexWristDist > 0.35 && middleWristDist < 0.25 && ringWristDist < 0.25 && pinkyWristDist < 0.25) {
                gesture = 'POINTING';
              }

              if (handLabel === 'Left') currentHandState.leftHandGesture = gesture;
              else currentHandState.rightHandGesture = gesture;

              // 3. Hand Movement Speed
              const prevHandPoints = prevHandLandmarksRef.current[handLabel];
              if (prevHandPoints) {
                const disp = Math.hypot(wrist.x - prevHandPoints[0].x, wrist.y - prevHandPoints[0].y);
                totalHandSpeed += disp * 400;
              }
              prevHandLandmarksRef.current[handLabel] = handPoints;

              // 4. Draw Hand Skeleton Overlay on main canvas
              ctx.save();
              ctx.lineWidth = 2;

              // Connections
              HAND_CONNECTIONS.forEach(([i, j]) => {
                const ptA = handPoints[i];
                const ptB = handPoints[j];
                if (ptA && ptB) {
                  const x1 = ptA.x * canvas.width;
                  const y1 = ptA.y * canvas.height;
                  const x2 = ptB.x * canvas.width;
                  const y2 = ptB.y * canvas.height;

                  ctx.strokeStyle = handLabel === 'Left' ? 'rgba(255, 255, 255, 0.85)' : 'rgba(255, 255, 255, 0.95)';
                  ctx.beginPath();
                  ctx.moveTo(x1, y1);
                  ctx.lineTo(x2, y2);
                  ctx.stroke();
                }
              });

              // Joints & Fingertips
              handPoints.forEach((pt: any, idx: number) => {
                const px = pt.x * canvas.width;
                const py = pt.y * canvas.height;
                const isTip = [4, 8, 12, 16, 20].includes(idx);

                ctx.fillStyle = isTip ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.7)';
                ctx.beginPath();
                ctx.arc(px, py, isTip ? 4 : 2, 0, Math.PI * 2);
                ctx.fill();

                if (isTip && rawPinch > 50) {
                  ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                  ctx.lineWidth = 1;
                  ctx.beginPath();
                  ctx.arc(px, py, 7, 0, Math.PI * 2);
                  ctx.stroke();
                }
              });

              // Label on Wrist
              const wx = wrist.x * canvas.width;
              const wy = wrist.y * canvas.height;
              ctx.font = '600 9px "JetBrains Mono", monospace';
              ctx.fillStyle = 'rgba(0,0,0,0.8)';
              ctx.fillRect(wx - 30, wy + 8, 60, 14);
              ctx.strokeStyle = 'rgba(255,255,255,0.4)';
              ctx.strokeRect(wx - 30, wy + 8, 60, 14);
              ctx.fillStyle = '#ffffff';
              ctx.fillText(`${handLabel.toUpperCase()}: ${gesture}`, wx - 26, wy + 18);

              ctx.restore();
            });

            // Hand Action summary string
            const sh = smoothedHandStateRef.current;
            sh.handSpeed += (totalHandSpeed - sh.handSpeed) * 0.2;
            currentHandState.handSpeed = Math.round(Math.min(100, sh.handSpeed));

            if (currentHandState.leftHandGesture === 'PINCH GESTURE' || currentHandState.rightHandGesture === 'PINCH GESTURE') {
              currentHandState.activeHandAction = 'PINCH / PRECISION TONE';
            } else if (currentHandState.leftHandGesture === 'POINTING' || currentHandState.rightHandGesture === 'POINTING') {
              currentHandState.activeHandAction = 'POINTING GESTURE';
            } else if (currentHandState.leftHandGesture === 'FIST / CLENCH' || currentHandState.rightHandGesture === 'FIST / CLENCH') {
              currentHandState.activeHandAction = 'CLENCHED FIST';
            } else if (sh.handSpeed > 35) {
              currentHandState.activeHandAction = 'FAST HAND WAVING';
            } else {
              currentHandState.activeHandAction = `HANDS DETECTED (${currentHandState.handsCount})`;
            }

            // Render Secondary Hand Scan Canvas
            if (handCanvasRef.current) {
              const hCanvas = handCanvasRef.current;
              const hCtx = hCanvas.getContext('2d');
              if (hCtx) {
                hCtx.clearRect(0, 0, hCanvas.width, hCanvas.height);
                hCtx.save();

                handResult.landmarks.forEach((handPoints) => {
                  HAND_CONNECTIONS.forEach(([i, j]) => {
                    const ptA = handPoints[i];
                    const ptB = handPoints[j];
                    if (ptA && ptB) {
                      hCtx.strokeStyle = '#ffffff';
                      hCtx.lineWidth = 1.5;
                      hCtx.beginPath();
                      hCtx.moveTo(ptA.x * hCanvas.width, ptA.y * hCanvas.height);
                      hCtx.lineTo(ptB.x * hCanvas.width, ptB.y * hCanvas.height);
                      hCtx.stroke();
                    }
                  });

                  handPoints.forEach((pt: any, idx: number) => {
                    const isTip = [4, 8, 12, 16, 20].includes(idx);
                    hCtx.fillStyle = isTip ? '#ffffff' : 'rgba(255, 255, 255, 0.6)';
                    hCtx.beginPath();
                    hCtx.arc(pt.x * hCanvas.width, pt.y * hCanvas.height, isTip ? 3 : 1.5, 0, Math.PI * 2);
                    hCtx.fill();
                  });
                });

                hCtx.restore();
              }
            }
          }
        }

        // -------------------------------------------------------------
        // 3. BODY PARTS MOVEMENT DETECTION (MEDIAPIPE POSE LANDMARKER)
        // -------------------------------------------------------------
        let currentMovementState: BodyMovementState = {
          kineticEnergy: 0,
          leftHandSpeed: 0,
          rightHandSpeed: 0,
          armElevation: 0,
          activeGesture: 'STATIONARY / CALM POSTURE',
          bodyPoseDetected: false,
          leftHandRaised: false,
          rightHandRaised: false
        };

        if (overlayMode === 'all' || overlayMode === 'pose') {
          const poseResult = latestPoseResultRef.current;
          
          if (poseResult.landmarks && poseResult.landmarks.length > 0) {
            const poseLandmarks = poseResult.landmarks[0];
            currentMovementState.bodyPoseDetected = true;

            // Velocity and movement energy calculation
            const dt = prevPoseTimeRef.current > 0 ? (nowTime - prevPoseTimeRef.current) / 1000 : 0.033;
            prevPoseTimeRef.current = nowTime;

            let totalDisplacement = 0;
            let leftHandDisp = 0;
            let rightHandDisp = 0;

            if (prevPoseLandmarksRef.current) {
              const prev = prevPoseLandmarksRef.current;
              
              // Wrists motion
              const lWristCurr = poseLandmarks[15];
              const lWristPrev = prev[15];
              if (lWristCurr && lWristPrev) {
                leftHandDisp = Math.hypot(lWristCurr.x - lWristPrev.x, lWristCurr.y - lWristPrev.y);
              }

              const rWristCurr = poseLandmarks[16];
              const rWristPrev = prev[16];
              if (rWristCurr && rWristPrev) {
                rightHandDisp = Math.hypot(rWristCurr.x - rWristPrev.x, rWristCurr.y - rWristPrev.y);
              }

              // Sample key joints for total kinetic motion (shoulders, elbows, wrists, knees, ankles)
              const sampleIndices = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28];
              sampleIndices.forEach(idx => {
                if (poseLandmarks[idx] && prev[idx]) {
                  totalDisplacement += Math.hypot(poseLandmarks[idx].x - prev[idx].x, poseLandmarks[idx].y - prev[idx].y);
                }
              });
            }

            prevPoseLandmarksRef.current = poseLandmarks;

            // Calculate speeds normalized to 0-100% scale
            const rawKinetic = Math.min(100, (totalDisplacement / Math.max(0.01, dt)) * 220);
            const rawLHandSpeed = Math.min(100, (leftHandDisp / Math.max(0.01, dt)) * 300);
            const rawRHandSpeed = Math.min(100, (rightHandDisp / Math.max(0.01, dt)) * 300);

            // Smooth movement signals
            const sm = smoothedMovementRef.current;
            sm.kineticEnergy += (rawKinetic - sm.kineticEnergy) * 0.2;
            sm.leftHandSpeed += (rawLHandSpeed - sm.leftHandSpeed) * 0.2;
            sm.rightHandSpeed += (rawRHandSpeed - sm.rightHandSpeed) * 0.2;

            // Check Hand / Arm Elevation relative to Shoulders
            const lShoulder = poseLandmarks[11];
            const rShoulder = poseLandmarks[12];
            const lWrist = poseLandmarks[15];
            const rWrist = poseLandmarks[16];

            const leftHandRaised = lWrist && lShoulder && lWrist.y < lShoulder.y;
            const rightHandRaised = rWrist && rShoulder && rWrist.y < rShoulder.y;

            let elevationVal = 0;
            if (leftHandRaised) elevationVal += 50;
            if (rightHandRaised) elevationVal += 50;
            sm.armElevation += (elevationVal - sm.armElevation) * 0.2;

            // Gesture / Pose classification
            let gesture = 'STATIONARY / CALM POSTURE';
            if (leftHandRaised && rightHandRaised) {
              gesture = 'BOTH ARMS RAISED';
            } else if (leftHandRaised) {
              gesture = 'LEFT ARM RAISED';
            } else if (rightHandRaised) {
              gesture = 'RIGHT ARM RAISED';
            } else if (sm.leftHandSpeed > 35 || sm.rightHandSpeed > 35) {
              gesture = 'RAPID HAND WAVING';
            } else if (sm.kineticEnergy > 45) {
              gesture = 'HIGH KINETIC MOTION';
            } else if (sm.kineticEnergy > 15) {
              gesture = 'MODERATE BODY MOTION';
            }

            currentMovementState = {
              kineticEnergy: Math.round(sm.kineticEnergy),
              leftHandSpeed: Math.round(sm.leftHandSpeed),
              rightHandSpeed: Math.round(sm.rightHandSpeed),
              armElevation: Math.round(sm.armElevation),
              activeGesture: gesture,
              bodyPoseDetected: true,
              leftHandRaised,
              rightHandRaised
            };

            // Draw Skeleton Overlay on main canvas
            ctx.save();
            ctx.lineWidth = 2;
            
            // Draw Bone Connections
            POSE_CONNECTIONS.forEach(([i, j]) => {
              const ptA = poseLandmarks[i];
              const ptB = poseLandmarks[j];
              if (ptA && ptB && (ptA.visibility ?? 1) > 0.3 && (ptB.visibility ?? 1) > 0.3) {
                const x1 = ptA.x * canvas.width;
                const y1 = ptA.y * canvas.height;
                const x2 = ptB.x * canvas.width;
                const y2 = ptB.y * canvas.height;

                const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
                if (sm.kineticEnergy > 40) {
                  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
                  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.5)');
                } else {
                  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.6)');
                  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.2)');
                }

                ctx.strokeStyle = gradient;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
              }
            });

            // Draw Joint Markers
            poseLandmarks.forEach((pt: any, idx: number) => {
              if (pt && (pt.visibility ?? 1) > 0.3) {
                const px = pt.x * canvas.width;
                const py = pt.y * canvas.height;
                const isWrist = idx === 15 || idx === 16;
                const speed = isWrist ? (idx === 15 ? sm.leftHandSpeed : sm.rightHandSpeed) : sm.kineticEnergy;

                ctx.fillStyle = isWrist ? 'rgba(255, 255, 255, 0.95)' : 'rgba(255, 255, 255, 0.7)';
                ctx.beginPath();
                ctx.arc(px, py, isWrist ? 4 : 2.5, 0, Math.PI * 2);
                ctx.fill();

                if (speed > 10) {
                  ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(0.8, speed / 80)})`;
                  ctx.lineWidth = 1;
                  ctx.beginPath();
                  ctx.arc(px, py, 4 + speed * 0.12, 0, Math.PI * 2);
                  ctx.stroke();
                }
              }
            });

            ctx.restore();

            // Render Secondary Kinetic Pose Scan Canvas
            if (poseCanvasRef.current) {
              const pCanvas = poseCanvasRef.current;
              const pCtx = pCanvas.getContext('2d');
              if (pCtx) {
                pCtx.clearRect(0, 0, pCanvas.width, pCanvas.height);
                pCtx.save();

                pCtx.lineWidth = 1.5;
                POSE_CONNECTIONS.forEach(([i, j]) => {
                  const ptA = poseLandmarks[i];
                  const ptB = poseLandmarks[j];
                  if (ptA && ptB) {
                    pCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                    pCtx.beginPath();
                    pCtx.moveTo(ptA.x * pCanvas.width, ptA.y * pCanvas.height);
                    pCtx.lineTo(ptB.x * pCanvas.width, ptB.y * pCanvas.height);
                    pCtx.stroke();
                  }
                });

                poseLandmarks.forEach((pt: any, idx: number) => {
                  if (pt) {
                    pCtx.fillStyle = idx === 15 || idx === 16 ? '#ffffff' : 'rgba(255, 255, 255, 0.6)';
                    pCtx.beginPath();
                    pCtx.arc(pt.x * pCanvas.width, pt.y * pCanvas.height, idx === 15 || idx === 16 ? 3 : 1.5, 0, Math.PI * 2);
                    pCtx.fill();
                  }
                });

                pCtx.restore();
              }
            }
          }
        }

        // -------------------------------------------------------------
        // 4. FACE DETECTION & BIOMETRICS (MEDIAPIPE FACE LANDMARKER)
        // -------------------------------------------------------------
        let currentEmotion = "neutral";
        let currentBlendshapes = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
        
        if (overlayMode === 'all' || overlayMode === 'face') {
          const faceResult = latestFaceResultRef.current;
          
          if (faceCanvasRef.current) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext('2d');
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
              
              if (faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
                const time = nowTime / 1500;
                
                let minX = video.videoWidth, maxX = 0, minY = video.videoHeight, maxY = 0;
                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                }
                const faceWidth = maxX - minX;
                const faceHeight = maxY - minY;
                const centerX = minX + faceWidth / 2;
                const centerY = minY + faceHeight / 2;
                
                const scanY = minY + ((Math.sin(time) + 1) / 2) * faceHeight;
                const scale = Math.min(fCanvas.width / faceWidth, fCanvas.height / faceHeight) * 0.8;

                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  const dist = Math.abs(py - scanY) / faceHeight;
                  const opacity = Math.max(0.15, 1.0 - dist * 4); 
                  
                  fCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                  fCtx.beginPath();
                  const drawX = fCanvas.width/2 + (px - centerX) * scale;
                  const drawY = fCanvas.height/2 + (py - centerY) * scale;
                  
                  fCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
                  fCtx.fill();
                }
              }
            }
          }

          if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
            const blendshapes = faceResult.faceBlendshapes[0].categories;
            const getScore = (name: string) => blendshapes.find(b => b.categoryName === name)?.score || 0;
            
            currentBlendshapes.smile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
            currentBlendshapes.frown = Math.min(1, (getScore('mouthFrownLeft') + getScore('mouthFrownRight') + getScore('mouthRollLower')) * 5);
            currentBlendshapes.mouthOpen = getScore('jawOpen');
            currentBlendshapes.browRaise = (getScore('browInnerUp') + getScore('browOuterUpLeft') + getScore('browOuterUpRight')) / 3;
            currentBlendshapes.eyeBlink = (getScore('eyeBlinkLeft') + getScore('eyeBlinkRight')) / 2;
            currentBlendshapes.pucker = getScore('mouthPucker');
            
            const surpriseScore = (getScore('jawOpen') + getScore('browInnerUp')) / 2;
            const angerScore = (getScore('browDownLeft') + getScore('browDownRight') + getScore('mouthPressLeft')) / 3;
            const fearScore = ((getScore('jawOpen') + getScore('browInnerUp') + getScore('mouthStretchLeft') + getScore('mouthStretchRight')) / 4) * 0.6;
            const disgustScore = Math.min(1, (getScore('noseSneerLeft') + getScore('noseSneerRight') + getScore('mouthUpperUpLeft') + getScore('mouthUpperUpRight')) * 4);
            
            const emotions = [
              { name: 'happy', score: currentBlendshapes.smile },
              { name: 'sadness', score: currentBlendshapes.frown },
              { name: 'surprised', score: surpriseScore },
              { name: 'angry', score: angerScore },
              { name: 'fear', score: fearScore },
              { name: 'disgust', score: disgustScore }
            ];
            
            const maxEmotion = emotions.reduce((max, e) => e.score > max.score ? e : max, emotions[0]);
            currentEmotion = maxEmotion.score > 0.2 ? maxEmotion.name : "neutral";
          }
        }

        // -------------------------------------------------------------
        // 5. REACT STATE THROTTLED UPDATE (~10 FPS)
        // -------------------------------------------------------------
        const now = performance.now();
        if (now - lastStateUpdateTimeRef.current > 100) {
          const smoothingFactor = 0.15;
          const smoothed = smoothedBlendshapesRef.current;
          smoothed.smile += (currentBlendshapes.smile - smoothed.smile) * smoothingFactor;
          smoothed.frown += (currentBlendshapes.frown - smoothed.frown) * smoothingFactor;
          smoothed.mouthOpen += (currentBlendshapes.mouthOpen - smoothed.mouthOpen) * smoothingFactor;
          smoothed.browRaise += (currentBlendshapes.browRaise - smoothed.browRaise) * smoothingFactor;
          smoothed.eyeBlink += (currentBlendshapes.eyeBlink - smoothed.eyeBlink) * smoothingFactor;
          smoothed.pucker += (currentBlendshapes.pucker - smoothed.pucker) * smoothingFactor;

          const classesArray = Array.from(detectedClasses).sort();

          setConsoleState({
            emotion: currentEmotion,
            objects: classesArray,
            objectDetails: objectDetailsList,
            blendshapes: { ...smoothed },
            bodyMovement: currentMovementState,
            handMovement: currentHandState
          });
          lastStateUpdateTimeRef.current = now;
        }

        // Combine scene state for audio vibe sync
        const classesArray = Array.from(detectedClasses).sort();
        const stateString = `${classesArray.join(',')}|${currentEmotion}|${currentMovementState.activeGesture}|${currentHandState.activeHandAction}|${Math.round(currentMovementState.kineticEnergy/10)}`;
        
        if (stateString !== pendingStateRef.current) {
          pendingStateRef.current = stateString;
          
          if (vibeTimeoutRef.current) {
            clearTimeout(vibeTimeoutRef.current);
          }
          
          vibeTimeoutRef.current = setTimeout(async () => {
            if (stateString !== lastStateRef.current) {
              lastStateRef.current = stateString;
              
              const newVibe = await getVibeFromGemini(classesArray, currentEmotion, currentMovementState.activeGesture, currentHandState.activeHandAction, currentMovementState.kineticEnergy);
              lastPromptRef.current = newVibe;
              setCurrentPrompt(newVibe);
              
              if (sessionRef.current) {
                sessionRef.current.setWeightedPrompts({
                  weightedPrompts: [{ text: newVibe, weight: 1.0 }]
                }).catch(console.error);
              }
            }
          }, 2500);
        }
      } catch (err) {
        console.error("Detection loop error:", err);
      }
    }
    
    if (isPlayingRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;
    
    try {
      setErrorMsg(null);
      setStatus('Starting camera feed...');
      
      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 1280 }, 
              height: { ideal: 720 },
              facingMode: 'user'
            } 
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(e => console.error("Video play error:", e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error("Camera error:", camErr);
          setStatus('Camera Error');
          setErrorMsg('Camera access denied. Please grant camera permission in browser settings and refresh.');
          return;
        }
      }

      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);
      }

      setStatus('Connecting to Lyria AI Engine...');
      playerRef.current = new PCMPlayer(48000);

      let timeoutId: any;
      let sessionPromise;
      try {
        const ai = new GoogleGenAI({ 
          apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY || 'dummy-key',
          apiVersion: 'v1alpha'
        });

        sessionPromise = ai.live.music.connect({
          model: "lyria-realtime-exp",
          callbacks: {
            onmessage: (message: any) => {
              if (message.setupComplete) {
                console.log("Lyria setup complete");
              }
              const audioChunk = message.audioChunk;
              if (audioChunk?.data && playerRef.current) {
                playerRef.current.playChunk(audioChunk.data);
              }
            },
            onclose: () => {
              clearTimeout(timeoutId);
              if (isPlayingRef.current) {
                setInfoMsg('Lyria soundscape connection closed.');
                stopSession(false);
              } else {
                stopSession(false);
              }
            },
            onerror: (err: any) => {
              clearTimeout(timeoutId);
              console.error("Lyria API Error:", err);
              setErrorMsg(err.message || 'Connection error with Lyria API.');
              setInfoMsg(null);
              stopSession(false);
            }
          }
        });
      } catch (err: any) {
        console.warn("Failed to initialize Lyria API:", err);
        clearTimeout(timeoutId);
        setErrorMsg('Lyria API connection setup failed.');
        setInfoMsg(null);
        stopSession(false);
        return;
      }
      
      timeoutId = setTimeout(() => {
        setErrorMsg('Lyria API connection timed out.');
        setInfoMsg(null);
        stopSession(false);
      }, 30000);

      sessionPromise.then(async session => {
        clearTimeout(timeoutId);
        sessionRef.current = session;
        setStatus('Connected & Processing');
        setIsPlaying(true);
        
        const initialPrompt = "minimalist ambient drone, quiet";
        setCurrentPrompt(initialPrompt);
        lastPromptRef.current = initialPrompt;
        
        try {
          await session.setMusicGenerationConfig({
            musicGenerationConfig: { bpm: 120, temperature: 1.0 }
          });
          await session.setWeightedPrompts({
            weightedPrompts: [{ text: initialPrompt, weight: 1.0 }]
          });
          session.play();
        } catch (e) {
          console.error("Error setting up session:", e);
        }
      }).catch(err => {
        clearTimeout(timeoutId);
        console.error("API Error:", err);
        setErrorMsg(err.message || 'An unknown API error occurred.');
        setInfoMsg(null);
        stopSession(false);
      });

    } catch (err: any) {
      console.error("Setup Error:", err);
      setStatus('Failed to connect');
      setErrorMsg(err.message || 'An error occurred during system initialization.');
      setInfoMsg(null);
      stopSession(false);
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    setIsPlaying(false);
    if (vibeTimeoutRef.current) {
      clearTimeout(vibeTimeoutRef.current);
      vibeTimeoutRef.current = null;
    }
    if (geminiScanIntervalRef.current) {
      clearInterval(geminiScanIntervalRef.current);
      geminiScanIntervalRef.current = null;
    }
    geminiBoxesRef.current.clear();
    pendingStateRef.current = null;
    
    if (status === 'Connected & Processing' || status === 'Connecting to Lyria AI Engine...') {
      setStatus('Idle');
    }
    
    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }
    if (sessionRef.current) {
      try { sessionRef.current.conn.close(); } catch (e) {}
      sessionRef.current = null;
    }
    
    setConsoleState({
      emotion: 'neutral',
      objects: [],
      objectDetails: [],
      blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 },
      bodyMovement: {
        kineticEnergy: 0,
        leftHandSpeed: 0,
        rightHandSpeed: 0,
        armElevation: 0,
        activeGesture: 'STATIONARY / CALM POSTURE',
        bodyPoseDetected: false,
        leftHandRaised: false,
        rightHandRaised: false
      },
      handMovement: {
        handsCount: 0,
        leftHandGesture: 'NONE',
        rightHandGesture: 'NONE',
        leftPinchScore: 0,
        rightPinchScore: 0,
        handSpeed: 0,
        activeHandAction: 'NO HANDS DETECTED'
      }
    });
    smoothedBlendshapesRef.current = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
    smoothedMovementRef.current = { kineticEnergy: 0, leftHandSpeed: 0, rightHandSpeed: 0, armElevation: 0 };
    smoothedHandStateRef.current = { leftPinch: 0, rightPinch: 0, handSpeed: 0 };
    smoothedBoxesRef.current.clear();
    
    setInfoMsg(null);

    if (closeCamera) {
      isPlayingRef.current = false;
      setCurrentPrompt('Waiting for camera...');
      
      if (detectLoopRef.current) {
        cancelAnimationFrame(detectLoopRef.current);
        detectLoopRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setIsCameraActive(false);
      }
    }
  };

  return (
    <div className="h-[100dvh] w-full bg-black text-white flex overflow-hidden font-mono relative select-none">
      {/* Background Camera Feed */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/50 z-10 font-mono text-sm">
            <Camera className="w-8 h-8 mb-4 opacity-50" />
            <p className="tracking-widest">SYSTEM.CAMERA_OFFLINE</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover grayscale contrast-125 opacity-60 transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-500 z-[15] ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        {/* Vignette & Cyber Grid Overlay */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.85)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] z-10" />
        
        {/* Futuristic HUD Background Elements */}
        <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center overflow-hidden">
          <div className="w-[150vw] h-[150vw] sm:w-[600px] sm:h-[600px] border border-white/10 rounded-full border-dashed animate-[spin_60s_linear_infinite] shrink-0" />
          <div className="absolute w-[100vw] h-[100vw] sm:w-[400px] sm:h-[400px] border border-white/5 rounded-full animate-[spin_40s_linear_infinite_reverse] shrink-0" />
          <div className="absolute w-px h-full bg-white/5" />
          <div className="absolute h-px w-full bg-white/5" />
        </div>
      </div>

      {/* Main UI Overlay Container */}
      <div className="relative z-20 w-full h-full pointer-events-none p-3 sm:p-5 flex flex-col justify-between overflow-hidden">
        
        {/* Top Header Bar */}
        <div className="flex items-center justify-between w-full pointer-events-auto shrink-0 gap-4">
          
          {/* Top Left: Brand Title & Status */}
          <div className="flex items-center gap-3 bg-black/60 backdrop-blur-xl border border-white/20 px-3.5 py-2 shadow-[0_0_20px_rgba(0,0,0,0.8)]">
            <div>
              <h1 className="text-base sm:text-lg font-extrabold tracking-tighter text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.8)] flex items-center gap-2">
                DYNOVISION
              </h1>
              <p className="text-[8px] text-white/60 font-mono uppercase tracking-widest hidden sm:block">Object, Body & Hand Vision Engine</p>
            </div>

            <div className="h-6 w-px bg-white/20 mx-1" />

            <div className="text-[10px] font-mono text-white/90 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-none ${
                status.includes('Connected') || status.includes('Processing') 
                  ? 'bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)]' 
                  : status.includes('Connecting') || status.includes('Starting') 
                  ? 'bg-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.8)]' 
                  : status.includes('Loading') 
                  ? 'bg-blue-400 animate-pulse' 
                  : status.includes('Error') 
                  ? 'bg-red-500' 
                  : 'bg-zinc-600'
              }`} />
              <span className="hidden md:inline max-w-[180px] truncate">{status}</span>
            </div>
          </div>

          {/* Top Right Controls: Info & Toggle HUD Sidebars */}
          <div className="flex items-center gap-2">
            <button 
              onClick={() => { playHoverSound(); setShowHUDPanels(!showHUDPanels); }}
              onMouseEnter={playHoverSound}
              className={`p-2 sm:px-3 sm:py-2 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all backdrop-blur-xl border ${
                showHUDPanels 
                  ? 'bg-white text-black border-white shadow-[0_0_12px_rgba(255,255,255,0.5)]' 
                  : 'bg-black/60 text-white/70 border-white/20 hover:text-white'
              }`}
              title={showHUDPanels ? "Hide Overlay Side Panels" : "Show Overlay Side Panels"}
            >
              {showHUDPanels ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{showHUDPanels ? "HIDE PANELS" : "SHOW PANELS"}</span>
            </button>

            <button 
              onClick={() => { playHoverSound(); setIsInfoOpen(true); }}
              onMouseEnter={playHoverSound}
              className="p-2 sm:p-2.5 bg-black/60 hover:bg-white/20 rounded-none transition-colors backdrop-blur-xl border border-white/20 shrink-0 text-white"
              title="System Information"
            >
              <Info className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Collapsible Side Panels Grid Area (Leaves Center Unobstructed) */}
        <AnimatePresence>
          {showHUDPanels && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="flex-1 w-full flex flex-col md:flex-row justify-between pointer-events-none gap-4 my-3 overflow-y-auto overflow-x-hidden pr-1"
            >
              {/* Left Column HUD Panels */}
              <div className="w-full md:w-80 flex flex-col gap-3 pointer-events-auto shrink-0">
                
                {/* Secondary Biometric & Kinetic Visualizer Canvas */}
                <div className="bg-black/60 backdrop-blur-xl border border-white/20 p-3 w-full shadow-[0_0_30px_rgba(0,0,0,0.9)] flex flex-col h-56 shrink-0">
                  <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-white/10">
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => { playHoverSound(); setActiveTab('hands'); }}
                        className={`text-[9px] font-bold uppercase tracking-widest flex items-center gap-1 px-1.5 py-0.5 border transition-all ${
                          activeTab === 'hands' 
                            ? 'bg-white text-black border-white' 
                            : 'bg-transparent text-white/50 border-white/20 hover:text-white'
                        }`}
                      >
                        <Hand className="w-2.5 h-2.5" /> Hands
                      </button>
                      <button
                        onClick={() => { playHoverSound(); setActiveTab('pose'); }}
                        className={`text-[9px] font-bold uppercase tracking-widest flex items-center gap-1 px-1.5 py-0.5 border transition-all ${
                          activeTab === 'pose' 
                            ? 'bg-white text-black border-white' 
                            : 'bg-transparent text-white/50 border-white/20 hover:text-white'
                        }`}
                      >
                        <Accessibility className="w-2.5 h-2.5" /> Pose
                      </button>
                      <button
                        onClick={() => { playHoverSound(); setActiveTab('face'); }}
                        className={`text-[9px] font-bold uppercase tracking-widest flex items-center gap-1 px-1.5 py-0.5 border transition-all ${
                          activeTab === 'face' 
                            ? 'bg-white text-black border-white' 
                            : 'bg-transparent text-white/50 border-white/20 hover:text-white'
                        }`}
                      >
                        <ScanFace className="w-2.5 h-2.5" /> Face
                      </button>
                    </div>
                  </div>

                  <div className="relative w-full flex-1 border border-white/10 flex items-center justify-center bg-white/5 min-h-0 overflow-hidden">
                    <canvas
                      ref={handCanvasRef}
                      width={300}
                      height={300}
                      className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-300 ${
                        activeTab === 'hands' && isCameraActive ? 'opacity-100' : 'opacity-0'
                      }`}
                    />
                    <canvas
                      ref={poseCanvasRef}
                      width={300}
                      height={300}
                      className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-300 ${
                        activeTab === 'pose' && isCameraActive ? 'opacity-100' : 'opacity-0'
                      }`}
                    />
                    <canvas
                      ref={faceCanvasRef}
                      width={300}
                      height={300}
                      className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-300 ${
                        activeTab === 'face' && isCameraActive ? 'opacity-100' : 'opacity-0'
                      }`}
                    />
                    {!isCameraActive && (
                      <p className="text-[10px] text-white/40 uppercase tracking-widest">CAMERA INACTIVE</p>
                    )}
                  </div>
                </div>

                {/* Hand Movement & Gesture Metrics */}
                <div className="bg-black/60 backdrop-blur-xl border border-white/20 p-3 w-full shadow-[0_0_30px_rgba(0,0,0,0.9)]">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 flex items-center justify-between">
                    <span className="flex items-center gap-1.5"><Hand className="w-3 h-3 text-white" /> Hand Detection & Gestures</span>
                    <span className="px-1.5 py-0.5 text-[8px] font-bold bg-white/20 text-white font-mono">
                      {consoleState.handMovement.handsCount} HANDS
                    </span>
                  </h3>

                  <div className="mb-2 p-1.5 bg-white/5 border border-white/10 flex items-center justify-between">
                    <span className="text-[9px] text-white/60 uppercase">Action State</span>
                    <span className="text-[9.5px] font-bold text-white tracking-wider flex items-center gap-1">
                      <Zap className="w-3 h-3 text-white" />
                      {consoleState.handMovement.activeHandAction}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[9px] font-mono">
                    <div className="p-1.5 bg-white/5 border border-white/10">
                      <div className="text-white/50 text-[8px] uppercase">L-Hand Gesture</div>
                      <div className="font-bold text-white truncate mt-0.5">{consoleState.handMovement.leftHandGesture}</div>
                      <div className="text-[8px] text-white/60 mt-1">Pinch: {consoleState.handMovement.leftPinchScore}%</div>
                    </div>
                    <div className="p-1.5 bg-white/5 border border-white/10">
                      <div className="text-white/50 text-[8px] uppercase">R-Hand Gesture</div>
                      <div className="font-bold text-white truncate mt-0.5">{consoleState.handMovement.rightHandGesture}</div>
                      <div className="text-[8px] text-white/60 mt-1">Pinch: {consoleState.handMovement.rightPinchScore}%</div>
                    </div>
                  </div>
                </div>

                {/* Body Motion Metrics Panel */}
                <div className="bg-black/60 backdrop-blur-xl border border-white/20 p-3 w-full shadow-[0_0_30px_rgba(0,0,0,0.9)]">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 flex items-center justify-between">
                    <span className="flex items-center gap-1.5"><Move className="w-3 h-3" /> Body Motion Metrics</span>
                    <span className={`px-1.5 py-0.5 text-[8px] font-bold tracking-wider ${
                      consoleState.bodyMovement.bodyPoseDetected ? 'bg-white/20 text-white' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {consoleState.bodyMovement.bodyPoseDetected ? 'POSE OK' : 'NO POSE'}
                    </span>
                  </h3>

                  <div className="space-y-1.5 text-[9.5px]">
                    <div>
                      <div className="flex justify-between mb-0.5">
                        <span className="text-white/60 uppercase">Kinetic Motion Energy</span>
                        <span className="font-bold">{consoleState.bodyMovement.kineticEnergy}%</span>
                      </div>
                      <div className="h-1 bg-white/10 overflow-hidden">
                        <motion.div 
                          className="h-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]"
                          animate={{ width: `${consoleState.bodyMovement.kineticEnergy}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-white/60 uppercase">L-Speed</span>
                          <span className="font-bold">{consoleState.bodyMovement.leftHandSpeed}%</span>
                        </div>
                        <div className="h-1 bg-white/10 overflow-hidden">
                          <motion.div 
                            className="h-full bg-white/80"
                            animate={{ width: `${consoleState.bodyMovement.leftHandSpeed}%` }}
                            transition={{ duration: 0.3 }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-white/60 uppercase">R-Speed</span>
                          <span className="font-bold">{consoleState.bodyMovement.rightHandSpeed}%</span>
                        </div>
                        <div className="h-1 bg-white/10 overflow-hidden">
                          <motion.div 
                            className="h-full bg-white/80"
                            animate={{ width: `${consoleState.bodyMovement.rightHandSpeed}%` }}
                            transition={{ duration: 0.3 }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Right Column HUD Panels */}
              <div className="w-full md:w-80 flex flex-col gap-3 pointer-events-auto shrink-0 md:items-end">
                
                {/* Detected Objects & Entities Panel */}
                <div className="w-full bg-black/60 backdrop-blur-xl border border-white/20 p-3 shadow-[0_0_30px_rgba(0,0,0,0.9)]">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 flex items-center justify-between">
                    <span className="flex items-center gap-1.5"><Cpu className="w-3 h-3 text-white" /> Object Detection</span>
                    <div className="flex items-center gap-1">
                      <span className="text-[8px] bg-white/20 text-white font-mono px-1.5 py-0.5 flex items-center gap-1 font-bold">
                        <Sparkles className="w-2.5 h-2.5" /> AI + MobileNetV2
                      </span>
                      <span className="text-[9px] bg-white/10 px-2 py-0.5 font-mono text-white/80 font-bold">
                        {consoleState.objects.length}
                      </span>
                    </div>
                  </h3>

                  {consoleState.objectDetails.length === 0 ? (
                    <p className="text-[9.5px] text-white/40 italic py-1">No objects detected in frame.</p>
                  ) : (
                    <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
                      {consoleState.objectDetails.map((obj, i) => (
                        <div key={`${obj.class}-${i}`} className="p-1.5 bg-white/5 border border-white/10 flex items-center justify-between text-[9.5px]">
                          <div>
                            <div className="font-bold text-white uppercase tracking-wider">{obj.class}</div>
                            <div className="text-[7.5px] text-white/50">{obj.category}</div>
                          </div>
                          <div className="font-mono text-white/80 font-bold">
                            {Math.round(obj.score * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Affective Emotion State */}
                <div className="w-full bg-black/60 backdrop-blur-xl border border-white/20 p-3 shadow-[0_0_30px_rgba(0,0,0,0.9)]">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                    <Activity className="w-3 h-3" /> Facial Affective Emotion
                  </h3>
                  <div className="text-xl font-light tracking-tight capitalize text-white mb-2">
                    {consoleState.emotion}
                  </div>
                  <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[8.5px]">
                    <div>
                      <div className="flex justify-between text-white/60 mb-0.5">
                        <span>Smile</span>
                        <span>{(consoleState.blendshapes.smile * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1 bg-white/10 overflow-hidden">
                        <div className="h-full bg-white" style={{ width: `${consoleState.blendshapes.smile * 100}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-white/60 mb-0.5">
                        <span>Brow Raise</span>
                        <span>{(consoleState.blendshapes.browRaise * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1 bg-white/10 overflow-hidden">
                        <div className="h-full bg-white" style={{ width: `${consoleState.blendshapes.browRaise * 100}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Audio Vibe Profile */}
                <div className="w-full bg-black/60 backdrop-blur-xl border border-white/20 p-3 shadow-[0_0_30px_rgba(0,0,0,0.9)]">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                    <Music className="w-3 h-3" /> Audio Soundscape Vibe
                  </h3>
                  <div className="relative overflow-hidden pl-2 border-l-2 border-white">
                    <p className="text-[11px] text-white/90 leading-snug font-mono">
                      {currentPrompt}
                    </p>
                  </div>
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* BOTTOM FLOATING CONTROL DECK: Overlay Filters + Start/Stop System Button */}
        {/* Placed cleanly at the bottom center of the viewport so it never covers the screen */}
        <div className="w-full pointer-events-auto flex flex-col sm:flex-row items-center justify-center gap-3 shrink-0 pt-2 pb-1 z-30">
          
          {/* MOVED DOWN: Overlay Filters Bar */}
          <div className="flex items-center gap-1 bg-black/80 backdrop-blur-2xl border border-white/20 p-1.5 shadow-[0_0_30px_rgba(0,0,0,0.9)] overflow-x-auto max-w-full">
            <span className="text-[9px] text-white/50 uppercase tracking-widest px-2 font-mono flex items-center gap-1 shrink-0">
              <Layers className="w-3 h-3" /> Filters:
            </span>
            <div className="flex items-center gap-1 shrink-0">
              {[
                { id: 'all', label: 'ALL' },
                { id: 'objects', label: 'OBJECTS' },
                { id: 'pose', label: 'POSE' },
                { id: 'hands', label: 'HANDS' },
                { id: 'face', label: 'FACE' },
                { id: 'none', label: 'NONE' },
              ].map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => { playHoverSound(); setOverlayMode(mode.id as OverlayMode); }}
                  onMouseEnter={playHoverSound}
                  className={`px-2.5 py-1.5 text-[9.5px] font-bold uppercase transition-all border ${
                    overlayMode === mode.id 
                      ? 'bg-white text-black border-white shadow-[0_0_10px_rgba(255,255,255,0.7)] font-black' 
                      : 'bg-black/40 text-white/60 border-white/10 hover:text-white hover:border-white/30'
                  }`}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          {/* Main System Start / Stop Button */}
          <button
            onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
            onMouseEnter={playHoverSound}
            disabled={!isModelLoaded}
            className={`flex justify-center items-center gap-2 px-6 py-2.5 font-mono text-xs font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-2xl shrink-0 ${
              isCameraActive 
                ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]' 
                : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {!isModelLoaded ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> LOADING MODELS...</>
            ) : isCameraActive ? (
              <><Square className="w-3.5 h-3.5 fill-current" /> STOP SYSTEM</>
            ) : (
              <><Play className="w-3.5 h-3.5 fill-current" /> START SYSTEM</>
            )}
          </button>
        </div>

      </div>

      {/* Error Modal */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.2)] relative"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-500 uppercase tracking-widest">{status}</h3>
                  <p className="text-sm mt-2 text-red-400/80 leading-relaxed">{errorMsg}</p>
                </div>
              </div>
              
              <button 
                onClick={() => setErrorMsg(null)}
                className="w-full py-3 text-xs font-mono font-bold uppercase tracking-widest transition-colors bg-white/5 hover:bg-white/10 border border-white/20 text-white/80"
              >
                Dismiss
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* System Information Modal */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
            onClick={() => setIsInfoOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto"
            >
              <div className="flex items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Info className="w-5 h-5 shrink-0" />
                  DynoVision System Architecture
                </h2>
                <button 
                  onClick={() => setIsInfoOpen(false)}
                  className="p-2 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="space-y-4 text-xs text-white/80 leading-relaxed font-mono">
                <p>
                  <strong>DYNOVISION</strong> combines real-time object detection, body posture, hand gesture tracking, and facial biometrics to synthesize adaptive ambient music.
                </p>

                <div className="p-3 bg-white/5 border border-white/10 space-y-2">
                  <p className="font-bold text-white flex items-center gap-1.5">
                    <Hand className="w-3.5 h-3.5 text-white" /> Hand Movement & Gesture Detection (MediaPipe Hand)
                  </p>
                  <p className="text-white/70">
                    Detects 21 keypoints per hand to recognize pinch gestures, fists, pointing, open palms, and motion speed to modulate synth arpeggios.
                  </p>
                </div>
                
                <div className="p-3 bg-white/5 border border-white/10 space-y-2">
                  <p className="font-bold text-white flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5" /> Object Detection (COCO-SSD)
                  </p>
                  <p className="text-white/70">
                    Scans surroundings for visible entities (laptops, chairs, cups, fauna, vehicles) and maps them into musical themes.
                  </p>
                </div>

                <div className="p-3 bg-white/5 border border-white/10 space-y-2">
                  <p className="font-bold text-white flex items-center gap-1.5">
                    <Accessibility className="w-3.5 h-3.5" /> Body Movement Detection (MediaPipe Pose)
                  </p>
                  <p className="text-white/70">
                    Tracks 33 body keypoints in real time, calculating hand velocity, arm elevation, posture, and kinetic energy to dynamically modulate tempo.
                  </p>
                </div>

                <div className="p-3 bg-white/5 border border-white/10 space-y-2">
                  <p className="font-bold text-white flex items-center gap-1.5">
                    <ScanFace className="w-3.5 h-3.5" /> Facial Biometrics (MediaPipe Face)
                  </p>
                  <p className="text-white/70">
                    Extracts facial blendshapes to identify affective emotional states (happy, sad, surprised, neutral) for acoustic scale selection.
                  </p>
                </div>

                <p className="text-[10px] text-white/50 pt-2 border-t border-white/10">
                  All computer vision models process frame-by-frame locally in your browser with zero video data stored.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
