# Real-time audio signal dropout correction using a neural network

#### Oprava výpadků audio signálů v reálném čase pomocí neuronové sítě

This repository contains code for 
> PAŘÍZEK, Radim. Real-time audio signal dropout correction using a neural network. Semestral Thesis. Ondřej MOKRÝ (supervisor). Brno: Brno University of Technology, Faculty of Electrical Engineering and Communication, 2025.

## Porovnání implementovaných modelů

| Model                 | NMSE ↓ | Mel-SC ↓ | SDR[dB] ↑ | PEAQ ↑ | t predikce [ms] |
|-----------------------|--------|----------|-----------|--------|-----------------|
| AR\_TCN\_S\_v1        | 0.230  |    0.307 |      6.39 |    -   |           45.43 |
| AR\_TCN\_v2           | 0.201  |    0.302 |      6.97 |    -   |          447.18 |
| AR\_TCN\_v1           | 0.172  |    0.286 |      7.66 |    -   |          336.21 |
| AR\_SSM\_v7\_4	    | 0.216  |    0.315 |      6.67 |    -   |          357.77 |
| AR\_SSM\_v7\_2	    | 0.203  |    0.308 |      6.91 |    -   |          417.75 |
| AR\_SSM\_v7 	 		| 0.282  |    0.371 |      5.97 |    -   |          447.77 |
| AR\_SSM\_v6 	 		| 0.324  |    0.402 |      5.10 |    -   |          550.34 |
| AR\_SSM\_v5 	 		| 0.424  |    0.482 |      3.73 |    -   |                 |
| PARCnet IS2 			| 0.302  |    0.136 |       -   |  −1.42 |                 |
| PARCnet IS2 (trained) | 0.290  |    0.205 |      8.77 |    -   |                 |

Byly implementovány dva modely, **AR\_SSM s využitím SaShiMi/S4 bloků** pro residuum a **AR\_TCN s využitím TCN encoder-decoder architektury s rekurentním GRU bottle-neckem**.


## 🆕 AR\_SSM
- byl implementován model kombinující AR predikci s residuální SSM sítí využívající S4 bloky

#### Features:
- vlastní upravená implementace S4 bloků v source kódu
- warm-up S4 matice
- ramp-up vážení S4D ve výsledné predikci na začátku restored burstu
- efektivní crossfade po vzoru PARCnet

#### Architektura:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PLCInference (packet-by-packet engine)               │
│                                                                         │
│  Ring Buffer ◄──── received packets (pass-through)                      │
│  (4224 samples)    concealed packets (predicted)                        │
│       │                                                                 │
│       │ is_lost?                                                        │
│       ├── NO ──► outbound crossfade (16 samples) ──► return received    │
│       │                                                                 │
│       └── YES ──► _conceal_packet()                                     │
│                        │                                                │
│   ┌────────────────────┴─────────────────────┐                          │
│   │                                          │                          │
│   ▼                                          ▼                          │
│  AR Branch (CPU)                    S4D Branch (GPU/CPU)                │
│  ┌───────────────────┐              ┌────────────────────────┐          │
│  │ 1. Read context   │              │ 3a. Warm-up (burst     │          │
│  │    (4096 samples) │              │     start only):       │          │
│  │                   │              │                        │          │
│  │ 2. Fit LP coeffs  │              │  context/rms ──►       │          │
│  │    (Levinson-     │              │  ┌──────────────────┐  │          │
│  │     Durbin,       │              │  │ input_proj (1→64)│  │          │
│  │     order=256)    │              │  │ causal_conv (k=4)│  │          │
│  │                   │              │  │ SiLU             │  │          │
│  │ 3. Forward predict│              │  │ LayerNorm        │  │ BATCH    │
│  │    768 samples    │              │  └────────┬─────────┘  │          │
│  │    (recursive)    │              │           │ z          │          │
│  │                   │              │  ┌────────▼─────────┐  │          │
│  │ Fix 3: gain clamp │              │  │ S4D state loop   │  │ SEQUENTIAL
│  │ [0.02, 5.0]×     │               │  │ h = A*h + B*z[t] │  │ (4096    │
│  │                   │              │  │ (no output)      │  │  steps)  │
│  └────────┬──────────┘              │  └────────┬─────────┘  │          │
│           │                         │           │ state      │          │
│           │ ar_pred                 │                        │          │
│           │ (768 samples)           │ 3b. Gap prediction:    │          │
│           │                         │                        │          │
│           │                         │  bias ──►              │          │
│           │                         │  ┌──────────────────┐  │          │
│           │                         │  │ input_proj (bias)│  │          │
│           │                         │  │ causal_conv      │  │ BATCH    │
│           │                         │  │ SiLU + LayerNorm │  │          │
│           │                         │  └────────┬─────────┘  │          │
│           │                         │           │ z           │          │
│           │                         │  ┌────────▼─────────┐  │          │
│           │                         │  │ S4D state loop   │  │ SEQUENTIAL
│           │                         │  │ h = A*h + B*z[t] │  │ (512     │
│           │                         │  │ out = 2Re(C*h)   │  │  steps)  │
│           │                         │  └────────┬─────────┘  │          │
│           │                         │           │             │          │
│           │                         │  ┌────────▼─────────┐  │          │
│           │                         │  │ Gate × S4D out   │  │ BATCH    │
│           │                         │  │ output_proj      │  │          │
│           │                         │  │ Tanh [-1, 1]     │  │          │
│           │                         │  │ × RMS (denorm)   │  │          │
│           │                         │  └────────┬─────────┘  │          │
│           │                         │           │             │          │
│           │                         └───────────┼────────────┘          │
│           │                                     │ s4d_correction        │
│           │                                     │ (512 samples)         │
│           ▼                                     ▼                       │
│   ┌───────────────────────────────────────────────┐                    │
│   │  concealed = ar_pred[:512] + s4d_correction   │                    │
│   └───────────────────┬───────────────────────────┘                    │
│                       │                                                 │
│                       ▼                                                 │
│   ┌─── Post-processing chain ────────────────────┐                    │
│   │                                               │                    │
│   │  Fix C: AR envelope compensation              │                    │
│   │  ├─ Compare 1st-quarter vs 4th-quarter RMS    │                    │
│   │  └─ If decay > 30%: linear gain ramp (cap 3×) │                    │
│   │                                               │                    │
│   │  Fix 4: Inbound crossfade                     │                    │
│   │  ├─ Burst start:  8 samples (cosine)          │                    │
│   │  └─ Burst cont: 256 samples (cosine)          │                    │
│   │                                               │                    │
│   │  Store _prior_overlap (tail for outbound)     │                    │
│   │                                               │                    │
│   │  Fix 6: Post-prediction gain matching         │                    │
│   │  └─ If output RMS outside [0.7, 1.4]×context  │                    │
│   │     → scale to match                          │                    │
│   │                                               │                    │
│   └───────────────────┬───────────────────────────┘                    │
│                       │                                                 │
│                       ▼                                                 │
│              Write to ring buffer                                      │
│              Return concealed (512 samples)                            │
└─────────────────────────────────────────────────────────────────────────┘
S4ResidualBlock internal pipeline:


(B, 1, L) input
    │
    ▼
Conv1d(1→64, k=1)          ─── input_proj
    │
    ▼
Conv1d(64→64, k=4, groups=64, left-pad=3)  ─── causal depthwise conv
    │
    ▼
SiLU ──────────────────────────────────────────► gate_proj(Linear 64→64)
    │                                                      │
    ▼                                                      ▼
LayerNorm(64)                                         SiLU(gate)
    │                                                      │
    ▼                                                      │
S4DLayer(d=64, N=64)                                      │
    │  h[t+1] = A_bar ⊙ h[t] + B_bar * u[t]              │
    │  y[t]   = 2·Re(C · h[t]) + D · u[t]                 │
    │                                                      │
    ▼                                                      │
    ×  ◄───────────────────────────────────────────────────┘
    │           element-wise gating
    ▼
Conv1d(64→1, k=1)          ─── output_proj
    │
    ▼
Tanh()                      ─── bound to [-1, 1]
    │
    ▼
Slice last pred_dim (768) samples
    │
    ▼
(B, 1, 768) output

```
#### Limitace:
- overhead inference S4 modelu v residuální funkci - <ins>vylučuje real-time</ins>
  - způsoben limitacemi PyTorch - každá iterace spouští 2-3 CUDA kernely - možnost implementace fúzovaného CUDA kernelu po vzoru Mamba
- výrazná šumová složka v predikcích
- chybný crossfade na konci doplněného burstu
- potřeba optimalizovat gain compensation opatření
![Prediction_spectrogram](/assets/images/AR_SSM_v7_spectrogram1.png)
	
#### Závěr:
- aktuálně nastavené požadavky na real-time jsou nedosažitelné
- kvůli krátkému dostupnému kontextu při inferenci ani nelze naplno využít potenciál S4 bloků
✴️ Navrhuji prozkoumat možnost pomalého, ale malého modelu pro inferenci s dlouhým kontextem
	

### AR\_SSM\_v7_2
![AR_SSM_v7_visualisation_1](/assets/images/AR_SSM_v7_visualisation_1.png)

### AR\_SSM\_v7_4
![AR_SSM_v7_4_visualisation_1](/assets/images/AR_SSM_v7_visualisation_1.png)



## 🆕 AR\_TCN
- byl implementován model kombinující AR predikci s residuální sítí kodér(dilatace)-GRU bottleneck-dekodér(dilatace)

#### Features:
- dilatační bloky s fixním řádem vyměněny za progresivní řád [1, 2, 4, 8] - recepční pole zachycující celý kontext
- skip-connections mezi kodérem i dekodérem
- frekvenční projekce - vylepšení oproti BS-PLCnet
- úsporná varianta GRU(128,128) pro bottleneck
- 630K parametrů
- band-split - rozdělění na pásmo 0-8kHz a 8kHz-20kHz
- vyšší pásmo zpracováno velmi úspornou GRU s průměrováním

#### Limitace:
- chybný crossfade na konci doplněného burstu
- potřeba optimalizovat gain compensation opatření

#### Další postup:
- F-FiLM podmiňování - určení základních tónů při trénování (BS-PLCnet)
- cross-band podmiňování - výstup GRU v high-band podmíněn výstupem low-band GRU
- ONNX Runtime (časová úspora 3-5x)

  
Struktura celé architektury
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     HYBRID AR + TCN/GRU PLC SYSTEM                        ║
║          44.1 kHz · 512 samples/packet · strictly causal                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

   Ring Buffer (4096 samples = 8 packets of recent audio)
        │                                    │
        ▼                                    ▼
┌───────────────────┐              ┌──────────────────────────────────────┐
│   AR BRANCH       │              │   TCN/GRU NEURAL BRANCH              │
│   (classical)     │              │   (STFT-domain, 630K params)         │
│                   │              │                                      │
│  Levinson-Durbin  │              │  Input: past_audio / rms             │
│  LP order=256     │              │  (B, 1, 4864) = context + zeros      │
│  Numba JIT        │              │                                      │
│  ~0 trainable     │              │  ┌──────────────────────────────┐    │
│                   │              │  │ STFT → 2D processing → iSTFT │    │
│  Predicts tonal   │              │  │ (see detail below)           │    │
│  continuity       │              │  └──────────────┬───────────────┘    │
└────────┬──────────┘              │                 │                    │
         │                         │          × RMS  (denormalise)        │
         │  ar_pred (768 samples)  └─────────────────┬────────────────────┘
         │                                           │
         │               residual_pred (768 samples) │
         │                                           │
         └──────────────────── + ────────────────────┘
                                │
                         hybrid_pred (768)
                                │
                    ┌───────────┴────────────────┐
                    │ output: 512 │ overlap: 256 │
                    │ (1 packet)  │ (fade-out)   │
                    └─────────────┴──────────────┘
```

Struktura residuální NN části
```
Input: (B, 1, 4864) normalised residual
  │
  ▼
╔═══════════════════════════════════════════════════════════╗
║  STFT FRONTEND                                            ║
║  left-pad 768 zeros → (B, 1, 5632) [causal guarantee]     ║
║  torch.stft(n_fft=1024, hop=256, center=False)            ║
║  → (B, 513, 19) complex → stack real/imag channels        ║
║  → (B, 2, 513, 19)                                        ║
╚══════════════════════╤════════════════════════════════════╝
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
  ┌──────────────────┐   ┌──────────────────┐
  │ LOW BAND         │   │ HIGH BAND        │
  │ bins 0–127       │   │ bins 128–512     │
  │ 0–5.5 kHz        │   │ 5.5–22 kHz       │
  │ (B, 2, 128, 19)  │   │ (B, 2, 385, 19)  │
  │                  │   │                  │
  │ Harmonics,       │   │ Overtones,       │
  │ fundamentals,    │   │ transients,      │
  │ tonal content    │   │ noise-like       │
  └────────┬─────────┘   └────────┬─────────┘
           │                      │
           ▼                      ▼
     FULL ENCODER            LIGHTWEIGHT
     + GRU + DECODER         GRU PROCESSOR
     (see below)             (see below)
           │                      │
           ▼                      ▼
  (B, 2, 128, 19)        (B, 2, 385, 19)
  predicted low           predicted high
           │                      │
           └──────────┬───────────┘
                      ▼
╔═══════════════════════════════════════════════════════════╗
║  iSTFT BACKEND                                            ║
║  merge bands → (B, 2, 513, 19)                            ║
║  reconstruct complex → manual overlap-add iSTFT           ║
║  trim left-pad → (B, 1, 4864)                             ║
║  Tanh → slice last 768 → (B, 1, 768)                      ║
╚═══════════════════════════════════════════════════════════╝
```

Struktura Low-Band zpracování (kodér-dekodér)
```
(B, 2, 128, 19)
  │
  ▼ input_proj: Conv2d(2→48, 1×1)              144 params
  │
  ├── GatedConvBlock2d d=1 ──── skip₁          5,280 params
  ├── GatedConvBlock2d d=2 ──── skip₂          5,280 params
  ├── GatedConvBlock2d d=4 ──── skip₃          5,280 params
  └── GatedConvBlock2d d=8 ──── skip₄          5,280 params
  │
  ▼ FreqProjection                             206,000 params
  │  Conv1d(48,48,k=4,s=4): 128→32 freq bins
  │  Linear(48×32=1536, 128)
  │  → (B, 19, 128)
  │
  ▼ GRU(128, 128, bidir=False)                 99,072 params
  │  Only 19 sequential steps (~0.15ms)
  │  → (B, 19, 128)
  │
  ▼ FreqUnprojection                           207,408 params
  │  Linear(128, 1536)
  │  ConvTranspose1d(48,48,k=4,s=4): 32→128 bins
  │  → (B, 48, 128, 19)
  │
  ├── GatedConvBlock2d d=8 ◄── skip₄          5,280 params
  ├── GatedConvBlock2d d=4 ◄── skip₃          5,280 params
  ├── GatedConvBlock2d d=2 ◄── skip₂          5,280 params
  └── GatedConvBlock2d d=1 ◄── skip₁          5,280 params
  │
  ▼ output_proj: Conv2d(48→2, 1×1)            98 params
  │
  → (B, 2, 128, 19) predicted low-band STFT
```

