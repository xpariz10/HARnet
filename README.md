# Real-time audio signal dropout correction using a neural network

#### Oprava výpadků audio signálů v reálném čase pomocí neuronové sítě

This repository contains code for 
> PAŘÍZEK, Radim. Real-time audio signal dropout correction using a neural network. Semestral Thesis. Ondřej MOKRÝ (supervisor). Brno: Brno University of Technology, Faculty of Electrical Engineering and Communication, 2025.

## Porovnání implementovaných modelů

| Model                 | NMSE ↓ | Mel-SC ↓ | SDR[dB] ↑ | PEAQ ↑ | t predikce [ms] |
|-----------------------|--------|----------|-----------|--------|-----------------|
| AR\_TCN\_v2           | 0.172  |    0.286 |      7.66 |    -   |          336.21 | ggg
| AR\_TCN\_v1           | 0.172  |    0.286 |      7.66 |    -   |          336.21 |
| AR\_SSM\_v7\_4	    | 0.203  |    0.308 |      6.91 |    -   |          417.75 | ggg
| AR\_SSM\_v7\_2	    | 0.203  |    0.308 |      6.91 |    -   |          417.75 |
| AR\_SSM\_v7 	 		| 0.282  |    0.371 |      5.97 |    -   |          447.77 |
| AR\_SSM\_v6 	 		| 0.324  |    0.402 |      5.10 |    -   |          550.34 |
| AR\_SSM\_v5 	 		| 0.424  |    0.482 |      3.73 |    -   |                 |
| PARCnet IS2 			| 0.302  |    0.136 |       -   |  −1.42 |                 |
| PARCnet IS2 (trained) | 0.290  |    0.205 |      9.77 |    -   |                 |

Byly implementovány dva modely, **AR\_SSM s využitím SaShiMi/S4 bloků** pro residuum a **AR\_TCN s využitím TCN encoder-decoder architektury s rekurentním GRU bottle-neckem**.


## 🆕 AR\_SSM

### AR\_SSM\_v7
- byl implementován model kombinující AR predikci s residuální SSM sítí využívající S4 bloky
#### Features:
- vlastní upravená implementace S4 bloků v source kódu
- warm-up S4 matice
- ramp-up vážení S4D ve výsledné predikci na začátku restored burstu
- efektivní cross-fade po vzoru PARCnet

#### Limitace:
	- overhead inference S4 modelu v residuální funkci - <ins>vylučuje real-time</ins>
	- výrazná šumová složka v predikcích
![Prediction_spectrogram](/assets/images/AR_SSM_v7_spectrogram1.png)
	
#### Závěr:
	- aktuálně nastavené požadavky na real-time jsou nedosažitelné
	- kvůli krátkému dostupnému kontextu při inferenci ani nelze naplno využít potenciál S4 bloků
	✴️ Navrhuji prozkoumat možnost pomalého, ale malého přesného modelu pro rekostrukci poškozených souborů
	

### AR\_SSM\_v7_2
![AR_SSM_v7_visualisation_1](/assets/images/AR_SSM_v7_visualisation_1.png)

![AR_SSM_v7_visualisation_2](/assets/images/AR_SSM_v7_visualisation_2.png)

![AR_SSM_v7_visualisation_3](/assets/images/AR_SSM_v7_visualisation_3.png) 


### AR\_SSM\_v7_4











## 🆕 AR\_TCN




#### AR\_TCN\_v1

- byl implementován model kombinující AR predikci s residuální sítí kodér(dilatace)-GRU bottleneck-dekodér(dilatace)

Features:
- struktura obdobná PARCnetu, ConvNet NN vyměněna za lehčí TCN-GRU variantu o 188,737 parameterech (PARCnet IS2 cca 480,000 parametrů)
- dilatační bloky s fixním řádem vyměněny za progresivní řád [1, 2, 4, 8] - recepční pole 31 vzorků (~0.7 ms)
- skip-connections v kodéru i dekodéru

Limitace:
- 


Schéma:
  Ring Buffer (4096 samples = 8 packets of recent audio)
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────────────────────────┐
│   AR BRANCH      │              │   TCN/GRU NEURAL BRANCH                │
│   (classical)    │              │   188,737 parameters                   │
│                  │              │                                         │
│  Levinson-Durbin │              │  Input: past_audio / rms               │
│  LP order=256    │              │  (B, 1, 4864) = context + zeros        │
│  diagonal load   │              │                                         │
│  1e-3            │              │  ┌───────────────────────────────────┐  │
│                  │              │  │  input_proj   Conv1d(1→48, k=1)  │  │
│  Numba JIT       │              │  │  96 params                       │  │
│  ~0 trainable    │              │  └──────────────┬────────────────────┘  │
│  params          │              │                 │ (B, 48, 4864)         │
└────────┬─────────┘              │                 ▼                       │
         │                        │  ╔═══════════════════════════════════╗  │
         │                        │  ║        ENCODER (19,968 params)   ║  │
         │                        │  ║                                   ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=1      │──skip₁ │
         │                        │  ║  │ DW-Sep Conv + GLU + LN  │     ║  │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  │ RF: 3 samples           │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=2      │──skip₂ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  │ RF: +4 → 7 samples      │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=4      │──skip₃ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  │ RF: +8 → 15 samples     │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=8      │──skip₄ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  │ RF: +16 → 31 samples    │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ╚═══════════════│═══════════════════╝  │
         │                        │                  │ (B, 48, 4864)        │
         │                        │                  ▼ transpose            │
         │                        │  ╔═══════════════════════════════════╗  │
         │                        │  ║  GRU BOTTLENECK (139,392 params) ║  │
         │                        │  ║                                   ║  │
         │                        │  ║  GRU(in=48, hidden=192,          ║  │
         │                        │  ║      layers=1, bidir=False)      ║  │
         │                        │  ║                                   ║  │
         │                        │  ║  weight_ih: 48×192×3 = 27,648    ║  │
         │                        │  ║  weight_hh: 192×192×3 = 110,592  ║  │
         │                        │  ║  biases: 1,152                   ║  │
         │                        │  ║                                   ║  │
         │                        │  ║  Hidden state cached between     ║  │
         │                        │  ║  packets at inference for burst  ║  │
         │                        │  ║  loss recovery                   ║  │
         │                        │  ╚═══════════════╤═══════════════════╝  │
         │                        │                  │ (B, 4864, 192)       │
         │                        │                  ▼ transpose            │
         │                        │  ┌───────────────────────────────────┐  │
         │                        │  │  decoder_proj  Conv1d(192→48,k=1)│  │
         │                        │  │  9,264 params                    │  │
         │                        │  └──────────────┬────────────────────┘  │
         │                        │                 │ (B, 48, 4864)         │
         │                        │  ╔══════════════▼════════════════════╗  │
         │                        │  ║        DECODER (19,968 params)   ║  │
         │                        │  ║                                   ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=8      │◄─skip₄ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=4      │◄─skip₃ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=2      │◄─skip₂ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ║               ▼                  ║  │
         │                        │  ║  ┌─────────────────────────┐     ║  │
         │                        │  ║  │ GatedConvBlock d=1      │◄─skip₁ │
         │                        │  ║  │ 4,992 params            │     ║  │
         │                        │  ║  └────────────┬────────────┘     ║  │
         │                        │  ╚═══════════════│═══════════════════╝  │
         │                        │                  │ (B, 48, 4864)        │
         │                        │                  ▼                      │
         │                        │  ┌───────────────────────────────────┐  │
         │                        │  │  output_proj  Conv1d(48→1, k=1)  │  │
         │                        │  │  49 params                       │  │
         │                        │  └──────────────┬────────────────────┘  │
         │                        │                 │ (B, 1, 4864)          │
         │                        │                 ▼                       │
         │                        │          ┌────────────┐                 │
         │                        │          │    Tanh    │ bounds to [-1,1]│
         │                        │          └──────┬─────┘                 │
         │                        │                 ▼                       │
         │                        │     Slice last 768 samples             │
         │                        │                 │ (B, 1, 768)           │
         │                        │                 ▼                       │
         │                        │         × RMS  (denormalise)            │
         │                        └─────────────────┬───────────────────────┘
         │                                          │
         │            residual_pred (768 samples)    │
         │                                          │
         └──────────────── + ───────────────────────┘
                           │
                    hybrid_pred (768 samples)
                           │
                    ┌──────┴──────┐
                    │ :512  │512: │
                    └──┬────┴──┬──┘
                 output     crossfade
               (1 packet)   (overlap)






#### AR\_TCN\_v2

- rozšířené řády dilatace pro širší recepční pole [1, 2, 4, 8, 16, 32, 64] - 1017 vzorků 
