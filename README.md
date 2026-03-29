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
✴️ Navrhuji prozkoumat možnost pomalého, ale malého modelu pro inferenci s dlouhým kontextem
	

### AR\_SSM\_v7_2
![AR_SSM_v7_visualisation_1](/assets/images/AR_SSM_v7_visualisation_1.png)

### AR\_SSM\_v7_4
![AR_SSM_v7_4_visualisation_1](/assets/images/AR_SSM_v7_visualisation_1.png)



## 🆕 AR\_TCN

#### AR\_TCN\_v1
- byl implementován model kombinující AR predikci s residuální sítí kodér(dilatace)-GRU bottleneck-dekodér(dilatace)

Features:
- struktura obdobná PARCnetu, ConvNet NN vyměněna za lehčí TCN-GRU variantu o 188,737 parameterech (PARCnet IS2 cca 480,000 parametrů)
- dilatační bloky s fixním řádem vyměněny za progresivní řád [1, 2, 4, 8] - recepční pole 31 vzorků (~0.7 ms)
- skip-connections v kodéru i dekodéru

#### AR\_TCN\_v2
- rozšířené řády dilatace pro širší recepční pole [1, 2, 4, 8, 16, 32, 64] - 1017 vzorků

#### AR\_TCN\_S\_v1





