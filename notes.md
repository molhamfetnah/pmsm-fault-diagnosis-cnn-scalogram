- BMDC
- PMAC
- PMSM : permemtnet magnet syncronuise motor

- ESC

- Electrical Power = Mechantical Power
- v*I = T*Omiga

- v->omiga
- I->Torque

- PI -> currunt control -> each single phase of the three -> to control torque 
- PWM -> Voltage Control -> H-bridge -> 3ph -> 3 half bridge -> three cruunt sensors one on each phase

- FOC: field oriented controller : it contorls eaah the currunt that neads to geenrate torque by pi loop that generates 3ph pwm & the currunt that not supoed to generate turque the same way -> needs encoder

BLDC vs PMSM
Trepusiadal signal | sinosidal signal

sinosidal -> SPWM -> voltage -> genrtaes sinosdial cruurnt 


---

Fourier transfrom vs Wavelet transfrom 
Fourier: blind to time only sees frequensy
Uncertinatiy Pricnle : deltaF* deltaT >= 1
so we cant represnt deffirecnce over time and frequancy at the same time
Amblitude ~ Freqauncy
Amblitude ~ Time
-> Wavelete -> is a short lived oscilation localized in time  ((litle wave in french)) -> special analysis function that replaces the sin in the transformation process
- Daubenchies
- coiflet
- symlet
- haar
- morlet
- guassian
- shannon
- meyer
- maxican hat
1. it have to have zero mean : inegral over infinity = 0 - admissibility condition - the zero frequancy compomnet abplitude should be zero 
2. it have to have finite energy : the ingral of the space of the absoulte value of the function should be finit (less than enfinity) -> localized in time at the abosot of sin which is inifit over time  
- morlet wavelete in simple terms = k0*cos(omega*t) * e^((-t^2)/2) -> a cosin dummbed (multiplied guaaisna beel curve)

frourier is time domain to frequancy domain  -> one demintion in and out
wavelet -> two dimtions output -> y(t) = T(t,f)
-> mother wavelete have two childs
1. time knobe -> the waveltete moved over the time access -> yb = y(t-b)
2. frequancy knob -> the wavelte is scaled over the time access -> ya=y(t/a)
- shrink -> bigger freqnacy
- strech -> smaller frequancy
=> ya,b= y((t-b)/a) -> the value of T(a,b) is equal to the conribtution of ya,b to comprising the signal -> T(a,b) represents how well our doughter wavelet ya,b matches the sinal
- multipling the wavelet to the signal will tell us how much they are in the same sign and how much they are in the opposit sign -> integrating that will subtract those two values and gets T(a,b) the contribution of a duaghter walet to the signal -> local similarity -> (integraying dot product)
- the dot product is a misuremnt of similarity for victors 
-  intagrating the multiplication is exactly clulting the dot product of two inifit dimtnion virctors (where each dicmetnion value reposrdn the value on y for each x the number of dicmtions is the samlling time of the signal) -> so we area calutuing the similarty of the frequancy for specif wavelet
- over sliding time frame it is named convolution = but it can give zero at peak and zero -> so we use the envelop
- we reaplce victors with complex numbers -> eurlar formula
=> y(t) = k.e^(i*omega_zero*t).e^(-t^2/2)
-> wavelte is the real cpmompnt of the imaginray number rotating -> which is cosine
-> the basoulout value is the sinsntace from the origin is the power of that specif frequency -> 
- color -> wavelet scalogram -> repsen the power of ach freaquncy over time 
- the blur in scalogram is the trade off of the uncertiatnty prencible -> represnted heisenberg boxes -> for low frequencies is accurate for frequancy and not acuurate at time on the other hand high freqncy boxs is not acurate on the freqancy acess but accurate in the time access 




---

A Convolutional Neural Network (CNN or ConvNet) is a specialized type of deep learning algorithm primarily designed to process, analyze, and classify visual data such as images and videos. Inspired by the biological structure of the human visual cortex, CNNs excel at automatically identifying spatial patterns like edges, textures, shapes, and full objects. [1, 2, 3] 
------------------------------
## How a CNN Works
To a computer, an image is not a picture; it is a grid of numbers representing pixel color intensities. A CNN processes this grid by passing it through a structured sequence of layers: [3, 4, 5] 

[Input Image] ➔ [Convolution Layer] ➔ [Activation (ReLU)] ➔ [Pooling Layer] ➔ [Flattening] ➔ [Fully Connected Layer] ➔ [Output/Prediction]


* 
* 1. Convolutional Layer: This is the core building block. Small mathematical matrices called filters (or kernels) slide (convolve) across the pixels of the image. At each step, they multiply filter values with image pixel values to create a Feature Map. This extracts crucial features like horizontal lines, vertical edges, or color contrasts. [6, 7] 
* 2. Activation Layer (ReLU): The network applies the Rectified Linear Unit (ReLU) function. It converts all negative pixel values in the feature map to zero. This introduces non-linearity, allowing the AI to learn complex, non-linear real-world patterns. [3, 7] 
* 3. Pooling Layer: This layer downsamples and reduces the spatial dimensions (width and height) of the feature maps. The most common method, Max Pooling, extracts only the highest value from a small section of the grid. This retains the most vital information while reducing computational load and memory usage. [3, 7] 
* 4. Flattening: Once the network extracts the features via several convolution and pooling stages, the final 2D feature maps are flattened into a single, long 1D vector (a list of numbers). [3, 7] 
* 5. Fully Connected (Dense) Layer: This acts like a traditional neural network. It takes the flattened 1D vector of features and performs high-level reasoning to make a final prediction (e.g., deciding whether the input image is a cat or a dog). [3, 7, 8] 
* 

------------------------------
## Why Use a CNN?
Before CNNs, traditional artificial neural networks (ANNs) struggled with images for two main reasons: [9] 

   1. Mathematical Explosion: A standard network connects every single pixel to every single neuron. A high-resolution image creates millions of connections, making training computationally impossible. [1, 9, 10] 
   2. Loss of Spatial Context: Traditional networks require flattening an image into a single row of numbers before processing. This completely destroys the spatial relationships between neighboring pixels (like how eyes sit next to a nose). [5, 10] 

CNNs fix this through parameter sharing (using the same filter across the entire image) and spatial invariance (the ability to recognize an object no matter where it appears in the frame). [2, 9, 10] 
------------------------------
## Pros and Cons## Pros

* 
* Automated Feature Extraction: You do not need to manually program the AI to look for edges or shapes. The network learns what features are important on its own during training.
* High Accuracy in Vision Tasks: They achieve state-of-the-art accuracy in complex computer vision tasks like facial recognition, self-driving car navigation, and medical image diagnostics.
* Parameter Efficiency: Because the same filters slide across the entire image, CNNs require significantly fewer parameters than fully connected networks, preventing massive computational waste. [7, 8, 9, 10, 11, 12] 
* 

## Cons

* 
* Data and Power Hungry: CNNs require massive, high-quality labeled datasets (often millions of images) and immense computing power (GPUs) to train effectively from scratch.
* No Inherent Understanding of Rotation or Scale: If a CNN is trained on upright images of cats, it may fail to recognize a cat that is upside down or heavily zoomed in unless it is explicitly trained with those variations or heavily augmented.
* The "Black Box" Problem: It can be incredibly difficult to interpret exactly why a deep CNN made a specific decision, which poses challenges in high-stakes fields like medicine or legal applications. [9, 13, 14, 15, 16] 
* 

---