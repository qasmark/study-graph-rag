# LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics

#### Jialiang Tang 1 2 Shuo Chen 3 Chen Gong 4 Jing Zhang 5 Dacheng Tao 2

### Abstract

# arXiv:2503.09656v1 [cs.LG] 12 Mar 2025

Time Series Forecasting (TSF) is critical in many real-world domains like financial planning and health monitoring.
Recent studies have revealed that Large Language Models (LLMs), with their powerful in-contextual modeling capabilities,
hold significant potential for TSF. However, existing LLM-based methods usually perform suboptimally because they
neglect the inherent characteristics of time series data. Unlike the textual data used in LLM pre-training, the time
series data is semantically sparse and comprises distinctive temporal patterns. To address this problem, we propose LLM-
PS to empower the LLM for TSF by learning the fundamental Patterns and meaningful Semantics from time series data. Our
LLM-PS incorporates a new multi-scale convolutional neural network adept at capturing both short-term fluctuations and
long-term trends within the time series. Meanwhile, we introduce a time-to-text module for extracting valuable semantics
across continuous time intervals rather than isolated time points. By integrating these patterns and semantics, LLM-PS
effectively models temporal dependencies, enabling a deep comprehension of time series and delivering accurate
forecasts. Intensive experimental results demonstrate that LLM-PS achieves state-of-the-art performance in both short-
and long-term forecasting tasks, as well as in few- and zero-shot settings.

Figure 1: Performance of our proposed LLM-PS, LLMbased methods (Liu et al., 2024a; Jin et al., 2024), and conventional
deep learning methods (Wu et al., 2023a; Zeng et al., 2023; Liu et al., 2024b).

### 1. Introduction

Time Series Forecasting (TSF) plays a crucial role in various real-world applications, such as weather forecasting
(Angryk et al., 2020) and energy consumption prediction (Trindade, 2015). To achieve reliable TSF, traditional deep
learning methods (Wu et al., 2023b; Zhang & Yan, 2023b; Zhou et al., 2022) usually rely on domain-specific expertise to
design customized models tailored to individual tasks. However, time series data often vary significantly across
different domains (Zhang et al., 2024a). For example, stock prices in the financial market frequently fluctuate in short
intervals, while temperature readings from the weather station typically evolve gradually over days (Hyndman, 2018), as
shown in Fig. 5. Therefore, those taskspecific models struggle to generalize effectively across different applications
(Jin et al., 2024). Moreover, these models are commonly trained from scratch and prone to

1School of Computer Science and Engineering, Nanjing University of Science and Technology, China 2College of Computing
and Data Science, Nanyang Technological University, Singapore 3School of Intellgence Science and Technology, Nanjing
University, China 4Department of Automation, Institute of Image ProceProceedingssing and Pattern Recognition, Shanghai
Jiao Tong University, China 5School of Computer Science, Wuhan University, China. Correspondence to: Chen Gong
<chen.gong@sjtu.edu.cn>, Dacheng Tao <dacheng.tao@ntu.edu.sg>.

1

Title Suppressed Due to Excessive Size

the wavelet transform and enhance them via global-to-local and local-to-global assembling. Meanwhile, T2T extracts
valuable semantics by precisely predicting labels for timeseries patches under a high masking ratio. Subsequently, the
temporal patterns and semantics are integrated through feature transfer and input into the LLM to generate time series.
Thanks to its ability to effectively handle diverse temporal patterns and accurately extract semantically rich
information, LLM-PS comprehensively understands time series data, consistently achieving state-of-the-art (SOTA)
performance across multiple datasets, as shown in Fig. 1.

overfitting in practical scenarios due to limited training data availability (Wen et al., 2023).

Recently, Large Language Models (LLMs), such as GPT (Brown et al., 2020) and Llama (Touvron et al., 2023), have achieved
great success in natural language processing. These LLMs are pre-trained on large-scale serialized textual datasets,
endowing them with strong capabilities in both contextual modeling and generalization. In general, time series and
textual data share two primary similarities: 1) Sequentiality, both time series and textual data consist of ordered sets
of elements; 2) Contextual dependency, the meaning of a textual sentence relies on its context, and the value at the
current time point is driven by its historical data. Building on these parallels, pioneering works (Zhou et al., 2023a;
Liu et al., 2024a; Jin et al., 2024; Cao et al.) attempt to fine-tune powerful LLMs for time series generation. Among
them, Liu et al. (Liu et al., 2024a) align the distributions of time series and textual data to enhance the LLM’s
effectiveness in TSF. TimeLLM (Jin et al., 2024) bridges the modalities of time series and textual data by reprogramming
time series data with text prototypes, thereby unlocking the TSF performance of LLMs.

• We recognize that time series data exhibits intrinsic characteristics that differ from those in textual data used in
LLM pre-training and propose a novel TSF framework to leverage these distinctive properties to derive the LLM for
reliable time series forecasting.

• We design new MSCNN and T2T modules tailored for effectively handling the diverse temporal patterns and accurately
extracting semantic information, and thus enhancing the LLM’s understanding of the input time series during forecasting.

Despite significant advancements in existing LLM-based methods for TSF, they generally prioritize aligning textual and
time series data while ignoring the inherent characteristics of time series data, resulting in suboptimal performance.
First, time series data exhibit diverse temporal patterns rarely present in textual data, including regular short-term
periodic fluctuations and persistent long-term trends that evolve over time (Wu et al., 2021; Zhou et al., 2022).
Second, the semantic information is spare in time series data (Cheng et al., 2023), usually requiring a prolonged period
to convey specific semantics like “rapid increase” or “sudden drop”. In comparison, words in textual data generally
express explicit meaning, such as “fast” or “slow”. Therefore, identifying fundamental temporal patterns and specific
semantic information within time series data is crucial to guide LLMs for reliable time series prediction.

• Our LLM-PS consistently achieves SOTA performance across a variety of mainstream time series prediction tasks,
especially in few-shot and zero-shot scenarios. Furthermore, our model is highly efficient while robust to noise
compared with other popular methods.

### 2. Related Works

#### 2.1. Time Series Forecasting

Time series forecasting aims to predict future values of a series based on historical data, serving as a crucial
capability in industries like finance management (Patton, 2013), weather forecasting (Angryk et al., 2020), and energy
consumption prediction (Zhou et al., 2021a). Recently, TSF methods have evolved from traditional statistical models
(Taylor & Letham, 2018; Oreshkin et al., 2019a) to sophisticated deep learning models (Wu et al., 2023b; Wang et al.,
2024). Deep learning methods generally utilize Recurrent Neural Networks (RNNs) (Salinas et al., 2020), Convolutional
Neural Networks (CNNs) (Wu et al., 2023b), Transformers (Wu et al., 2021), and Multi-Layer Perceptrons (MLPs) (Wang et
al., 2024) as their backbones. By leveraging domain expertise, they can perform well on specific tasks. Nonetheless,
their real-world applicability is usually constrained by the variability of temporal patterns across domains (Jin et
al., 2024).

In this paper, we propose a novel LLM fine-tuning framework called LLM-PS, which enhances time series forecasting by
leveraging the Patterns and Semantics in time series data. LLM-PS comprises two pivotal modules: the MultiScale
Convolutional Neural Network (MSCNN), designed to capture the intrinsic temporal patterns, and the Timeto-Text semantic
information extractor (T2T). Specifically, MSCNN extracts multi-scale features with varying receptive fields by
hierarchical stacked convolutional layers. Features with small receptive fields primarily capture short-term patterns
(i.e., periodicity fluctuations), while those with large receptive fields focus on long-term patterns (i.e., global
trends). To further cope with short-term and long-term patterns, we decouple them from multi-scale features based on

To address these challenges, some researchers (Zhou et al., 2023a; Liu et al., 2024a) have turned to LLMs for TSF and
achieved great success. Current methods primarily focus on bridging the gap between time series and textual

2

Title Suppressed Due to Excessive Size

pooling in the temporal domain or Fourier transform in the frequency domain, our method concurrently learns from both
the temporal and frequency domains. Therefore, our approach can accurately decompose short-term and longterm components,
as visualized in Fig. 6.

modalities. For example, PromptCast (Xue & Salim, 2023) encodes time series and textual data into prompts to guide the
LLM prediction. The prompts contain contextual information, task requirements, and the desired output format. TimeLLM
(Jin et al., 2024) further enhances guidance for LLM by incorporating domain information, instructions, and data
statistics within the prompts. Instead of designing prompts, (Chung et al., 2023) directly train a codebook to convert
continuous time series into discrete input embeddings by mapping them to the most similar codewords in the codebook.
Similarly, (Rubenstein et al., 2023) apply K-Means clustering to time series embeddings and construct a codebook (Van
Den Oord et al., 2017) by the cluster centroids. Additionally, CALF (Liu et al., 2024a) trains separate LLM branches for
time series and textual data, which aligns their features across the intermediate and output layers. LLM-TS (Chen et
al., 2024) employs a CNN as the time series branch and guides it with the LLM by minimizing their mutual information.

### 3. Approach

Time series forecasting aims to predict the series Y ∈ RT ×V for the next T time steps given the H time steps historical
observations X ∈ RH×V , where V denotes the number of variables. As shown in Fig. 2, our LLM-PS incorporates a new MSCNN
(detailed in Sections 3.1&3.2) to learn multi-scale features from the input time series, which captures both short-term
and long-term patterns. Meanwhile, within LLM-PS, the T2T module (described in Section 3.3) enriches the multi-scale
features by the semantic information extracted from the input time series. Consequently, the multi-scale features with
diverse temporal dependencies and valuable semantics are input into the LLM to facilitate accurate future time series ˆY
∈ RT ×V .

Despite the promising performance of LLM-based methods, they inadequately solve the intrinsic characteristics of time
series data, limiting their effectiveness for TSF. Our LLMPS effectively handles these properties of temporal patterns
and semantics, thereby achieving reliable performance.

#### 3.1. Multi-Scale Convolutional Neural Network

Real-world time series data is inherently complex, manifesting short-term and long-term patterns (Cleveland et al.,
1990; Zhang & Qi, 2005), both of which are critical for accurate forecasting. Short-term patterns reflect localized
fluctuations and periodic dynamics, while long-term patterns encapsulate broader trends that signal future trajectories.
In conventional CNNs, each convolutional layer has a fixed receptive field, which limits its output features to a narrow
temporal scope with only a single temporal pattern. To capture diverse temporal patterns, we follow classic CNNs (He et
al., 2016; Gao et al., 2019) and design a new MSCNN stacked with bottleneck blocks. As depicted in Fig. 3, each MSCNN
block learns multi-scale features by parallel branches. Features with smaller receptive fields focus on periodic
fluctuations, whereas those with larger receptive fields concentrate on overarching trends.

#### 2.2. Temporal Patterns Learning

Time series data comprises short-term fluctuations and longterm trends (Zhang & Qi, 2005). To capture these temporal
patterns, existing methods (Wang et al., 2024; Kowsher et al., 2024) convert the original inputs to multiple time series
inputs with varying scales through pooling operations with various window sizes. Consequently, the model can learn
short-term periods and long-term trends from low-scale and high-scale inputs, respectively. However, these methods incur
prohibitive computational overheads due to the intricate processing of numerous scaled signals. To address this, we
introduce a new MSCNN that enables efficiently generating multi-scale features with diverse temporal patterns using a
single forward, as analyzed in Section 4.4.

In each MSCNN block, the C channels input features Fin ∈ RC×V first undergo 1×1 convolutional layer before partitioned
to B branches {F1, . . . , FB}, where Fi ∈ RC/B×V

Besides to construct multi-scale signals, some methods (Wu et al., 2021; 2023c) directly separate temporal patterns from
time series. Earlier studies (Wu et al., 2021; 2023c) employ average pooling to extract long-term trends from time
series data while considering the remaining segments as shortterm patterns. Recent studies (Zhou et al., 2022; Wang et
al., 2024) highlighted that low-frequency and high-frequency components in the frequency domain correspond to longterm
and short-term patterns, respectively. Therefore, they utilize the Fourier transform (Cochran et al., 1967) to isolate
them. In this paper, we decouple short-term and longterm patterns based on the wavelet transform (Zhang, 2019). Unlike
the aforementioned methods that rely on average

(batch dimension is omitted for simplify description). Then, B branches features are recursively fed into their
respective 3×3 convolutional layer and added with the output of the preceding branches (except for F1), as follows:

( Convi (Fi) , i = 1, Convi   Fi + ¯Fi−1  , 1 < i ≤ B. (1)

¯Fi =

The receptive fields of features in {¯F1, . . . , ¯FB} increase sequentially as each ¯Fi (if i > 1) aggregates
information from preceding branches (detailed in Appendix B.3). Finally,

Figure 2: An overview of our proposed LLM-PS. Our LLM-PS incorporates a Multi-Scale Convolutional Neural Network (MSCNN)
and Time-to-Text (T2T) semantics extractor. Specifically, for input time series Y, MSCNN constructs multi-scale features
FMS with various receptive fields (darker colors indicate larger receptive fields), thereby capturing localized short-
term fluctuations and broader long-term trends. T2T extracts features FT2T with meaningful semantics to promote the LLM
to precisely understand the input time series. Finally, the diverse temporal patterns and rich semantics are integrated
via feature transferring and input into the LLM to generate precise time series ˆY.

constructed by the inverse wavelet transform IWT(·): ( Pb S = IWT(Zero(Wb low), {Wb high i}w i=1),

these features are concatenated together and fused through a 1×1 convolution layer to derive the output features Fout:

Fout = Conv1×1(Concate({¯F1, . . . , ¯FB}) + Fin, (2)

Pb L = IWT(Wb low, {Zero(Wb high i)}w i=1), (4)

where Fin is added in the output features via shortcut. Afterward, Fout is input into the subsequent MSCNN blocks to
produce the multi-scale features FMS for the LLM.

where Zero(·) operation generates features matching the input dimensions but filled with zeros.

For features in ¯F1, . . . , ¯FB , their receptive fields expand sequentially. Features with smaller receptive fields
more effectively capture local periodic fluctuations, whereas those with larger receptive fields focus on broader global
trends. Therefore, short-/long-term patterns PS/PL are reinforced in local-to-global/global-to-local assembling, as
follows: ( For b : 2 → B do: Pb S = Pb S + Pb−1 S ,

#### 3.2. Temporal Patterns Decoupling and Assembling

In the previous section, we show how MSCNN captures diverse temporal patterns by learning multi-scale features. For a
time series, its high-frequency and low-frequency components can effectively represent the corresponding short-term and
long-term temporal patterns, respectively (Wang et al., 2024; Kowsher et al., 2024). Therefore, we introduce a novel
patterns decoupling-assembling mechanism based on the wavelet transform (Zhang, 2019) (as shown in Fig. 3) to further
refine temporal patterns in the multi-scale features.

For b : (B − 1) → 1 do: Pb L = Pb L + Pb+1 L . (5)

After the assembling, features in ¯F1, . . . , ¯FB are reconstructed by combining the short-term and long-term
patterns: ¯Fb = Pb S + Pb L, b ∈{1, . . . , B}. (6)

Initially, the features ¯F1, . . . , ¯FB output by B branches in the MSCNN block are decoupled into low-frequency
component Wlow and high-frequency component Whigh using the wavelet transform (see Appendix B.1) WT(·):

#### 3.3. Time-to-Text Semantics Extraction

Wb low, {Wb high i}w i=1 =WT(¯Fb, w), b ∈{1, . . . , B}, (3)

LLMs are pre-trained on extensive textual data rich in semantic information, where individual words carry clear meaning.
In comparison, time series data is semantically

where w denotes the decomposition levels. Subsequently, the short-term pattern PS and long-term pattern PL are

4

|3×3|Col2|
|---|---|
|3×3||
|3×3|3×3|


3×3

Input Features

Channel-Wise Division

Data flow of high- and low-frequency patterns

WT

1×1

Title Suppressed Due to Excessive Size

indicator function that equals 1 if the i-th patch is masked (i.e., M(i)=1); li and ˆli denote the semantic labels of
the original and reconstructed patches, respectively. For a patch Xi (resp. ˆXi), its semantic label li (resp. ˆli) is
assigned as the word with the most similar LLM text embeddings, based on the similarity Si calculated as:

Si = Proj(Xi) · E⊤. (8)

IWT IWT Zero

Shortcut

Here, Proj(·) linearly transforms the input to the same dimensions of the LLM text embeddings E, and “⊤” denotes the
transpose. Further details of T2T and the semantic label assignment process are provided in the Appendix B.4.

3×3

Decoupling

Small Large

Pattern Decoupling + + +

Receptive Field

#### 3.4. Efficient Training of LLM-PS

Pattern Assembling

To efficiently train the LLM with huge parameters, we employ the parameter-efficient Low-Rank Adaptation (LoRA) (Hu et
al., 2021) to fine-tune the LLM on time series data. The total objective function is defined as:

+ + + Local-to-Global

1×1

Multi-Scale Features

Global-to-Local

Figure 3: The diagram of our MSCNN block. The divided features are initially fed into their related 3×3 convolutional
layers to obtain features (e.g., ¯F1) with various receptive fields. Then, these features are decoupled into long-term
patterns (e.g., P1 L) and short-term patterns (e.g., P1 S) using the Wavelet Transform (WT) and Inverse Wavelet
Transform (IWT). Subsequently, the long-term and short-term patterns are enhanced through global-to-local and local-to-
global assembling, respectively. Finally, the improved patterns are added together and passed through a 1×1
convolutional layer to obtain the multi-scale features.

LOBJ = LTIME + λLFEAT, (9)

where

T X

LTIME = 1

i=1 ∥Yi − ˆYi∥2,

T

(10)

C X

LFEAT = 1

j=1 ∥Fj MS − Fj T2T∥2.

C

Here, λ > 0 is a trade-off parameter to balance the contributation of LTIME and LFEAT, LTIME encourages the LLM to
forecast time series ˆY that closely matching the ground truth Y, LFEAT enriches the multi-scale features FMS through
semantic alignment with the T2T-generated features FT2T.

sparse, requiring an entire sequence to convey specific content. Consequently, LLMs pre-trained on textual data struggle
to precisely interpret semantics present in time series data. Recent advancements in audio processing (D´efossez et al.,
2023; Zhang et al., 2024b) suggest that self-supervised learning (Devlin et al., 2018) can promote models to understand
the serialized audio sequences that are similar to time series data. Inspired by them, we propose the T2T module with an
encoder-decoder structure to extract semantics within time series data for the LLM, as shown in Fig. 7.

### 4. Experiments

To demonstrate the effectiveness of our proposed LLM-PS, we conduct intensive experiments on multiple widely used time
series datasets for various tasks, including long-term, short-term, few-shot, and zero-shot forecasting. Baselines. We
compare our LLM-PS with a large range of SOTA methods, mainly including: 1) LLM-based methods: CALF (2024a), GPT4TS
(2023b), and TimeLLM (2024); 2) Transformer-based methods: Crossformer (2023a), FEDformer (2022), PatchTST (2023),
iTransformer (2024b), ETSformer (2022), and Autoformer (2021); 3) CNNbased methods: TimesNet (2023a), TCN (2018), and
MICN (2022); 4) MLP-based methods: DLinear (2023), TiDE (2023), and TimeMixer (2024). Besides, classical works N-HiTS
(2022) and N-BEATS (2019b) are also included for short-term forecasting. The details of these comparison methods can be
found in the Appendix A.1. Implementation Details. We follow (Zhou et al., 2023b; Liu et al., 2024a) and employ the pre-
trained GPT2

Initially, the input time series X ∈ RH×V is divided into P patches {Xi}P i=1, where Xi ∈ RL×V and L denotes the patch
length. Following (Hsu et al., 2021), approximately 75% of the patches are randomly masked. During training, the T2T
module learns precise semantic information by reconstructing the masked patches while predicting their labels. The loss
function is defined as:

P X

LT2T = 1

i=1 (1[M(i)=1]∥Xi − ˆXi∥2 + li log li

ˆli ). (7)

P

Here, ˆXi represents the reconstructed patch, 1[M(i)=1] is the

5

|Models Ours|Col2|(2024a)|(2024)|(2023b)|(2023)|(2024b)|(2023a)|(2022)|(2023a)|(2022)|(2023)|(2023)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Metric&lt;br&gt;MSE&lt;br&gt;MAE|Metric&lt;br&gt;MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|MSE&lt;br&gt;MAE|
|ETTm1|**0.354**&lt;br&gt;**0.376**|0.395&lt;br&gt;0.390|0.410&lt;br&gt;0.409|0.389&lt;br&gt;0.397|0.381&lt;br&gt;0.395|0.407&lt;br&gt;0.411|0.502&lt;br&gt;0.502|0.448&lt;br&gt;0.452|0.400&lt;br&gt;0.406|0.392&lt;br&gt;0.413|0.403&lt;br&gt;0.407|0.412&lt;br&gt;0.406|
|ETTm2|**0.262**&lt;br&gt;**0.314**|0.281&lt;br&gt;0.321|0.296&lt;br&gt;0.340|0.285&lt;br&gt;0.331|0.285&lt;br&gt;0.327|0.291&lt;br&gt;0.335|1.216&lt;br&gt;0.707|0.305&lt;br&gt;0.349|0.291&lt;br&gt;0.333|0.328&lt;br&gt;0.382|0.350&lt;br&gt;0.401|0.289&lt;br&gt;0.326|
|ETTh1|**0.418**&lt;br&gt;**0.420**|0.432&lt;br&gt;0.428|0.460&lt;br&gt;0.449|0.447&lt;br&gt;0.436|0.450&lt;br&gt;0.441|0.455&lt;br&gt;0.448|0.620&lt;br&gt;0.572|0.440&lt;br&gt;0.460|0.458&lt;br&gt;0.450|0.558&lt;br&gt;0.535|0.456&lt;br&gt;0.452|0.445&lt;br&gt;0.432|
|ETTh2|0.350&lt;br&gt;0.390|**0.349**&lt;br&gt;**0.382**|0.389&lt;br&gt;0.408|0.381&lt;br&gt;0.408|0.366&lt;br&gt;0.394|0.381&lt;br&gt;0.405|0.942&lt;br&gt;0.684|0.437&lt;br&gt;0.449|0.414&lt;br&gt;0.427|0.587&lt;br&gt;0.525|0.559&lt;br&gt;0.515|0.611&lt;br&gt;0.550|
|Weather|**0.238**&lt;br&gt;**0.269**|0.250&lt;br&gt;0.274|0.274&lt;br&gt;0.290|0.264&lt;br&gt;0.284|0.258&lt;br&gt;0.280|0.257&lt;br&gt;0.279|0.259&lt;br&gt;0.315|0.309&lt;br&gt;0.360|0.259&lt;br&gt;0.287|0.242&lt;br&gt;0.299|0.265&lt;br&gt;0.317|0.271&lt;br&gt;0.320|
|Electricity|**0.161**&lt;br&gt;**0.254**|0.175&lt;br&gt;0.265|0.223&lt;br&gt;0.309|0.205&lt;br&gt;0.290|0.216&lt;br&gt;0.304|0.178&lt;br&gt;0.270|0.244&lt;br&gt;0.334|0.214&lt;br&gt;0.327|0.192&lt;br&gt;0.295|0.186&lt;br&gt;0.294|0.212&lt;br&gt;0.300|0.251&lt;br&gt;0.344|
|Traffic|**0.427**&lt;br&gt;**0.279**|0.439&lt;br&gt;0.281|0.541&lt;br&gt;0.358|0.488&lt;br&gt;0.317|0.555&lt;br&gt;0.361|0.428&lt;br&gt;0.282|0.550&lt;br&gt;0.304|0.610&lt;br&gt;0.376|0.620&lt;br&gt;0.336|0.541&lt;br&gt;0.315|0.625&lt;br&gt;0.383|0.760&lt;br&gt;0.473|
|ILI|**1.735**&lt;br&gt;0.854|1.861&lt;br&gt;0.924|1.829&lt;br&gt;0.924|1.871&lt;br&gt;**0.852**|2.145&lt;br&gt;0.897|2.258&lt;br&gt;0.957|3.749&lt;br&gt;1.284|2.705&lt;br&gt;1.097|2.267&lt;br&gt;0.927|2.985&lt;br&gt;1.186|4.453&lt;br&gt;1.553|5.216&lt;br&gt;1.614|
|ECG|**0.225**&lt;br&gt;**0.250**|0.258&lt;br&gt;0.260|0.250&lt;br&gt;0.264|0.262&lt;br&gt;0.260|0.253&lt;br&gt;0.277|0.257&lt;br&gt;0.271|0.244&lt;br&gt;0.269|0.255&lt;br&gt;0.279|0.291&lt;br&gt;0.305|0.305&lt;br&gt;0.314|0.291&lt;br&gt;0.307|0.291&lt;br&gt;0.307|
|1st _Count_|**15**|2|0|1|0|0|0|0|0|0|0|0|


Table 1: Average results of multivariate long-term forecasting across four different prediction lengths T in the set
{96, 192, 336, 720}, and the full results are shown in Tab. 7. The best and second best results are in bold and
underlined, respectively. The row “1st Count” records the times of each method achieving the top results. Here, we
reproduce the methods with superscript “⋆” according to their official codebase with the identical experimental setups
as ours for fairness comparison. Results for other compared methods are from (Liu et al., 2024b).

Models LLM-PS CALF⋆ TimeLLM⋆ GPT4TS⋆ PatchTST iTransformer Crossformer FEDformer TimesNet MICN DLinear TiDE

Ours (2024a) (2024) (2023b) (2023) (2024b) (2023a) (2022) (2023a) (2022) (2023) (2023)

Title Suppressed Due to Excessive Size

Table 2: Results of short-term forecasting across monthly, quarterly, yearly, and others subsets. The input and output
time series lengths are [12, 96] and [6, 48], respectively.

Models LLM-PS CALF TimeLLM GPT4TS PatchTST iTransformer ETSformer FEDformer Autoformer TimesNet TCN N-HiTS N-BEATS
DLinear LSSL LSTM (Ours) (2024a) (2024) (2023b) (2023) (2024b) (2022) (2022) (2021) (2023a) (2018) (2022) (2019b) (2023)
(2022) (1997)

|Yearly|SMAPE&lt;br&gt;MASE&lt;br&gt;OWA|13.277 13.314 13.419 13.531 13.477 14.252 18.009 13.728 13.974 13.387 14.920 13.418 13.436 16.965 16.675 176.040&lt;br&gt;2.973 3.009 3.005 3.015 3.019 3.208 4.487 3.048 3.134 2.996 3.364 3.045 3.043 4.283 19.953 31.033&lt;br&gt;0.780 0.786 0.789 0.793 0.792 0.840 1.115 0.803 0.822 0.786 0.880 0.793 0.794 1.058 4.397 9.290|
|---|---|---|


SMAPE 13.277 13.314 13.419 13.531 13.477 14.252 18.009 13.728 13.974 13.387 14.920 13.418 13.436 16.965 16.675 176.040
MASE 2.973 3.009 3.005 3.015 3.019 3.208 4.487 3.048 3.134 2.996 3.364 3.045 3.043 4.283 19.953 31.033 OWA 0.780 0.786
0.789 0.793 0.792 0.840 1.115 0.803 0.822 0.786 0.880 0.793 0.794 1.058 4.397 9.290

Yearly

|Quarterly|SMAPE&lt;br&gt;MASE&lt;br&gt;OWA|9.995 10.049 10.110 10.177 10.380 10.755 13.376 10.792 11.338 10.100 11.122 10.202 10.124 12.145 65.999 172.808&lt;br&gt;1.164 1.166 1.178 1.194 1.233 1.284 1.906 1.283 1.365 1.182 1.360 1.194 1.169 1.520 17.662 19.753&lt;br&gt;0.878 0.871 0.889 0.898 0.921 0.957 1.302 0.958 1.012 0.890 1.001 0.899 0.886 1.106 9.436 15.049|
|---|---|---|


SMAPE 9.995 10.049 10.110 10.177 10.380 10.755 13.376 10.792 11.338 10.100 11.122 10.202 10.124 12.145 65.999 172.808
MASE 1.164 1.166 1.178 1.194 1.233 1.284 1.906 1.283 1.365 1.182 1.360 1.194 1.169 1.520 17.662 19.753 OWA 0.878 0.871
0.889 0.898 0.921 0.957 1.302 0.958 1.012 0.890 1.001 0.899 0.886 1.106 9.436 15.049

Quarterly

|Monthly|SMAPE&lt;br&gt;MASE&lt;br&gt;OWA|12.585 12.624 12.980 12.894 12.959 13.721 14.588 14.260 13.958 12.679 15.626 12.791 12.677 13.514 64.664 143.237&lt;br&gt;0.924 0.922 0.963 0.956 0.970 1.074 1.368 1.102 1.103 0.933 1.274 0.969 0.937 1.037 16.245 16.551&lt;br&gt;0.871 0.871 0.903 0.897 0.905 0.981 1.149 1.012 1.002 0.878 1.141 0.899 0.880 0.956 9.879 12.747|
|---|---|---|


SMAPE 12.585 12.624 12.980 12.894 12.959 13.721 14.588 14.260 13.958 12.679 15.626 12.791 12.677 13.514 64.664 143.237
MASE 0.924 0.922 0.963 0.956 0.970 1.074 1.368 1.102 1.103 0.933 1.274 0.969 0.937 1.037 16.245 16.551 OWA 0.871 0.871
0.903 0.897 0.905 0.981 1.149 1.012 1.002 0.878 1.141 0.899 0.880 0.956 9.879 12.747

Monthly

|Others|SMAPE&lt;br&gt;MASE&lt;br&gt;OWA|4.550 4.773 4.795 4.940 4.952 5.615 7.267 4.954 5.485 4.891 7.186 5.061 4.925 6.709 121.844 186.282&lt;br&gt;3.089 3.119 3.178 3.228 3.347 3.977 5.240 3.264 3.865 3.302 4.677 3.216 3.391 4.953 91.650 119.294&lt;br&gt;0.966 0.990 1.006 1.029 1.049 1.218 1.591 1.036 1.187 1.035 1.494 1.040 1.053 1.487 27.273 38.411|
|---|---|---|


SMAPE 4.550 4.773 4.795 4.940 4.952 5.615 7.267 4.954 5.485 4.891 7.186 5.061 4.925 6.709 121.844 186.282 MASE 3.089
3.119 3.178 3.228 3.347 3.977 5.240 3.264 3.865 3.302 4.677 3.216 3.391 4.953 91.650 119.294 OWA 0.966 0.990 1.006 1.029
1.049 1.218 1.591 1.036 1.187 1.035 1.494 1.040 1.053 1.487 27.273 38.411

Others

|Average|SMAPE&lt;br&gt;MASE&lt;br&gt;OWA|11.721 11.770 11.983 11.991 12.059 12.726 14.718 12.840 12.909 11.829 13.961 11.927 11.851 13.639 67.156 160.031&lt;br&gt;1.561 1.570 1.595 1.600 1.623 15.336 2.408 1.701 1.771 1.585 1.945 1.613 1.599 2.095 21.208 25.788&lt;br&gt;0.840 0.845 0.859 0.861 0.869 0.929 1.172 0.918 0.939 0.851 1.023 0.861 0.855 1.051 8.021 12.642|
|---|---|---|


SMAPE 11.721 11.770 11.983 11.991 12.059 12.726 14.718 12.840 12.909 11.829 13.961 11.927 11.851 13.639 67.156 160.031
MASE 1.561 1.570 1.595 1.600 1.623 15.336 2.408 1.701 1.771 1.585 1.945 1.613 1.599 2.095 21.208 25.788 OWA 0.840 0.845
0.859 0.861 0.869 0.929 1.172 0.918 0.939 0.851 1.023 0.861 0.855 1.051 8.021 12.642

Average

1st Count 14 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0

model (Radford et al., 2019) (the first six layers) as the default LLM backbone. Our LLM-PS utilizes Adam as the
optimizier and the learning rate is 0.0005. For the parameterefficient LoRA, its rank, scale factor, and dropout ratio
are set to 8, 32, and 0.1, respectively. Additionally, the tradeoff parameter λ in Eq. (9) is set to 0.01 and its
parametric sensitivities are analyzed in Appendix B.5.

Error (MAE), where lower values of these metrics indicate better model performance. The dataset descriptions are
provided in Appendix A.2. Results. Tab. 1 reports the brief average results of long-term forecasting with various
prediction lengths. Firstly, we can observe that our LLM-PS surpasses the baseline methods in most instances, achieving
the top results in 15 of 18 cases. Secondly, compared with the SOTA LLM-based methods, i.e., CALF, TimeLLM, and GPT4TS,
our approach achieves consistent MSE/MAE reductions of 6%/3%, 11%/9%, and 9%/5%, respectively. Thirdly, our method
significantly outperforms traditional deep learning methods based on Transformer, CNN, and MLP, especially on the
Traffic, ILI, and ECG datasets. These results indicate that our LLM-PS can precisely predict long-term time series by
effectively leveraging temporal patterns and semantics within the input series with limited length.

#### 4.1. Long-Term Forecasting

Setups. We conduct intensive experiments across various popular real-world datasets, including four Electricity
Transformer Temperature (ETT) subsets (ETTh1, ETTh2, ETTm1, and ETTm2) (Zhou et al., 2021a), Weather, Electricity,
Traffic, Illness (Wu et al., 2021), and Electrocardiography (ECG) (Moody & Mark, 2001) datasets. The input length H of
time series data is set to 96, with prediction lengths T spanning {96, 192, 336, 720}. The evaluation metrics are Mean
Squared Error (MSE) and Mean Absolute

6

Title Suppressed Due to Excessive Size

Table 3: Average results of few-shot forecasting with 10% training data, where the prediction lengths T ∈ {96, 192, 336,
720} and the full results are provided in Tab. 8.

Models LLM-PS CALF TimeLLM GPT4TS PatchTST Crossformer FEDformer TimesNet MICN DLinear TiDE Ours (2024a) (2024) (2023b)
(2023) (2023a) (2022) (2023a) (2022) (2023) (2023)

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

ETTm1 0.497 0.454 0.504 0.462 0.636 0.512 0.608 0.500 0.557 0.483 1.340 0.848 0.696 0.572 0.673 0.534 0.970 0.674 0.567
0.499 0.515 0.469

ETTm2 0.281 0.324 0.302 0.330 0.348 0.343 0.303 0.336 0.295 0.334 1.985 1.048 0.356 0.392 0.321 0.354 1.073 0.716 0.329
0.382 0.303 0.337

ETTh1 0.632 0.546 0.644 0.541 0.765 0.584 0.689 0.555 0.683 0.546 1.744 0.914 0.750 0.607 0.865 0.625 1.405 0.814 0.647
0.552 0.779 0.604

ETTh2 0.409 0.420 0.419 0.427 0.589 0.498 0.579 0.497 0.550 0.487 3.139 1.378 0.553 0.525 0.476 0.463 2.533 1.158 0.441
0.458 0.421 0.428

1st Count 7 1 0 0 0 0 0 0 0 0 0

Table 4: Average results of zero-shot forecasting, where prediction lengths T ∈{96, 192, 336, 720} and Tab. 9 shows the
full results. The term “ETTh1 → ETTm1” indicates that models trained on the ETTh1 dataset and are evaluated on the ETTm1
dataset, the convention is also followed for other terms.

Models LLM-PS CALF TimeLLM GPT4TS PatchTST Crossformer FEDformer TimesNet MICN DLinear TiDE Ours (2024a) (2024) (2023b)
(2023) (2023a) (2022) (2023a) (2022) (2023) (2023)

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

ETTh1 →ETTm1 0.721 0.541 0.755 0.574 0.847 0.565 0.798 0.574 0.894 0.610 0.999 0.736 0.765 0.588 0.794 0.575 1.439 0.780
0.760 0.577 0.774 0.574

ETTh1 →ETTm2 0.316 0.361 0.316 0.355 0.315 0.357 0.317 0.359 0.318 0.362 1.120 0.789 0.357 0.403 0.339 0.370 2.428 1.236
0.399 0.439 0.314 0.355

ETTh2 →ETTm1 0.714 0.552 0.836 0.586 0.868 0.595 0.920 0.610 0.871 0.596 1.195 0.711 0.741 0.588 1.286 0.705 0.764 0.601
0.778 0.594 0.841 0.590

ETTh2 →ETTm2 0.322 0.359 0.319 0.360 0.322 0.363 0.331 0.371 0.420 0.433 2.043 1.124 0.365 0.405 0.361 0.390 0.527 0.519
0.496 0.496 0.321 0.364

1st Count 5 2 0 0 0 0 0 0 0 0 2

#### 4.2. Short-Term Forecasting

zons for the challenging few-shot forecasting tasks are reported in Tab. 3. In such cases, our LLM-PS always achieves
the best results in 7 of 8 cases, which demonstrates that our LLM-PS can effectively learn valuable temporal information
only using limited data. Compared with the LLM-based methods, i.e., CALF, TimeLLM, and GPT4TS, our method yields 3%,
17%, and 13% performance improvements, respectively.

Setups. We evaluate the M4 dataset (Makridakis et al., 2018), which records marketing data in monthly, quarterly,
yearly, and others. For short-term forecasting, the prediction horizons are relatively small and range in [6, 48], while
the input lengths are twice their corresponding prediction horizons. The evaluation metrics are Mean Absolute Scaled
Error (MSAE), Symmetric Mean Absolute Percentage Error (SMAPE), and Overall Weighted Average (OWA). Formal definitions
of these metrics are provided in Appendix A.3. Results. The results of short-term forecasting across monthly, quarterly,
yearly, and others subsets are shown in Tab. 2. Our proposed LLM-PS achieves a SMAPE of 11.721, a MASE of 1.561, and an
OWA of 0.840, which consistently outperforms other compared methods.

In addition to the few-shot learning tasks, we also considered the more challenging zero-shot learning tasks. The
experimental results are provided in Tab. 4. Our method significantly outperforms the comparison methods in 5 of 8
cases. Notably, our LLM-PS outperforms CALF, TimeLLM, and GPT4TS, which also fine-tune the LLM for time series
forecasting, showing performance improvements of 5%, 8%, and 9%, respectively.

#### 4.3. Few/Zero-Shot Forecasting

#### 4.4. Model Analysis

Setups. LLMs possess powerful few-shot and zero-shot learning capabilities (Zhou et al., 2023b; Jin et al., 2024), which
are crucial for time series forecasting in real-world scenarios. We also verify the few-shot and zero-shot learning
performance of our LLM-PS. For few-shot forecasting, there are only 10% training data of ETT datasets are available. For
zero-shot forecasting, LLMs trained on one dataset are directly tested on other datasets. The training setups are the
same as those in long-term forecasting. Results. The average results over various prediction hori-

Multi-Scale Feature Extraction. We compare our MSCNN against existing multi-scale features extraction techniques
(Kowsher et al., 2024), which convert the input time series into diverse scales using average or max pooling layers with
varying window sizes. As shown in Fig. 4a, our method consistently outperforms those employing pooling operations. These
results indicate that our method excels in extracting the multi-scale features, thereby effectively capturing the short-
term and long-term patterns in input time

7

|0.50&lt;br&gt;0.48&lt;br&gt;MAE&lt;br&gt;0.46&lt;br&gt;and&lt;br&gt;0.44&lt;br&gt;MSE&lt;br&gt;0.42&lt;br&gt;of&lt;br&gt;0.40 LLM&amp;#45;PS (Ours) Mean&lt;br&gt;0.38 Average Pooling&lt;br&gt;0.36 Fourier Transform&lt;br&gt;96 192 336 72&lt;br&gt;Prediction Horizon|LLM&amp;#45;PS (Ours)&lt;br&gt;Average Pooling&lt;br&gt;Fourier Transform|0.65&lt;br&gt;LLM&amp;#45;PS (Ours)&lt;br&gt;0.60 CALF&lt;br&gt;0.55 PatchTSTS&lt;br&gt;TimesNet&lt;br&gt;0.50 DLinear MSE&lt;br&gt;0.45&lt;br&gt;0.40&lt;br&gt;0.35&lt;br&gt;0.30&lt;br&gt;0 0.0 0.1 0.3 0.5&lt;br&gt;Noise Factor|LLM&amp;#45;PS (Ours)&lt;br&gt;CALF&lt;br&gt;PatchTSTS&lt;br&gt;TimesNet&lt;br&gt;DLinear|0.65&lt;br&gt;LLM&amp;#45;PS (Ours)&lt;br&gt;0.60 CALF&lt;br&gt;0.55 PatchTSTS&lt;br&gt;TimesNet&lt;br&gt;0.50 DLinear MAE&lt;br&gt;0.45&lt;br&gt;0.40&lt;br&gt;0.35&lt;br&gt;0.30&lt;br&gt;0.0 0.1 0.3 0.5&lt;br&gt;Noise Factor|LLM&amp;#45;PS (Ours)&lt;br&gt;CALF&lt;br&gt;PatchTSTS&lt;br&gt;TimesNet&lt;br&gt;DLinear|
|---|---|---|---|---|---|

Figure 4: Analysis of (a) multi-scale feature extraction and (b) temporal patterns decoupling. Subfigures (c) and (d)
show the MSE/MAE of various methods on noisy ETTh1 datasets. Notably, lower MSE/MAE indicates better model performance.

Table 6: The fine-tuning costs, mean MSE/MAE across four datasets of our LLM-PS and other LLM-based methods.

series. Further analysis is provided in Appendix B.2. Temporal Patterns Decoupling. We compare our wavelettransform-
based decoupling with the Fourier-transformbased and pooling-based decoupling techniques used in existing methods (Wu et
al., 2021; Wang et al., 2024). As shown in Fig. 4b, our decoupling operation based on wavelet transform achieves better
performance than those based on Fourier transform and average pooling. The short-term and long-term components
decomposed by our LLM-PS and compared methods are visualized in Fig. 6. Semantic Information Utilization. Tab. 5 reports
the results of ablating our proposed T2T module for semantic information learning. It can be found that the performance
of LLM decreases in 9 of 10 cases, which demonstrates that the semantic information extracted by our T2T is helpful in
enhancing the TSF performance of LLM. The detailed structure of the T2T module is provided in Appendix B.4.

|Model|Time (s)&lt;br&gt;ETTh1 ETTm1 Weather Traffci|Mean MSE Mean MAE|
|---|---|---|
|GPT4TS (2023b)&lt;br&gt;Time&amp;#45;LLM (2024)&lt;br&gt;LLMMixer (2024)&lt;br&gt;CALF (2024a)&lt;br&gt;LLM&amp;#45;PS(Ours)|421&lt;br&gt;1140&lt;br&gt;4565&lt;br&gt;59164&lt;br&gt;2780&lt;br&gt;11929&lt;br&gt;36188&lt;br&gt;465136&lt;br&gt;635&lt;br&gt;2493&lt;br&gt;9640&lt;br&gt;10464&lt;br&gt;354&lt;br&gt;1394&lt;br&gt;1259&lt;br&gt;4929&lt;br&gt;**192**&lt;br&gt;**481**&lt;br&gt;**260**&lt;br&gt;**1092**|0.339&lt;br&gt;0.323&lt;br&gt;0.372&lt;br&gt;0.346&lt;br&gt;0.372&lt;br&gt;0.346&lt;br&gt;0.315&lt;br&gt;0.302&lt;br&gt;**0.301**&lt;br&gt;**0.298**|


Model Time (s) Mean MSE Mean MAE ETTh1 ETTm1 Weather Traffic

GPT4TS (2023b) 421 1140 4565 59164 0.339 0.323 Time-LLM (2024) 2780 11929 36188 465136 0.372 0.346 LLMMixer (2024) 635
2493 9640 10464 0.372 0.346 CALF (2024a) 354 1394 1259 4929 0.315 0.302 LLM-PS (Ours) 192 481 260 1092 0.301 0.298

patterns and semantic information from time series data, and thus achieving reliable TSF performance. Noisy Data. Time
series data in real-world applications is usually noisy due to measurement errors and missing values. In such cases, the
target time series is more challenging than that in training data, and the models are hard to predict reliably. To
assess the robustness of our LLM-PS against noise, we evaluate it on the ETTh1 dataset with Gaussian noise, where the
noise factors are in [0.0, 0.1, 0.3, 0.5]. Both the input and prediction lengths are set to 96. As reported in Fig.
4c&4d, our method consistently achieves superior performance across various noise factors. In particular, our method
outperforms other comparison approaches by an even more significant margin as the noise factor increases. These
experimental results demonstrate that our LLM-PS is robust to noise and can be effectively applied to real-world
scenarios for time series forecasting.

Table 5: Long-term forecasting results of ablating our proposed T2T module on the ETTh1 dataset.

Type MSE / MAE 96 192 336 720 Mean

w/o T2T 0.373 / 0.395 0.416 / 0.425 0.439 / 0.432 0.464 /0.470 0.426 /0.431 LLM-PS 0.369 / 0.388 0.418 / 0.415 0.432 /
0.426 0.452 / 0.451 0.418 / 0.420

Model Efficiency. We evaluate the efficiency of our method with other LLM fine-tuning methods on four datasets,
including ETTh1, ETTm1, Weather, and Traffic. For a fair comparison, all experiments adopt the same experimental
settings and LLM backbone (i.e., GPT2), where input length, prediction length, and patience for early stopping are set
to 96, 96, and 5, respectively. All experiments are conducted on a single NVIDIA RTX 4090 GPU. As reported in Tab. 6,
our proposed LLM-PS achieves the best performance with significantly lower training costs. Compared with LLMMixer, which
also learns multi-scale features from the multiple input series with various scales, our method attains a 17%
performance edge while utilizing just 9% of the training time. These experimental results indicate that our method can
efficiently and effectively learn the temporal

### 5. Conclusion

In this paper, we identify the intrinsic characteristics of time series data, i.e., diverse temporal patterns and
semantic sparsity. These properties are critical for reliable time series forecasting but are usually neglected by
existing LLM-based methods, and thus resulting in suboptimal performance. To address this problem, we propose LLM-PS, a
novel TSF framework that learns fundamental temporal patterns and valuable semantics from time series data through the
novel MSCNN and T2T modules. As a result, our LLM-PS can comprehensively understand time series data, thereby enabling
accurate generation of time series. Our intensive

8

Title Suppressed Due to Excessive Size

ts integrator: Integrating llm for enhanced time series modeling. arXiv preprint arXiv:2410.16489, 2024.

experiments demonstrate that LLM-PS achieves SOTA performance across multiple benchmark datasets spanning critical real-
world domains, mainly including finance, energy, transportation, and healthcare.

Cheng, M., Liu, Q., Liu, Z., Zhang, H., Zhang, R., and Chen, E. Timemae: Self-supervised representations of time series
with decoupled masked autoencoders. arXiv preprint arXiv:2303.00320, 2023.

### Impact Statement

This paper proposes a novel framework, LLM-PS, which aims to advance time series forecasting using large language
models. Our work focuses on integrating temporal patterns and semantics in time series data, enabling state-of-the-art
performance in various forecasting scenarios.

Chung, H., Kim, J., Kwon, J.-m., Jeon, K.-H., Lee, M. S., and Choi, E. Text-to-ecg: 12-lead electrocardiogram synthesis
conditioned on clinical text reports. In IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), pp. 1–5. IEEE, 2023.

We anticipate several potential societal benefits, including improved decision-making in critical areas such as finance,
healthcare, and energy management. However, as with any machine learning advancement, ethical considerations must be
acknowledged, including the potential misuse of forecasting technologies in domains like financial speculation or
surveillance. To mitigate these risks, we encourage responsible and transparent applications of our methods.

Cleveland, R. B., Cleveland, W. S., McRae, J. E., Terpenning, I., et al. Stl: A seasonal-trend decomposition. J. off.
Stat, 6(1):3–73, 1990.

Cochran, W. T., Cooley, J. W., Favin, D. L., Helms, H. D., Kaenel, R. A., Lang, W. W., Maling, G. C., Nelson, D. E.,
Rader, C. M., and Welch, P. D. What is the fast fourier transform? Proceedings of the IEEE, 55(10):1664–1674, 1967.

In conclusion, we do not identify any immediate ethical concerns or risks unique to our methodology, and we believe it
can benefit society.

Das, A., Kong, W., Leach, A., Sen, R., and Yu, R. Longterm forecasting with TiDE: Time-series dense encoder. arXiv
preprint arXiv:2304.08424, 2023.

### References

D´efossez, A., Copet, J., Synnaeve, G., and Adi, Y. High fidelity neural audio compression. Transactions on Machine
Learning Research (TMLR), 2023.

Angryk, R. A., Martens, P. C., Aydin, B., Kempton, D., Mahajan, S. S., Basodi, S., Ahmadzadeh, A., Cai, X., Filali
Boubrahimi, S., Hamdi, S. M., et al. Multivariate time series dataset for space weather data analytics. Scientific Data,
7(1):227, 2020.

Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language
understanding. CoRR, abs/1810.04805, 2018.

Bai, S., Kolter, J. Z., and Koltun, V. An empirical evaluation of generic convolutional and recurrent networks for
sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

Gao, S.-H., Cheng, M.-M., Zhao, K., Zhang, X.-Y., Yang, M.-H., and Torr, P. Res2net: A new multi-scale backbone
architecture. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 43(2):652–662, 2019.

Gu, A., Goel, K., and R´e, C. Efficiently modeling long sequences with structured state spaces. In International
Conference on Learning Representations (ICLR), 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS),
33:1877–1901, 2020.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.

Cao, D., Jia, F., Arik, S. O., Pfister, T., Zheng, Y., Ye, W., and Liu, Y. Tempo: Prompt-based generative pre-trained
transformer for time series forecasting. In International Conference on Learning Representations (ICLR).

Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Comput (NC), 1997.

Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A. Hubert: Selfsupervised speech
representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language
Processing (TASLR), 29: 3451–3460, 2021.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler, M., and Dubrawski, A. N-HiTs: Neural hierarchical
interpolation for time series forecasting. arXiv preprint arXiv:2201.12886, 2022.

Chen, C., Oliveira, G., Noghabi, H. S., and Sylvain, T. Llm-

9

Title Suppressed Due to Excessive Size

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask
learners. OpenAI blog, 1(8):9, 2019.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of
large language models. arXiv preprint arXiv:2106.09685, 2021.

Rubenstein, P. K., Asawaroengchai, C., Nguyen, D. D., Bapna, A., Borsos, Z., Quitry, F. d. C., Chen, P., Badawy, D. E.,
Han, W., Kharitonov, E., et al. Audiopalm: A large language model that can speak and listen. arXiv preprint
arXiv:2306.12925, 2023.

Hyndman, R. Forecasting: principles and practice. OTexts, 2018.

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-f., Pan, S., et al. Time-llm:
Time series forecasting by reprogramming large language models. In International Conference on Learning Representations
(ICLR), 2024.

Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. Deepar: Probabilistic forecasting with autoregressive
recurrent networks. International Journal of Forecasting (IJF), 36(3):1181–1191, 2020.

Kowsher, M., Sobuj, M. S. I., Prottasha, N. J., Alanis, E. A., Garibay, O. O., and Yousefi, N. Llm-mixer: Multiscale
mixing in llms for time series forecasting. arXiv preprint arXiv:2410.11674, 2024.

Taylor, S. J. and Letham, B. Forecasting at scale. The American Statistician, 72(1):37–45, 2018.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Liu, P., Guo, H., Dai, T., Li, N., Bao, J., Ren, X., Jiang, Y., and Xia, S.-T. Calf: Aligning llms for time series
forecasting via cross-modal fine-tuning. arXiv preprint arXiv:2403.07300, 2024a.

Trindade, A. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. DOI:
https://doi.org/10.24432/C58C86.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. iTransformer: Inverted transformers are effective for
time series forecasting. International Conference on Learning Representations (ICLR), 2024b.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in Neural Information Processing
Systems (NeurIPS), 30, 2017.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. The m4 competition: Results, findings, conclusion and way forward.
International Journal of Forecasting (IJF), 34 (4):802–808, 2018.

Wang, H., Peng, J., Huang, F., Wang, J., Chen, J., and Xiao, Y. MICN: Multi-scale local and global context modeling for
long-term series forecasting. In International Conference on Learning Representations (ICLR), 2022.

Moody, G. B. and Mark, R. G. The impact of the mit-bih arrhythmia database. IEEE Engineering in Medicine and Biology
Magazine (IEMBM), 20(3):45–50, 2001.

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., and ZHOU, J. Timemixer: Decomposable multiscale mixing
for time series forecasting. In International Conference on Learning Representations (ICLR), 2024.

Nie, Y., H. Nguyen, N., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with
transformers. In International Conference on Learning Representations (ICLR), 2023.

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., and Sun, L. Transformers in time series: a survey. In
Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI), pp. 6778– 6786,
2023.

Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. N-beats: Neural basis expansion analysis for interpretable
time series forecasting. arXiv preprint arXiv:1905.10437, 2019a.

Woo, G., Liu, C., Sahoo, D., Kumar, A., and Hoi, S. C. H. ETSformer: Exponential smoothing transformers for time-series
forecasting. arXiv preprint arXiv:2202.01381, 2022.

Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. N-BEATS: Neural basis expansion analysis for interpretable
time series forecasting. International Conference on Learning Representations (ICLR), 2019b.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with Auto-Correlation for longterm series
forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Patton, A. Copula methods for forecasting multivariate time series. Handbook of economic forecasting, 2:899–960, 2013.

10

Title Suppressed Due to Excessive Size

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. TimesNet: Temporal 2d-variation modeling for general time
series analysis. In International Conference on Learning Representations (ICLR), 2023a.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for
long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp.
11106–11115, 2021b.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time
series analysis. In International Conference on Learning Representations (ICLR), 2023b.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. Fedformer: Frequency enhanced decomposed transformer for long-
term series forecasting. In International Conference on Machine Learning (ICML), pp. 27268– 27286. PMLR, 2022.

Wu, H., Zhou, H., Long, M., and Wang, J. Interpretable weather forecasting for worldwide stations with a unified deep
model. Nature Machine Intelligence (NMI), 5(6): 602–611, 2023c.

Zhou, T., Niu, P., Sun, L., Jin, R., et al. One fits all: Power general time series analysis by pretrained lm. Advances
in Neural Information Processing Systems (NeurIPS), 36: 43322–43355, 2023a.

Xue, H. and Salim, F. D. Promptcast: A new promptbased learning paradigm for time series forecasting. IEEE Transactions
on Knowledge and Data Engineering (TKDE), 2023.

Zhou, T., Niu, P., Wang, X., Sun, L., and Jin, R. One Fits All: Power general time series analysis by pretrained lm.
Advances in Neural Information Processing Systems (NeurIPS), 36, 2023b.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the
AAAI Conference on Artificial Intelligence, volume 37, pp. 11121–11128, 2023.

Zhang, D. Wavelet transform. Fundamentals of Image Data Mining: Analysis, Features, Classification and Retrieval, pp.
35–44, 2019.

Zhang, G. P. and Qi, M. Neural network forecasting for seasonal and trend time series. European Journal of Operational
Research (EJOR), 160(2):501–514, 2005.

Zhang, X., Chowdhury, R. R., Gupta, R. K., and Shang, J. Large language models for time series: A survey. arXiv preprint
arXiv:2402.01801, 2024a.

Zhang, X., Zhang, D., Li, S., Zhou, Y., and Qiu, X. Speechtokenizer: Unified speech tokenizer for speech language
models. In International Conference on Learning Representations (ICLR), 2024b.

Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series
forecasting. In International Conference on Learning Representations (ICLR), 2023a.

Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series
forecasting. In International Conference on Learning Representations (ICLR), 2023b.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for
long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp.
11106–11115, 2021a.

11

### A. Full Experimental Results

Table 7: Full results for long-term forecasting consider prediction horizons H within {96, 192, 336, 720}. The term
“Avg.” represents the average results across the four prediction lengths. The best and second best outcomes are
highlighted in bold and underlined, respectively. The notation “1st Count” denotes the frequency of each method
achieving the top results.

|LLM&amp;#45;based|Transformer&amp;#45;based|CNN&amp;#45;based|MLP&amp;#45;based|
|---|---|---|---|


Title Suppressed Due to Excessive Size

|LLM&amp;#45;PS CALF TimeLLM GPT4TS PatchTST iTransformer Crossformer FEDformer Autoformer Informer TimesNet MICN DLinear TiDE&lt;br&gt;Models&lt;br&gt;Ours ((2024a)) (2024) (2023b) (2023) (2024b) (2023a) (2022) (2021) (2021b) (2023a) (2022) (2023) (2023)&lt;br&gt;Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE|LLM&amp;#45;PS|Col3|CALF|Col5|TimeLLM|Col7|GPT4TS|Col9|PatchTST|Col11|iTransformer|Col13|Crossformer|Col15|FEDformer|Col17|Autoformer|Col19|Informer|Col21|TimesNet|Col23|MICN|Col25|DLinear|Col27|TiDE|Col29|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Models&lt;br&gt;LLM&amp;#45;PS&lt;br&gt;CALF&lt;br&gt;TimeLLM&lt;br&gt;GPT4TS&lt;br&gt;PatchTST&lt;br&gt;iTransformer Crossformer FEDformer&lt;br&gt;Autoformer&lt;br&gt;Informer&lt;br&gt;TimesNet&lt;br&gt;MICN&lt;br&gt;DLinear&lt;br&gt;TiDE&lt;br&gt;**Ours**&lt;br&gt;((2024a))&lt;br&gt;(2024)&lt;br&gt;(2023b)&lt;br&gt;(2023)&lt;br&gt;(2024b)&lt;br&gt;(2023a)&lt;br&gt;(2022)&lt;br&gt;(2021)&lt;br&gt;(2021b)&lt;br&gt;(2023a)&lt;br&gt;(2022)&lt;br&gt;(2023)&lt;br&gt;(2023)&lt;br&gt;Metric&lt;br&gt;MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE|**Ours**|**Ours**|((2024a))|((2024a))|(2024)|(2024)|(2023b)|(2023b)|(2023)|(2023)|(2024b)|(2024b)|(2023a)|(2023a)|(2022)|(2022)|(2021)|(2021)|(2021b)|(2021b)|(2023a)|(2023a)|(2022)|(2022)|(2023)|(2023)|(2023)|(2023)|
|Models&lt;br&gt;LLM&amp;#45;PS&lt;br&gt;CALF&lt;br&gt;TimeLLM&lt;br&gt;GPT4TS&lt;br&gt;PatchTST&lt;br&gt;iTransformer Crossformer FEDformer&lt;br&gt;Autoformer&lt;br&gt;Informer&lt;br&gt;TimesNet&lt;br&gt;MICN&lt;br&gt;DLinear&lt;br&gt;TiDE&lt;br&gt;**Ours**&lt;br&gt;((2024a))&lt;br&gt;(2024)&lt;br&gt;(2023b)&lt;br&gt;(2023)&lt;br&gt;(2024b)&lt;br&gt;(2023a)&lt;br&gt;(2022)&lt;br&gt;(2021)&lt;br&gt;(2021b)&lt;br&gt;(2023a)&lt;br&gt;(2022)&lt;br&gt;(2023)&lt;br&gt;(2023)&lt;br&gt;Metric&lt;br&gt;MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|


Models LLM-PS CALF TimeLLM GPT4TS PatchTST iTransformer Crossformer FEDformer Autoformer Informer TimesNet MICN DLinear
TiDE Ours ((2024a)) (2024) (2023b) (2023) (2024b) (2023a) (2022) (2021) (2021b) (2023a) (2022) (2023) (2023)

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

|96 0.288 0.334 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672 0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373&lt;br&gt;192 0.333 0.361 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384 0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391 ETTm1&lt;br&gt;336 0.367 0.386 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410 0.411 0.408 0.426 0.413 0.413 0.423 0.413&lt;br&gt;720 0.429 0.424 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456 0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448&lt;br&gt;Avg. 0.354 0.376 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|0.288|0.334|0.323|0.349|0.359|0.381|0.329|0.364|0.321|0.360|0.341|0.376|0.360|0.401|0.379|0.419|0.505|0.475|0.672|0.571|0.338|0.375|0.316|0.362|0.345|0.372|0.352|0.373|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ETTm1&lt;br&gt;96&lt;br&gt;**0.288 0.334** 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672 0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373&lt;br&gt;192&lt;br&gt;**0.333 0.361** 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384 0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391&lt;br&gt;336&lt;br&gt;**0.367 0.386** 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410 0.411 0.408 0.426 0.413 0.413 0.423 0.413&lt;br&gt;720&lt;br&gt;**0.429 0.424** 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456 0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448&lt;br&gt;_Avg._&lt;br&gt;**0.354 0.376** 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|**0.333 **|**0.361**|0.374|0.375|0.383|0.393|0.368|0.382|0.362|0.384|0.382|0.395|0.402|0.440|0.426|0.441|0.553|0.496|0.795|0.669|0.374|0.387|0.363|0.390|0.380|0.389|0.389|0.391|
|ETTm1&lt;br&gt;96&lt;br&gt;**0.288 0.334** 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672 0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373&lt;br&gt;192&lt;br&gt;**0.333 0.361** 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384 0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391&lt;br&gt;336&lt;br&gt;**0.367 0.386** 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410 0.411 0.408 0.426 0.413 0.413 0.423 0.413&lt;br&gt;720&lt;br&gt;**0.429 0.424** 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456 0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448&lt;br&gt;_Avg._&lt;br&gt;**0.354 0.376** 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|**0.367 **|**0.386**|0.409|0.399|0.416|0.414|0.400|0.403|0.392|0.402|0.418|0.418|0.543|0.528|0.445|0.459|0.621|0.537|1.212|0.871|0.410|0.411|0.408|0.426|0.413|0.413|0.423|0.413|
|ETTm1&lt;br&gt;96&lt;br&gt;**0.288 0.334** 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672 0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373&lt;br&gt;192&lt;br&gt;**0.333 0.361** 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384 0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391&lt;br&gt;336&lt;br&gt;**0.367 0.386** 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410 0.411 0.408 0.426 0.413 0.413 0.423 0.413&lt;br&gt;720&lt;br&gt;**0.429 0.424** 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456 0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448&lt;br&gt;_Avg._&lt;br&gt;**0.354 0.376** 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|**0.429 **|**0.424**|0.477|0.438|0.483|0.449|0.460|0.439|0.450|0.435|0.487|0.456|0.704|0.642|0.543|0.490|0.671|0.561|1.166|0.823|0.478|0.450|0.481|0.476|0.474|0.453|0.485|0.448|
|ETTm1&lt;br&gt;96&lt;br&gt;**0.288 0.334** 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672 0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373&lt;br&gt;192&lt;br&gt;**0.333 0.361** 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384 0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391&lt;br&gt;336&lt;br&gt;**0.367 0.386** 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410 0.411 0.408 0.426 0.413 0.413 0.423 0.413&lt;br&gt;720&lt;br&gt;**0.429 0.424** 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456 0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448&lt;br&gt;_Avg._&lt;br&gt;**0.354 0.376** 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|**0.354 **|**0.376**|0.395|0.390|0.410|0.409|0.389|0.397|0.381|0.395|0.407|0.411|0.502|0.502|0.448|0.452|0.588|0.517|0.961|0.734|0.400|0.406|0.392|0.413|0.403|0.407|0.412|0.406|


96 0.288 0.334 0.323 0.349 0.359 0.381 0.329 0.364 0.321 0.360 0.341 0.376 0.360 0.401 0.379 0.419 0.505 0.475 0.672
0.571 0.338 0.375 0.316 0.362 0.345 0.372 0.352 0.373 192 0.333 0.361 0.374 0.375 0.383 0.393 0.368 0.382 0.362 0.384
0.382 0.395 0.402 0.440 0.426 0.441 0.553 0.496 0.795 0.669 0.374 0.387 0.363 0.390 0.380 0.389 0.389 0.391 336 0.367
0.386 0.409 0.399 0.416 0.414 0.400 0.403 0.392 0.402 0.418 0.418 0.543 0.528 0.445 0.459 0.621 0.537 1.212 0.871 0.410
0.411 0.408 0.426 0.413 0.413 0.423 0.413 720 0.429 0.424 0.477 0.438 0.483 0.449 0.460 0.439 0.450 0.435 0.487 0.456
0.704 0.642 0.543 0.490 0.671 0.561 1.166 0.823 0.478 0.450 0.481 0.476 0.474 0.453 0.485 0.448

ETTm1

Avg. 0.354 0.376 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961
0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406

|Avg. 0.354 0.376 0.395 0.390 0.410 0.409 0.389 0.397 0.381 0.395 0.407 0.411 0.502 0.502 0.448 0.452 0.588 0.517 0.961 0.734 0.400 0.406 0.392 0.413 0.403 0.407 0.412 0.406|0.354|0.376|0.395|0.390|0.410|0.409|0.389|0.397|0.381|0.395|0.407|0.411|0.502|0.502|0.448|0.452|0.588|0.517|0.961|0.734|0.400|0.406|0.392|0.413|0.403|0.407|0.412|0.406|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ETTm2&lt;br&gt;96&lt;br&gt;**0.170 0.254** 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365 0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264&lt;br&gt;192&lt;br&gt;**0.224 0.289** 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307 0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304&lt;br&gt;336&lt;br&gt;**0.280 0.327** 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321 0.351 0.325 0.388 0.369 0.427 0.307 0.341&lt;br&gt;720&lt;br&gt;**0.374 0.384** 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406 3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397&lt;br&gt;_Avg._&lt;br&gt;**0.262 0.314** 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410 0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326|**0.170 **|**0.254**|0.178|0.256|0.193|0.280|0.178|0.263|0.178|0.260|0.185|0.272|0.273|0.356|0.203|0.287|0.255|0.339|0.365|0.453|0.187|0.267|0.179|0.275|0.193|0.292|0.181|0.264|
|ETTm2&lt;br&gt;96&lt;br&gt;**0.170 0.254** 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365 0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264&lt;br&gt;192&lt;br&gt;**0.224 0.289** 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307 0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304&lt;br&gt;336&lt;br&gt;**0.280 0.327** 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321 0.351 0.325 0.388 0.369 0.427 0.307 0.341&lt;br&gt;720&lt;br&gt;**0.374 0.384** 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406 3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397&lt;br&gt;_Avg._&lt;br&gt;**0.262 0.314** 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410 0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326|**0.224 **|**0.289**|0.242|0.297|0.257|0.318|0.245|0.306|0.249|0.307|0.253|0.313|0.426|0.487|0.269|0.328|0.249|0.309|0.281|0.340|0.533|0.563|0.307|0.376|0.284|0.362|0.246|0.304|
|ETTm2&lt;br&gt;96&lt;br&gt;**0.170 0.254** 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365 0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264&lt;br&gt;192&lt;br&gt;**0.224 0.289** 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307 0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304&lt;br&gt;336&lt;br&gt;**0.280 0.327** 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321 0.351 0.325 0.388 0.369 0.427 0.307 0.341&lt;br&gt;720&lt;br&gt;**0.374 0.384** 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406 3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397&lt;br&gt;_Avg._&lt;br&gt;**0.262 0.314** 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410 0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326|**0.280 **|**0.327**|0.307|0.339|0.317|0.353|0.309|0.347|0.313|0.346|0.315|0.350|1.013|0.714|0.325|0.366|0.339|0.372|1.363|0.887|0.321|0.351|0.325|0.388|0.369|0.427|0.307|0.341|
|ETTm2&lt;br&gt;96&lt;br&gt;**0.170 0.254** 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365 0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264&lt;br&gt;192&lt;br&gt;**0.224 0.289** 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307 0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304&lt;br&gt;336&lt;br&gt;**0.280 0.327** 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321 0.351 0.325 0.388 0.369 0.427 0.307 0.341&lt;br&gt;720&lt;br&gt;**0.374 0.384** 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406 3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397&lt;br&gt;_Avg._&lt;br&gt;**0.262 0.314** 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410 0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326|**0.374 **|**0.384**|0.397|0.393|0.419|0.411|0.409|0.408|0.400|0.398|0.413|0.406|3.154|1.274|0.421|0.415|0.433|0.432|3.379|1.338|0.408|0.403|0.502|0.490|0.554|0.522|0.407|0.397|
|ETTm2&lt;br&gt;96&lt;br&gt;**0.170 0.254** 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365 0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264&lt;br&gt;192&lt;br&gt;**0.224 0.289** 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307 0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304&lt;br&gt;336&lt;br&gt;**0.280 0.327** 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321 0.351 0.325 0.388 0.369 0.427 0.307 0.341&lt;br&gt;720&lt;br&gt;**0.374 0.384** 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406 3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397&lt;br&gt;_Avg._&lt;br&gt;**0.262 0.314** 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410 0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326|**0.262 **|**0.314**|0.281|0.321|0.296|0.340|0.285|0.331|0.285|0.327|0.291|0.335|1.216|0.707|0.305|0.349|0.327|0.371|1.410|0.810|0.291|0.333|0.328|0.382|0.350|0.401|0.289|0.326|


96 0.170 0.254 0.178 0.256 0.193 0.280 0.178 0.263 0.178 0.260 0.185 0.272 0.273 0.356 0.203 0.287 0.255 0.339 0.365
0.453 0.187 0.267 0.179 0.275 0.193 0.292 0.181 0.264 192 0.224 0.289 0.242 0.297 0.257 0.318 0.245 0.306 0.249 0.307
0.253 0.313 0.426 0.487 0.269 0.328 0.249 0.309 0.281 0.340 0.533 0.563 0.307 0.376 0.284 0.362 0.246 0.304 336 0.280
0.327 0.307 0.339 0.317 0.353 0.309 0.347 0.313 0.346 0.315 0.350 1.013 0.714 0.325 0.366 0.339 0.372 1.363 0.887 0.321
0.351 0.325 0.388 0.369 0.427 0.307 0.341 720 0.374 0.384 0.397 0.393 0.419 0.411 0.409 0.408 0.400 0.398 0.413 0.406
3.154 1.274 0.421 0.415 0.433 0.432 3.379 1.338 0.408 0.403 0.502 0.490 0.554 0.522 0.407 0.397

ETTm2

Avg. 0.262 0.314 0.281 0.321 0.296 0.340 0.285 0.331 0.285 0.327 0.291 0.335 1.216 0.707 0.305 0.349 0.327 0.371 1.410
0.810 0.291 0.333 0.328 0.382 0.350 0.401 0.289 0.326

|96 0.369 0.388 0.369 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865 0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393&lt;br&gt;192 0.418 0.415 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434 0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422 ETTh1&lt;br&gt;336 0.432 0.426 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491 0.469 0.569 0.551 0.481 0.459 0.480 0.445&lt;br&gt;720 0.452 0.451 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493 0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469&lt;br&gt;Avg. 0.418 0.420 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040 0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432|0.369|0.388|0.369|0.389|0.398|0.410|0.376|0.397|0.393|0.408|0.386|0.404|0.420|0.439|0.376|0.419|0.449|0.459|0.865|0.713|0.384|0.402|0.421|0.431|0.386|0.400|0.384|0.393|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ETTh1&lt;br&gt;96&lt;br&gt;**0.369 0.388 0.369** 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865 0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393&lt;br&gt;192&lt;br&gt;**0.418 0.415** 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434 0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422&lt;br&gt;336&lt;br&gt;**0.432 0.426** 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491 0.469 0.569 0.551 0.481 0.459 0.480 0.445&lt;br&gt;720&lt;br&gt;**0.452 0.451** 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493 0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469&lt;br&gt;_Avg._&lt;br&gt;**0.418 0.420** 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040 0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432|**0.418 **|**0.415**|0.427|0.423|0.451|0.440|0.438|0.426|0.445|0.434|0.441|0.436|0.540|0.519|0.420|0.448|0.436|0.429|0.500|0.482|1.008|0.792|0.474|0.487|0.437|0.432|0.436|0.422|
|ETTh1&lt;br&gt;96&lt;br&gt;**0.369 0.388 0.369** 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865 0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393&lt;br&gt;192&lt;br&gt;**0.418 0.415** 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434 0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422&lt;br&gt;336&lt;br&gt;**0.432 0.426** 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491 0.469 0.569 0.551 0.481 0.459 0.480 0.445&lt;br&gt;720&lt;br&gt;**0.452 0.451** 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493 0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469&lt;br&gt;_Avg._&lt;br&gt;**0.418 0.420** 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040 0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432|**0.432 **|**0.426**|0.456|0.436|0.508|0.471|0.479|0.446|0.484|0.451|0.489|0.461|0.722|0.648|0.459|0.465|0.521|0.496|1.107|0.809|0.491|0.469|0.569|0.551|0.481|0.459|0.480|0.445|
|ETTh1&lt;br&gt;96&lt;br&gt;**0.369 0.388 0.369** 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865 0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393&lt;br&gt;192&lt;br&gt;**0.418 0.415** 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434 0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422&lt;br&gt;336&lt;br&gt;**0.432 0.426** 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491 0.469 0.569 0.551 0.481 0.459 0.480 0.445&lt;br&gt;720&lt;br&gt;**0.452 0.451** 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493 0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469&lt;br&gt;_Avg._&lt;br&gt;**0.418 0.420** 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040 0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432|**0.452 **|**0.451**|0.479|0.467|0.483|0.478|0.495|0.476|0.480|0.471|0.508|0.493|0.799|0.685|0.506|0.507|0.514|0.512|1.181|0.865|0.521|0.500|0.770|0.672|0.519|0.516|0.481|0.469|
|ETTh1&lt;br&gt;96&lt;br&gt;**0.369 0.388 0.369** 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865 0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393&lt;br&gt;192&lt;br&gt;**0.418 0.415** 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434 0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422&lt;br&gt;336&lt;br&gt;**0.432 0.426** 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491 0.469 0.569 0.551 0.481 0.459 0.480 0.445&lt;br&gt;720&lt;br&gt;**0.452 0.451** 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493 0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469&lt;br&gt;_Avg._&lt;br&gt;**0.418 0.420** 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040 0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432|**0.418 **|**0.420**|0.432|0.428|0.460|0.449|0.447|0.436|0.450|0.441|0.455|0.448|0.620|0.572|0.440|0.460|0.496|0.487|1.040|0.795|0.458|0.450|0.558|0.535|0.456|0.452|0.445|0.432|


96 0.369 0.388 0.369 0.389 0.398 0.410 0.376 0.397 0.393 0.408 0.386 0.404 0.420 0.439 0.376 0.419 0.449 0.459 0.865
0.713 0.384 0.402 0.421 0.431 0.386 0.400 0.384 0.393 192 0.418 0.415 0.427 0.423 0.451 0.440 0.438 0.426 0.445 0.434
0.441 0.436 0.540 0.519 0.420 0.448 0.436 0.429 0.500 0.482 1.008 0.792 0.474 0.487 0.437 0.432 0.436 0.422 336 0.432
0.426 0.456 0.436 0.508 0.471 0.479 0.446 0.484 0.451 0.489 0.461 0.722 0.648 0.459 0.465 0.521 0.496 1.107 0.809 0.491
0.469 0.569 0.551 0.481 0.459 0.480 0.445 720 0.452 0.451 0.479 0.467 0.483 0.478 0.495 0.476 0.480 0.471 0.508 0.493
0.799 0.685 0.506 0.507 0.514 0.512 1.181 0.865 0.521 0.500 0.770 0.672 0.519 0.516 0.481 0.469

ETTh1

Avg. 0.418 0.420 0.432 0.428 0.460 0.449 0.447 0.436 0.450 0.441 0.455 0.448 0.620 0.572 0.440 0.460 0.496 0.487 1.040
0.795 0.458 0.450 0.558 0.535 0.456 0.452 0.445 0.432

|96 0.279 0.341 0.279 0.331 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755 1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440&lt;br&gt;192 0.356 0.387 0.353 0.380 0.386 0.399 0.386 0.404 0.377 0.393 0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509 ETTh2&lt;br&gt;336 0.350 0.393 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452 0.452 0.654 0.567 0.594 0.541 0.643 0.571&lt;br&gt;720 0.413 0.437 0.404 0.426 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445 1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679&lt;br&gt;Avg. 0.350 0.390 0.349 0.382 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|0.279|0.341|0.279|0.331|0.295|0.346|0.295|0.348|0.294|0.343|0.300|0.349|0.745|0.584|0.358|0.397|0.346|0.388|3.755|1.525|0.340|0.374|0.299|0.364|0.333|0.387|0.400|0.440|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ETTh2&lt;br&gt;96&lt;br&gt;**0.279** 0.341 **0.279 0.331** 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755 1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440&lt;br&gt;192&lt;br&gt;0.356 0.387 **0.353 0.380** 0.386 0.399 0.386 0.404 0.377 0.393 0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509&lt;br&gt;336&lt;br&gt;**0.350 0.393** 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452 0.452 0.654 0.567 0.594 0.541 0.643 0.571&lt;br&gt;720&lt;br&gt;0.413 0.437 **0.404 0.426** 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445 1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679&lt;br&gt;_Avg._&lt;br&gt;0.350 0.390 **0.349 0.382** 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|0.356|0.387|**0.353 **|**0.380**|0.386|0.399|0.386|0.404|0.377|0.393|0.379|0.398|0.877|0.656|0.429|0.439|0.456|0.452|5.602|1.931|0.402|0.414|0.441|0.454|0.477|0.476|0.528|0.509|
|ETTh2&lt;br&gt;96&lt;br&gt;**0.279** 0.341 **0.279 0.331** 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755 1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440&lt;br&gt;192&lt;br&gt;0.356 0.387 **0.353 0.380** 0.386 0.399 0.386 0.404 0.377 0.393 0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509&lt;br&gt;336&lt;br&gt;**0.350 0.393** 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452 0.452 0.654 0.567 0.594 0.541 0.643 0.571&lt;br&gt;720&lt;br&gt;0.413 0.437 **0.404 0.426** 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445 1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679&lt;br&gt;_Avg._&lt;br&gt;0.350 0.390 **0.349 0.382** 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|**0.350 **|**0.393**|0.362|0.394|0.447|0.443|0.421|0.435|0.381|0.409|0.418|0.429|1.043|0.731|0.496|0.487|0.482|0.486|4.721|1.835|0.452|0.452|0.654|0.567|0.594|0.541|0.643|0.571|
|ETTh2&lt;br&gt;96&lt;br&gt;**0.279** 0.341 **0.279 0.331** 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755 1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440&lt;br&gt;192&lt;br&gt;0.356 0.387 **0.353 0.380** 0.386 0.399 0.386 0.404 0.377 0.393 0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509&lt;br&gt;336&lt;br&gt;**0.350 0.393** 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452 0.452 0.654 0.567 0.594 0.541 0.643 0.571&lt;br&gt;720&lt;br&gt;0.413 0.437 **0.404 0.426** 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445 1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679&lt;br&gt;_Avg._&lt;br&gt;0.350 0.390 **0.349 0.382** 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|0.413|0.437|**0.404 **|**0.426**|0.428|0.444|0.422|0.445|0.412|0.433|0.428|0.445|1.104|0.763|0.463|0.474|0.515|0.511|3.647|1.625|0.462|0.468|0.956|0.716|0.831|0.657|0.874|0.679|
|ETTh2&lt;br&gt;96&lt;br&gt;**0.279** 0.341 **0.279 0.331** 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755 1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440&lt;br&gt;192&lt;br&gt;0.356 0.387 **0.353 0.380** 0.386 0.399 0.386 0.404 0.377 0.393 0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509&lt;br&gt;336&lt;br&gt;**0.350 0.393** 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452 0.452 0.654 0.567 0.594 0.541 0.643 0.571&lt;br&gt;720&lt;br&gt;0.413 0.437 **0.404 0.426** 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445 1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679&lt;br&gt;_Avg._&lt;br&gt;0.350 0.390 **0.349 0.382** 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|0.350|0.390|**0.349 **|**0.382**|0.389|0.408|0.381|0.408|0.366|0.394|0.381|0.405|0.942|0.684|0.437|0.449|0.450|0.459|4.431|1.729|0.414|0.427|0.587|0.525|0.559|0.515|0.611|0.550|


96 0.279 0.341 0.279 0.331 0.295 0.346 0.295 0.348 0.294 0.343 0.300 0.349 0.745 0.584 0.358 0.397 0.346 0.388 3.755
1.525 0.340 0.374 0.299 0.364 0.333 0.387 0.400 0.440 192 0.356 0.387 0.353 0.380 0.386 0.399 0.386 0.404 0.377 0.393
0.379 0.398 0.877 0.656 0.429 0.439 0.456 0.452 5.602 1.931 0.402 0.414 0.441 0.454 0.477 0.476 0.528 0.509 336 0.350
0.393 0.362 0.394 0.447 0.443 0.421 0.435 0.381 0.409 0.418 0.429 1.043 0.731 0.496 0.487 0.482 0.486 4.721 1.835 0.452
0.452 0.654 0.567 0.594 0.541 0.643 0.571 720 0.413 0.437 0.404 0.426 0.428 0.444 0.422 0.445 0.412 0.433 0.428 0.445
1.104 0.763 0.463 0.474 0.515 0.511 3.647 1.625 0.462 0.468 0.956 0.716 0.831 0.657 0.874 0.679

ETTh2

Avg. 0.350 0.390 0.349 0.382 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431
1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550

|Avg. 0.350 0.390 0.349 0.382 0.389 0.408 0.381 0.408 0.366 0.394 0.381 0.405 0.942 0.684 0.437 0.449 0.450 0.459 4.431 1.729 0.414 0.427 0.587 0.525 0.559 0.515 0.611 0.550|0.350|0.390|0.349|0.382|0.389|0.408|0.381|0.408|0.366|0.394|0.381|0.405|0.942|0.684|0.437|0.449|0.450|0.459|4.431|1.729|0.414|0.427|0.587|0.525|0.559|0.515|0.611|0.550|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Weather&lt;br&gt;96&lt;br&gt;**0.157** 0.205 0.164 **0.204** 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300 0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261&lt;br&gt;192&lt;br&gt;**0.202 0.245** 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259 0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298&lt;br&gt;336&lt;br&gt;**0.255 0.286** 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280 0.306 0.278 0.331 0.283 0.335 0.287 0.335&lt;br&gt;720&lt;br&gt;0.336 **0.338** 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349 0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 **0.311** 0.356 0.345 0.381 0.351 0.386&lt;br&gt;_Avg._&lt;br&gt;**0.238 0.269** 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|**0.157**|0.205|0.164|**0.204**|0.195|0.233|0.182|0.223|0.177|0.218|0.174|0.214|0.158|0.230|0.217|0.296|0.266|0.336|0.300|0.384|0.172|0.220|0.161|0.229|0.196|0.255|0.202|0.261|
|Weather&lt;br&gt;96&lt;br&gt;**0.157** 0.205 0.164 **0.204** 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300 0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261&lt;br&gt;192&lt;br&gt;**0.202 0.245** 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259 0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298&lt;br&gt;336&lt;br&gt;**0.255 0.286** 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280 0.306 0.278 0.331 0.283 0.335 0.287 0.335&lt;br&gt;720&lt;br&gt;0.336 **0.338** 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349 0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 **0.311** 0.356 0.345 0.381 0.351 0.386&lt;br&gt;_Avg._&lt;br&gt;**0.238 0.269** 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|**0.202 **|**0.245**|0.214|0.250|0.240|0.269|0.231|0.263|0.225|0.259|0.221|0.254|0.206|0.277|0.276|0.336|0.307|0.367|0.598|0.544|0.219|0.261|0.220|0.281|0.237|0.296|0.242|0.298|
|Weather&lt;br&gt;96&lt;br&gt;**0.157** 0.205 0.164 **0.204** 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300 0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261&lt;br&gt;192&lt;br&gt;**0.202 0.245** 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259 0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298&lt;br&gt;336&lt;br&gt;**0.255 0.286** 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280 0.306 0.278 0.331 0.283 0.335 0.287 0.335&lt;br&gt;720&lt;br&gt;0.336 **0.338** 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349 0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 **0.311** 0.356 0.345 0.381 0.351 0.386&lt;br&gt;_Avg._&lt;br&gt;**0.238 0.269** 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|**0.255 **|**0.286**|0.269|0.291|0.293|0.306|0.283|0.300|0.278|0.297|0.278|0.296|0.272|0.335|0.339|0.380|0.359|0.395|0.578|0.523|0.280|0.306|0.278|0.331|0.283|0.335|0.287|0.335|
|Weather&lt;br&gt;96&lt;br&gt;**0.157** 0.205 0.164 **0.204** 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300 0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261&lt;br&gt;192&lt;br&gt;**0.202 0.245** 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259 0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298&lt;br&gt;336&lt;br&gt;**0.255 0.286** 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280 0.306 0.278 0.331 0.283 0.335 0.287 0.335&lt;br&gt;720&lt;br&gt;0.336 **0.338** 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349 0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 **0.311** 0.356 0.345 0.381 0.351 0.386&lt;br&gt;_Avg._&lt;br&gt;**0.238 0.269** 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|0.336|**0.338**|0.355|0.352|0.368|0.354|0.360|0.350|0.354|0.348|0.358|0.349|0.398|0.418|0.403|0.428|0.419|0.428|1.059|0.741|0.365|0.359|**0.311**|0.356|0.345|0.381|0.351|0.386|
|Weather&lt;br&gt;96&lt;br&gt;**0.157** 0.205 0.164 **0.204** 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300 0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261&lt;br&gt;192&lt;br&gt;**0.202 0.245** 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259 0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298&lt;br&gt;336&lt;br&gt;**0.255 0.286** 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280 0.306 0.278 0.331 0.283 0.335 0.287 0.335&lt;br&gt;720&lt;br&gt;0.336 **0.338** 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349 0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 **0.311** 0.356 0.345 0.381 0.351 0.386&lt;br&gt;_Avg._&lt;br&gt;**0.238 0.269** 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|**0.238 **|**0.269**|0.250|0.274|0.274|0.290|0.264|0.284|0.258|0.280|0.257|0.279|0.259|0.315|0.309|0.360|0.338|0.382|0.634|0.548|0.259|0.287|0.242|0.299|0.265|0.317|0.271|0.320|


96 0.157 0.205 0.164 0.204 0.195 0.233 0.182 0.223 0.177 0.218 0.174 0.214 0.158 0.230 0.217 0.296 0.266 0.336 0.300
0.384 0.172 0.220 0.161 0.229 0.196 0.255 0.202 0.261 192 0.202 0.245 0.214 0.250 0.240 0.269 0.231 0.263 0.225 0.259
0.221 0.254 0.206 0.277 0.276 0.336 0.307 0.367 0.598 0.544 0.219 0.261 0.220 0.281 0.237 0.296 0.242 0.298 336 0.255
0.286 0.269 0.291 0.293 0.306 0.283 0.300 0.278 0.297 0.278 0.296 0.272 0.335 0.339 0.380 0.359 0.395 0.578 0.523 0.280
0.306 0.278 0.331 0.283 0.335 0.287 0.335 720 0.336 0.338 0.355 0.352 0.368 0.354 0.360 0.350 0.354 0.348 0.358 0.349
0.398 0.418 0.403 0.428 0.419 0.428 1.059 0.741 0.365 0.359 0.311 0.356 0.345 0.381 0.351 0.386

Weather

Avg. 0.238 0.269 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634
0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320

|Avg. 0.238 0.269 0.250 0.274 0.274 0.290 0.264 0.284 0.258 0.280 0.257 0.279 0.259 0.315 0.309 0.360 0.338 0.382 0.634 0.548 0.259 0.287 0.242 0.299 0.265 0.317 0.271 0.320|0.238|0.269|0.250|0.274|0.274|0.290|0.264|0.284|0.258|0.280|0.257|0.279|0.259|0.315|0.309|0.360|0.338|0.382|0.634|0.548|0.259|0.287|0.242|0.299|0.265|0.317|0.271|0.320|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Electricity&lt;br&gt;96&lt;br&gt;**0.131 0.222** 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274 0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329&lt;br&gt;192&lt;br&gt;**0.151 0.240** 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289 0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330&lt;br&gt;336&lt;br&gt;**0.162 0.256** 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198 0.300 0.193 0.304 0.209 0.301 0.249 0.344&lt;br&gt;720&lt;br&gt;0.213 **0.297** 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317 0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 **0.212** 0.321 0.245 0.333 0.284 0.373&lt;br&gt;_Avg._&lt;br&gt;**0.164 0.254** 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311 0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344|**0.131 **|**0.222**|0.145|0.238|0.204|0.293|0.185|0.272|0.195|0.285|0.148|0.240|0.219|0.314|0.193|0.308|0.201|0.317|0.274|0.368|0.168|0.272|0.164|0.269|0.197|0.282|0.237|0.329|
|Electricity&lt;br&gt;96&lt;br&gt;**0.131 0.222** 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274 0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329&lt;br&gt;192&lt;br&gt;**0.151 0.240** 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289 0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330&lt;br&gt;336&lt;br&gt;**0.162 0.256** 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198 0.300 0.193 0.304 0.209 0.301 0.249 0.344&lt;br&gt;720&lt;br&gt;0.213 **0.297** 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317 0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 **0.212** 0.321 0.245 0.333 0.284 0.373&lt;br&gt;_Avg._&lt;br&gt;**0.164 0.254** 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311 0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344|**0.151 **|**0.240**|0.161|0.252|0.207|0.295|0.189|0.276|0.199|0.289|0.162|0.253|0.231|0.322|0.201|0.315|0.222|0.334|0.296|0.386|0.184|0.289|0.177|0.285|0.196|0.285|0.236|0.330|
|Electricity&lt;br&gt;96&lt;br&gt;**0.131 0.222** 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274 0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329&lt;br&gt;192&lt;br&gt;**0.151 0.240** 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289 0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330&lt;br&gt;336&lt;br&gt;**0.162 0.256** 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198 0.300 0.193 0.304 0.209 0.301 0.249 0.344&lt;br&gt;720&lt;br&gt;0.213 **0.297** 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317 0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 **0.212** 0.321 0.245 0.333 0.284 0.373&lt;br&gt;_Avg._&lt;br&gt;**0.164 0.254** 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311 0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344|**0.162 **|**0.256**|0.175|0.267|0.219|0.308|0.204|0.291|0.215|0.305|0.178|0.269|0.246|0.337|0.214|0.329|0.231|0.338|0.300|0.394|0.198|0.300|0.193|0.304|0.209|0.301|0.249|0.344|
|Electricity&lt;br&gt;96&lt;br&gt;**0.131 0.222** 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274 0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329&lt;br&gt;192&lt;br&gt;**0.151 0.240** 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289 0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330&lt;br&gt;336&lt;br&gt;**0.162 0.256** 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198 0.300 0.193 0.304 0.209 0.301 0.249 0.344&lt;br&gt;720&lt;br&gt;0.213 **0.297** 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317 0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 **0.212** 0.321 0.245 0.333 0.284 0.373&lt;br&gt;_Avg._&lt;br&gt;**0.164 0.254** 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311 0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344|0.213|**0.297**|0.222|0.303|0.263|0.341|0.245|0.324|0.256|0.337|0.225|0.317|0.280|0.363|0.246|0.355|0.254|0.361|0.373|0.439|0.220|0.320|**0.212**|0.321|0.245|0.333|0.284|0.373|
|Electricity&lt;br&gt;96&lt;br&gt;**0.131 0.222** 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274 0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329&lt;br&gt;192&lt;br&gt;**0.151 0.240** 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289 0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330&lt;br&gt;336&lt;br&gt;**0.162 0.256** 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198 0.300 0.193 0.304 0.209 0.301 0.249 0.344&lt;br&gt;720&lt;br&gt;0.213 **0.297** 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317 0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 **0.212** 0.321 0.245 0.333 0.284 0.373&lt;br&gt;_Avg._&lt;br&gt;**0.164 0.254** 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311 0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344|**0.164 **|**0.254**|0.175|0.265|0.223|0.309|0.205|0.290|0.216|0.304|0.178|0.270|0.244|0.334|0.214|0.327|0.227|0.338|0.311|0.397|0.192|0.295|0.186|0.294|0.212|0.300|0.251|0.344|


96 0.131 0.222 0.145 0.238 0.204 0.293 0.185 0.272 0.195 0.285 0.148 0.240 0.219 0.314 0.193 0.308 0.201 0.317 0.274
0.368 0.168 0.272 0.164 0.269 0.197 0.282 0.237 0.329 192 0.151 0.240 0.161 0.252 0.207 0.295 0.189 0.276 0.199 0.289
0.162 0.253 0.231 0.322 0.201 0.315 0.222 0.334 0.296 0.386 0.184 0.289 0.177 0.285 0.196 0.285 0.236 0.330 336 0.162
0.256 0.175 0.267 0.219 0.308 0.204 0.291 0.215 0.305 0.178 0.269 0.246 0.337 0.214 0.329 0.231 0.338 0.300 0.394 0.198
0.300 0.193 0.304 0.209 0.301 0.249 0.344 720 0.213 0.297 0.222 0.303 0.263 0.341 0.245 0.324 0.256 0.337 0.225 0.317
0.280 0.363 0.246 0.355 0.254 0.361 0.373 0.439 0.220 0.320 0.212 0.321 0.245 0.333 0.284 0.373

Electricity

Avg. 0.164 0.254 0.175 0.265 0.223 0.309 0.205 0.290 0.216 0.304 0.178 0.270 0.244 0.334 0.214 0.327 0.227 0.338 0.311
0.397 0.192 0.295 0.186 0.294 0.212 0.300 0.251 0.344

|96 0.392 0.267 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719 0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493&lt;br&gt;192 0.413 0.265 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354 0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474 Traffic&lt;br&gt;336 0.440 0.282 0.444 0.281 0.530 0.349 0.488 0.317 0.551 0.358 0.433 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629 0.336 0.534 0.313 0.605 0.373 0.762 0.477&lt;br&gt;720 0.464 0.300 0.477 0.300 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302 0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449&lt;br&gt;Avg. 0.427 0.279 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|0.392|0.267|0.407|0.268|0.536|0.359|0.468|0.307|0.544|0.359|0.395|0.268|0.522|0.290|0.587|0.366|0.613|0.388|0.719|0.391|0.593|0.321|0.519|0.309|0.650|0.396|0.805|0.493|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Traffic&lt;br&gt;96&lt;br&gt;**0.392 0.267** 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719 0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493&lt;br&gt;192&lt;br&gt;**0.413 0.265** 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354 0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474&lt;br&gt;336&lt;br&gt;0.440 0.282 0.444 **0.281** 0.530 0.349 0.488 0.317 0.551 0.358 **0.433** 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629 0.336 0.534 0.313 0.605 0.373 0.762 0.477&lt;br&gt;720&lt;br&gt;**0.464 0.300** 0.477 **0.300** 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302 0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449&lt;br&gt;_Avg._&lt;br&gt;**0.427 0.279** 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|**0.413 **|**0.265**|0.430|0.278|0.530|0.354|0.476|0.311|0.540|0.354|0.417|0.276|0.530|0.293|0.604|0.373|0.616|0.382|0.696|0.379|0.617|0.336|0.537|0.315|0.598|0.370|0.756|0.474|
|Traffic&lt;br&gt;96&lt;br&gt;**0.392 0.267** 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719 0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493&lt;br&gt;192&lt;br&gt;**0.413 0.265** 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354 0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474&lt;br&gt;336&lt;br&gt;0.440 0.282 0.444 **0.281** 0.530 0.349 0.488 0.317 0.551 0.358 **0.433** 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629 0.336 0.534 0.313 0.605 0.373 0.762 0.477&lt;br&gt;720&lt;br&gt;**0.464 0.300** 0.477 **0.300** 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302 0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449&lt;br&gt;_Avg._&lt;br&gt;**0.427 0.279** 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|0.440|0.282|0.444|**0.281**|0.530|0.349|0.488|0.317|0.551|0.358|**0.433**|0.283|0.558|0.305|0.621|0.383|0.622|0.337|0.777|0.420|0.629|0.336|0.534|0.313|0.605|0.373|0.762|0.477|
|Traffic&lt;br&gt;96&lt;br&gt;**0.392 0.267** 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719 0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493&lt;br&gt;192&lt;br&gt;**0.413 0.265** 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354 0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474&lt;br&gt;336&lt;br&gt;0.440 0.282 0.444 **0.281** 0.530 0.349 0.488 0.317 0.551 0.358 **0.433** 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629 0.336 0.534 0.313 0.605 0.373 0.762 0.477&lt;br&gt;720&lt;br&gt;**0.464 0.300** 0.477 **0.300** 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302 0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449&lt;br&gt;_Avg._&lt;br&gt;**0.427 0.279** 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|**0.464 **|**0.300**|0.477|**0.300**|0.569|0.371|0.521|0.333|0.586|0.375|0.467|0.302|0.589|0.328|0.626|0.382|0.660|0.408|0.864|0.472|0.640|0.350|0.577|0.325|0.645|0.394|0.719|0.449|
|Traffic&lt;br&gt;96&lt;br&gt;**0.392 0.267** 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719 0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493&lt;br&gt;192&lt;br&gt;**0.413 0.265** 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354 0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474&lt;br&gt;336&lt;br&gt;0.440 0.282 0.444 **0.281** 0.530 0.349 0.488 0.317 0.551 0.358 **0.433** 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629 0.336 0.534 0.313 0.605 0.373 0.762 0.477&lt;br&gt;720&lt;br&gt;**0.464 0.300** 0.477 **0.300** 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302 0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449&lt;br&gt;_Avg._&lt;br&gt;**0.427 0.279** 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|**0.427 **|**0.279**|0.439|0.281|0.541|0.358|0.488|0.317|0.555|0.361|0.428|0.282|0.550|0.304|0.610|0.376|0.628|0.379|0.764|0.416|0.620|0.336|0.541|0.315|0.625|0.383|0.760|0.473|


96 0.392 0.267 0.407 0.268 0.536 0.359 0.468 0.307 0.544 0.359 0.395 0.268 0.522 0.290 0.587 0.366 0.613 0.388 0.719
0.391 0.593 0.321 0.519 0.309 0.650 0.396 0.805 0.493 192 0.413 0.265 0.430 0.278 0.530 0.354 0.476 0.311 0.540 0.354
0.417 0.276 0.530 0.293 0.604 0.373 0.616 0.382 0.696 0.379 0.617 0.336 0.537 0.315 0.598 0.370 0.756 0.474 336 0.440
0.282 0.444 0.281 0.530 0.349 0.488 0.317 0.551 0.358 0.433 0.283 0.558 0.305 0.621 0.383 0.622 0.337 0.777 0.420 0.629
0.336 0.534 0.313 0.605 0.373 0.762 0.477 720 0.464 0.300 0.477 0.300 0.569 0.371 0.521 0.333 0.586 0.375 0.467 0.302
0.589 0.328 0.626 0.382 0.660 0.408 0.864 0.472 0.640 0.350 0.577 0.325 0.645 0.394 0.719 0.449

Traffic

Avg. 0.427 0.279 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764
0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473

|Avg. 0.427 0.279 0.439 0.281 0.541 0.358 0.488 0.317 0.555 0.361 0.428 0.282 0.550 0.304 0.610 0.376 0.628 0.379 0.764 0.416 0.620 0.336 0.541 0.315 0.625 0.383 0.760 0.473|0.427|0.279|0.439|0.281|0.541|0.358|0.488|0.317|0.555|0.361|0.428|0.282|0.550|0.304|0.610|0.376|0.628|0.379|0.764|0.416|0.620|0.336|0.541|0.315|0.625|0.383|0.760|0.473|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**1.630 **|**0.798**|1.672|0.841|1.651|0.841|1.869|0.823|2.221|0.883|2.321|0.937|3.449|1.238|2.721|1.133|3.280|1.265|5.280|1.578|1.826|0.893|2.715|1.125|5.060|1.709|5.855|1.633|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**1.650 **|**0.821**|1.725|0.872|1.701|0.861|1.853|0.854|2.313|0.904|2.188|0.945|3.743|1.271|2.768|1.118|3.424|1.271|5.094|1.565|2.678|0.986|2.817|1.154|4.413|1.549|5.598|1.715|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**1.810**|0.921|1.937|0.937|2.153|1.041|1.886|**0.855**|2.048|0.886|2.231|0.956|3.853|1.306|2.637|1.088|3.009|1.520|4.884|1.530|2.584|0.937|3.038|1.199|4.109|1.473|4.795|1.568|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**1.850 **|**0.874**|2.128|0.999|2.064|0.953|1.877|0.877|2.008|0.915|2.292|0.991|3.951|1.323|2.696|1.050|2.803|1.133|5.326|1.571|1.980|0.894|3.372|1.269|4.233|1.481|4.616|1.543|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**1.735**|0.854|1.861|0.924|1.829|0.924|1.871|**0.852**|2.145|0.897|2.258|0.957|3.749|1.284|2.705|1.097|3.129|1.297|5.123|1.561|2.267|0.927|2.985|1.186|4.453|1.553|5.216|1.614|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**0.122 **|**0.173**|0.185|0.175|0.143|0.179|0.212|0.198|0.159|0.195|0.210|0.212|0.143|0.194|0.214|0.223|0.163|0.183|0.201|0.238|0.261|0.270|0.276|0.264|0.264|0.270|0.255|0.256|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**0.196 **|**0.223**|0.235|0.234|0.234|0.245|0.241|0.236|0.237|0.259|0.244|0.255|0.214|0.246|0.237|0.263|0.217|0.237|0.246|0.243|0.283|0.293|0.292|0.306|0.279|0.295|0.273|0.291|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**0.261 **|**0.277**|0.286|0.290|0.288|0.287|0.267|0.270|0.278|0.301|0.267|0.286|0.284|0.286|**0.261**|0.289|0.284|0.295|0.305|0.310|0.295|0.318|0.300|0.325|0.296|0.317|0.305|0.322|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|0.314|**0.327**|0.326|0.341|0.335|0.347|0.329|0.338|0.339|0.356|**0.308**|0.332|0.338|0.350|0.311|0.342|0.334|0.341|0.359|0.348|0.325|0.342|0.354|0.363|0.328|0.348|0.334|0.359|
|ILI&lt;br&gt;24&lt;br&gt;**1.630 0.798** 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280 1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633&lt;br&gt;36&lt;br&gt;**1.650 0.821** 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904 2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715&lt;br&gt;48&lt;br&gt;**1.810** 0.921 1.937 0.937 2.153 1.041 1.886 **0.855** 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584 0.937 3.038 1.199 4.109 1.473 4.795 1.568&lt;br&gt;60&lt;br&gt;**1.850 0.874** 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991 3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543&lt;br&gt;_Avg._&lt;br&gt;**1.735** 0.854 1.861 0.924 1.829 0.924 1.871 **0.852** 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123 1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614&lt;br&gt;ECG&lt;br&gt;96&lt;br&gt;**0.122 0.173** 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201 0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256&lt;br&gt;192&lt;br&gt;**0.196 0.223** 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259 0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291&lt;br&gt;336&lt;br&gt;**0.261 0.277** 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 **0.261** 0.289 0.284 0.295 0.305 0.310 0.295 0.318 0.300 0.325 0.296 0.317 0.305 0.322&lt;br&gt;720&lt;br&gt;0.314 **0.327** 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 **0.308** 0.332 0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359&lt;br&gt;_Avg._&lt;br&gt;**0.225 0.250** 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277 0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307&lt;br&gt;1st _Count_&lt;br&gt;**73**&lt;br&gt;12&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;1&lt;br&gt;0&lt;br&gt;0&lt;br&gt;0&lt;br&gt;2&lt;br&gt;0&lt;br&gt;0|**0.225 **|**0.250**|0.258|0.260|0.250|0.264|0.262|0.260|0.253|0.277|0.257|0.271|0.244|0.269|0.255|0.279|0.249|0.264|0.277|0.284|0.291|0.305|0.305|0.314|0.291|0.307|0.291|0.307|


24 1.630 0.798 1.672 0.841 1.651 0.841 1.869 0.823 2.221 0.883 2.321 0.937 3.449 1.238 2.721 1.133 3.280 1.265 5.280
1.578 1.826 0.893 2.715 1.125 5.060 1.709 5.855 1.633 36 1.650 0.821 1.725 0.872 1.701 0.861 1.853 0.854 2.313 0.904
2.188 0.945 3.743 1.271 2.768 1.118 3.424 1.271 5.094 1.565 2.678 0.986 2.817 1.154 4.413 1.549 5.598 1.715 48 1.810
0.921 1.937 0.937 2.153 1.041 1.886 0.855 2.048 0.886 2.231 0.956 3.853 1.306 2.637 1.088 3.009 1.520 4.884 1.530 2.584
0.937 3.038 1.199 4.109 1.473 4.795 1.568 60 1.850 0.874 2.128 0.999 2.064 0.953 1.877 0.877 2.008 0.915 2.292 0.991
3.951 1.323 2.696 1.050 2.803 1.133 5.326 1.571 1.980 0.894 3.372 1.269 4.233 1.481 4.616 1.543

ILI

Avg. 1.735 0.854 1.861 0.924 1.829 0.924 1.871 0.852 2.145 0.897 2.258 0.957 3.749 1.284 2.705 1.097 3.129 1.297 5.123
1.561 2.267 0.927 2.985 1.186 4.453 1.553 5.216 1.614

96 0.122 0.173 0.185 0.175 0.143 0.179 0.212 0.198 0.159 0.195 0.210 0.212 0.143 0.194 0.214 0.223 0.163 0.183 0.201
0.238 0.261 0.270 0.276 0.264 0.264 0.270 0.255 0.256 192 0.196 0.223 0.235 0.234 0.234 0.245 0.241 0.236 0.237 0.259
0.244 0.255 0.214 0.246 0.237 0.263 0.217 0.237 0.246 0.243 0.283 0.293 0.292 0.306 0.279 0.295 0.273 0.291 336 0.261
0.277 0.286 0.290 0.288 0.287 0.267 0.270 0.278 0.301 0.267 0.286 0.284 0.286 0.261 0.289 0.284 0.295 0.305 0.310 0.295
0.318 0.300 0.325 0.296 0.317 0.305 0.322 720 0.314 0.327 0.326 0.341 0.335 0.347 0.329 0.338 0.339 0.356 0.308 0.332
0.338 0.350 0.311 0.342 0.334 0.341 0.359 0.348 0.325 0.342 0.354 0.363 0.328 0.348 0.334 0.359

ECG

Avg. 0.225 0.250 0.258 0.260 0.250 0.264 0.262 0.260 0.253 0.277 0.257 0.271 0.244 0.269 0.255 0.279 0.249 0.264 0.277
0.284 0.291 0.305 0.305 0.314 0.291 0.307 0.291 0.307

1st Count 73 12 0 2 0 2 0 1 0 0 0 2 0 0

#### A.1. Baseline Time Series Forecasting Methods

In this paper, we compare an extensive range of SOTA methods, primarily categorized as follows:

#### 1) LLM-based Methods:

• GPT4TS (Zhou et al., 2023b), the pioneering work that employs LLMs for time series forecasting by segmenting
continuous time series into discrete tokens compatible with LLMs. • TimeLLM (Jin et al., 2024), which proposes patch
reprogramming to encode prior knowledge from time series datasets

12

|0.409 0.411&lt;br&gt;0.468 0.440&lt;br&gt;0.527 0.475&lt;br&gt;0.584 0.491|0.468 0.445&lt;br&gt;0.479 0.446&lt;br&gt;0.499 0.463&lt;br&gt;0.572 0.496|0.587 0.491&lt;br&gt;0.606 0.490&lt;br&gt;0.719 0.555&lt;br&gt;0.632 0.514|0.615 0.497&lt;br&gt;0.597 0.492&lt;br&gt;0.597 0.501&lt;br&gt;0.623 0.513|0.558 0.478&lt;br&gt;0.539 0.471&lt;br&gt;0.558 0.488&lt;br&gt;0.574 0.498|1.037 0.705&lt;br&gt;1.170 0.778&lt;br&gt;1.463 0.913&lt;br&gt;1.693 0.997|0.604 0.530&lt;br&gt;0.641 0.546&lt;br&gt;0.768 0.606&lt;br&gt;0.771 0.606|0.583 0.503&lt;br&gt;0.608 0.515&lt;br&gt;0.733 0.572&lt;br&gt;0.768 0.548|0.677 0.585&lt;br&gt;0.784 0.627&lt;br&gt;0.972 0.684&lt;br&gt;1.449 0.800|0.552 0.488&lt;br&gt;0.546 0.487&lt;br&gt;0.567 0.501&lt;br&gt;0.606 0.522|
|---|---|---|---|---|---|---|---|---|---|


Table 8: The full results for few-shot forecasting using only 10% of the training data from the ETT datasets, where the
prediction lengths H ∈{96, 192, 336, 720}. The term “Avg.” reports the average result obtained from all four prediction
lengths. The best and second best results are highlighted in bold and underlined, respectively. The term “1st Count”
indicates the number of times each method achieves the best results.

Models LLM-PS CALF TimeLLM GPT4TS PatchTST Crossformer FEDformer TimesNet MICN DLinear TiDE Ours (2024a) (2024) (2023b)
(2023) (2023a) (2022) (2023a) (2022) (2023) (2023)

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

Title Suppressed Due to Excessive Size

ETTm1

Avg. 0.497 0.454 0.504 0.462 0.636 0.512 0.608 0.500 0.557 0.483 1.340 0.848 0.696 0.572 0.673 0.534 0.970 0.674 0.567
0.499 0.515 0.469

|0.186 0.263&lt;br&gt;0.239 0.297&lt;br&gt;0.308 0.344&lt;br&gt;0.389 0.390|0.190 0.268&lt;br&gt;0.257 0.311&lt;br&gt;0.323 0.334&lt;br&gt;0.441 0.410|0.189 0.270&lt;br&gt;0.264 0.319&lt;br&gt;0.327 0.358&lt;br&gt;0.454 0.428|0.187 0.266&lt;br&gt;0.253 0.308&lt;br&gt;0.332 0.353&lt;br&gt;0.438 0.417|0.189 0.268&lt;br&gt;0.248 0.307&lt;br&gt;0.311 0.346&lt;br&gt;0.435 0.418|1.397 0.866&lt;br&gt;1.757 0.987&lt;br&gt;2.075 1.086&lt;br&gt;2.712 1.253|0.222 0.314&lt;br&gt;0.284 0.351&lt;br&gt;0.392 0.419&lt;br&gt;0.527 0.485|0.214 0.288&lt;br&gt;0.271 0.325&lt;br&gt;0.329 0.356&lt;br&gt;0.473 0.448|0.389 0.448&lt;br&gt;0.622 0.575&lt;br&gt;1.055 0.755&lt;br&gt;2.226 1.087|0.225 0.320&lt;br&gt;0.291 0.362&lt;br&gt;0.354 0.402&lt;br&gt;0.446 0.447|
|---|---|---|---|---|---|---|---|---|---|


96 0.186 0.263 0.190 0.268 0.189 0.270 0.187 0.266 0.189 0.268 1.397 0.866 0.222 0.314 0.214 0.288 0.389 0.448 0.225
0.320 0.191 0.269 192 0.239 0.297 0.257 0.311 0.264 0.319 0.253 0.308 0.248 0.307 1.757 0.987 0.284 0.351 0.271 0.325
0.622 0.575 0.291 0.362 0.256 0.310 336 0.308 0.344 0.323 0.334 0.327 0.358 0.332 0.353 0.311 0.346 2.075 1.086 0.392
0.419 0.329 0.356 1.055 0.755 0.354 0.402 0.321 0.349 720 0.389 0.390 0.441 0.410 0.454 0.428 0.438 0.417 0.435 0.418
2.712 1.253 0.527 0.485 0.473 0.448 2.226 1.087 0.446 0.447 0.446 0.421

ETTm2

Avg. 0.281 0.324 0.302 0.330 0.308 0.343 0.303 0.336 0.295 0.334 1.985 1.048 0.356 0.392 0.321 0.354 1.073 0.716 0.329
0.382 0.303 0.337

|0.586 0.529&lt;br&gt;0.620 0.537&lt;br&gt;0.658 0.553&lt;br&gt;0.664 0.563|0.468 0.457&lt;br&gt;0.550 0.501&lt;br&gt;0.581 0.521&lt;br&gt;0.978 0.685|0.500 0.464&lt;br&gt;0.590 0.516&lt;br&gt;0.638 0.542&lt;br&gt;1.334 0.816|0.462 0.449&lt;br&gt;0.551 0.495&lt;br&gt;0.630 0.539&lt;br&gt;1.113 0.738|0.433 0.428&lt;br&gt;0.509 0.474&lt;br&gt;0.572 0.509&lt;br&gt;1.221 0.773|1.129 0.775&lt;br&gt;1.832 0.922&lt;br&gt;2.022 0.973&lt;br&gt;1.903 0.986|0.651 0.563&lt;br&gt;0.666 0.562&lt;br&gt;0.767 0.602&lt;br&gt;0.918 0.703|0.855 0.625&lt;br&gt;0.791 0.589&lt;br&gt;0.939 0.648&lt;br&gt;0.876 0.641|0.689 0.592&lt;br&gt;1.160 0.748&lt;br&gt;1.747 0.899&lt;br&gt;2.024 1.019|0.590 0.515&lt;br&gt;0.634 0.541&lt;br&gt;0.659 0.554&lt;br&gt;0.708 0.598|
|---|---|---|---|---|---|---|---|---|---|


96 0.586 0.529 0.468 0.457 0.500 0.464 0.462 0.449 0.433 0.428 1.129 0.775 0.651 0.563 0.855 0.625 0.689 0.592 0.590
0.515 0.642 0.545 192 0.620 0.537 0.550 0.501 0.590 0.516 0.551 0.495 0.509 0.474 1.832 0.922 0.666 0.562 0.791 0.589
1.160 0.748 0.634 0.541 0.761 0.595 336 0.658 0.553 0.581 0.521 0.638 0.542 0.630 0.539 0.572 0.509 2.022 0.973 0.767
0.602 0.939 0.648 1.747 0.899 0.659 0.554 0.789 0.610 720 0.664 0.563 0.978 0.685 1.334 0.816 1.113 0.738 1.221 0.773
1.903 0.986 0.918 0.703 0.876 0.641 2.024 1.019 0.708 0.598 0.927 0.667

ETTh1

Avg. 0.632 0.546 0.644 0.541 0.765 0.584 0.689 0.555 0.683 0.645 1.744 0.914 0.750 0.607 0.865 0.625 1.405 0.814 0.647
0.552 0.779 0.604

|0.332 0.372&lt;br&gt;0.398 0.412&lt;br&gt;0.430 0.431&lt;br&gt;0.476 0.463|0.314 0.360&lt;br&gt;0.404 0.411&lt;br&gt;0.458 0.452&lt;br&gt;0.502 0.487|0.329 0.365&lt;br&gt;0.414 0.413&lt;br&gt;0.579 0.506&lt;br&gt;1.034 0.711|0.327 0.359&lt;br&gt;0.403 0.405&lt;br&gt;0.568 0.499&lt;br&gt;1.020 0.725|0.314 0.354&lt;br&gt;0.420 0.415&lt;br&gt;0.543 0.489&lt;br&gt;0.926 0.691|2.482 1.206&lt;br&gt;3.136 1.372&lt;br&gt;2.925 1.331&lt;br&gt;4.014 1.603|0.359 0.404&lt;br&gt;0.460 0.461&lt;br&gt;0.569 0.530&lt;br&gt;0.827 0.707|0.372 0.405&lt;br&gt;0.483 0.463&lt;br&gt;0.541 0.496&lt;br&gt;0.510 0.491|0.510 0.502&lt;br&gt;1.809 1.036&lt;br&gt;3.250 1.419&lt;br&gt;4.564 1.676|0.361 0.407&lt;br&gt;0.444 0.453&lt;br&gt;0.509 0.501&lt;br&gt;0.453 0.471|
|---|---|---|---|---|---|---|---|---|---|


96 0.332 0.372 0.314 0.360 0.329 0.365 0.327 0.359 0.314 0.354 2.482 1.206 0.359 0.404 0.372 0.405 0.510 0.502 0.361
0.407 0.337 0.379 192 0.398 0.412 0.404 0.411 0.414 0.413 0.403 0.405 0.420 0.415 3.136 1.372 0.460 0.461 0.483 0.463
1.809 1.036 0.444 0.453 0.424 0.427 336 0.430 0.431 0.458 0.452 0.579 0.506 0.568 0.499 0.543 0.489 2.925 1.331 0.569
0.530 0.541 0.496 3.250 1.419 0.509 0.501 0.435 0.426 720 0.476 0.463 0.502 0.487 1.034 0.711 1.020 0.725 0.926 0.691
4.014 1.603 0.827 0.707 0.510 0.491 4.564 1.676 0.453 0.471 0.489 0.480

ETTh2

Avg. 0.409 0.420 0.419 0.427 0.589 0.498 0.579 0.497 0.550 0.487 3.139 1.378 0.553 0.525 0.476 0.463 2.533 1.158 0.441
0.458 0.421 0.428

1st Count 23 5 0 1 8 0 0 0 0 1 3

into prompts for guiding the LLM in time series forecasting.

• CALF (Liu et al., 2024a) trains separate branches for temporal and textual modalities and closely aligns them with
leveraging textual knowledge in LLMs for time series prediction.

#### 2) Transformer-based Methods:

• Crossformer (Zhang & Yan, 2023a) identifies that relationships between variables in time series data are crucial for
time series forecasting, so it captures them using attention mechanisms.

• FEDformer (Zhou et al., 2022) and Autoformer (Wu et al., 2021), which decouple seasonal and trend components in the
frequency domain and learn them based on attention mechanisms.

• PatchTST (Nie et al., 2023), the first work proposed partitioning input series into multiple patches, effectively
enhancing the long-range time series prediction capability of Transformers.

• iTransformer (Liu et al., 2024b) captures relationships between variables by transposing input time series data.

• ETSformer (Woo et al., 2022) introduces both smoothing attention and frequency attention to replace the original self-
attention mechanism in Transformers.

#### 3) CNN-based Methods:

• TimesNet (Wu et al., 2023a), which selects representative periods in the frequency domain and processes them using 2D
convolution layers.

• TCN (Bai et al., 2018) conducts a systematic evaluation of generic convolutional and recurrent architectures for
sequence modeling.

• MICN (Wang et al., 2022) decomposes the time series signal into seasonal and trend components and learns them
separately using convolutional and linear regression layers.

13

|0.719 0.575&lt;br&gt;0.724 0.528&lt;br&gt;0.725 0.543&lt;br&gt;0.730 0.564|0.767 0.564&lt;br&gt;0.753 0.570&lt;br&gt;0.745 0.575&lt;br&gt;0.758 0.590|0.804 0.565&lt;br&gt;0.827 0.593&lt;br&gt;0.835 0.600&lt;br&gt;0.922 0.644|0.809 0.563&lt;br&gt;0.799 0.567&lt;br&gt;0.803 0.577&lt;br&gt;0.783 0.589|0.908 0.596&lt;br&gt;0.927 0.616&lt;br&gt;0.920 0.621&lt;br&gt;0.822 0.608|0.856 0.649&lt;br&gt;0.906 0.684&lt;br&gt;1.104 0.796&lt;br&gt;1.131 0.816|0.731 0.561&lt;br&gt;0.746 0.573&lt;br&gt;0.775 0.596&lt;br&gt;0.808 0.625|0.764 0.563&lt;br&gt;0.798 0.562&lt;br&gt;0.790 0.584&lt;br&gt;0.827 0.594|0.832 0.621&lt;br&gt;1.288 0.854&lt;br&gt;1.721 0.972&lt;br&gt;1.915 1.036|0.735 0.554&lt;br&gt;0.752 0.570&lt;br&gt;0.749 0.579&lt;br&gt;0.805 0.606|
|---|---|---|---|---|---|---|---|---|---|


Table 9: Full results for zero-shot forecasting on the ETT datasets, where prediction lengths H ∈{96, 192, 336, 720}.
“h1”, “h2”, “m1”, and “m2” denote ETTh1, ETTh2, ETTm1, and ETTm2 respectively. “h1 → m1” indicates that models trained
on ETTh1 are evaluated on ETTm1, and the same applies to other items. The team “Avg.” reports the results averaged from
all four prediction lengths. The best and the second best results are in bold and underlined. “1st Count” indicates the
number of times each method achieves the best results.

Models LLM-PS CALF TimeLLM GPT4TS PatchTST Crossformer FEDformer TimesNet MICN DLinear TiDE Ours (2024a) (2024) (2023b)
(2023) (2023a) (2022) (2023a) (2022) (2023) (2023)

Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

Title Suppressed Due to Excessive Size

h1 →m1

Avg. 0.721 0.541 0.755 0.574 0.847 0.600 0.798 0.574 0.894 0.610 0.999 0.736 0.765 0.588 0.794 0.575 1.439 0.870 0.760
0.577 0.774 0.574

|96 0.217 0.327 0.218 0.301 0.212 0.298 0.218 0.304 0.219 0.305 0.611 0.588 0.257 0.345 0.245 0.322 0.496 0.556 0.239 0.343 0.215 0.299&lt;br&gt;192 0.289 0.340 0.278 0.334 0.277 0.338 0.279 0.338 0.280 0.341 0.789 0.685 0.318 0.380 0.293 0.346 1.798 1.137 0.320 0.397 0.277 0.335 →m2&lt;br&gt;336 0.330 0.364 0.338 0.369 0.336 0.371 0.342 0.376 0.341 0.376 1.469 0.927 0.375 0.417 0.361 0.382 2.929 1.472 0.409 0.453 0.337 0.370&lt;br&gt;720 0.429 0.411 0.431 0.418 0.435 0.424 0.431 0.419 0.432 0.426 1.612 0.957 0.480 0.472 0.460 0.432 4.489 1.782 0.629 0.565 0.429 0.418 h1&lt;br&gt;Avg. 0.316 0.361 0.316 0.355 0.315 0.357 0.317 0.359 0.318 0.362 1.120 0.789 0.357 0.403 0.339 0.370 2.428 1.236 0.399 0.439 0.314 0.355|0.217 0.327&lt;br&gt;0.289 0.340&lt;br&gt;0.330 0.364&lt;br&gt;0.429 0.411|0.218 0.301&lt;br&gt;0.278 0.334&lt;br&gt;0.338 0.369&lt;br&gt;0.431 0.418|0.212 0.298&lt;br&gt;0.277 0.338&lt;br&gt;0.336 0.371&lt;br&gt;0.435 0.424|0.218 0.304&lt;br&gt;0.279 0.338&lt;br&gt;0.342 0.376&lt;br&gt;0.431 0.419|0.219 0.305&lt;br&gt;0.280 0.341&lt;br&gt;0.341 0.376&lt;br&gt;0.432 0.426|0.611 0.588&lt;br&gt;0.789 0.685&lt;br&gt;1.469 0.927&lt;br&gt;1.612 0.957|0.257 0.345&lt;br&gt;0.318 0.380&lt;br&gt;0.375 0.417&lt;br&gt;0.480 0.472|0.245 0.322&lt;br&gt;0.293 0.346&lt;br&gt;0.361 0.382&lt;br&gt;0.460 0.432|0.496 0.556&lt;br&gt;1.798 1.137&lt;br&gt;2.929 1.472&lt;br&gt;4.489 1.782|0.239 0.343&lt;br&gt;0.320 0.397&lt;br&gt;0.409 0.453&lt;br&gt;0.629 0.565|0.215 0.299&lt;br&gt;0.277 0.335&lt;br&gt;0.337 0.370&lt;br&gt;0.429 0.418|
|---|---|---|---|---|---|---|---|---|---|---|---|


96 0.217 0.327 0.218 0.301 0.212 0.298 0.218 0.304 0.219 0.305 0.611 0.588 0.257 0.345 0.245 0.322 0.496 0.556 0.239
0.343 0.215 0.299 192 0.289 0.340 0.278 0.334 0.277 0.338 0.279 0.338 0.280 0.341 0.789 0.685 0.318 0.380 0.293 0.346
1.798 1.137 0.320 0.397 0.277 0.335 336 0.330 0.364 0.338 0.369 0.336 0.371 0.342 0.376 0.341 0.376 1.469 0.927 0.375
0.417 0.361 0.382 2.929 1.472 0.409 0.453 0.337 0.370 720 0.429 0.411 0.431 0.418 0.435 0.424 0.431 0.419 0.432 0.426
1.612 0.957 0.480 0.472 0.460 0.432 4.489 1.782 0.629 0.565 0.429 0.418

h1 →m2

Avg. 0.316 0.361 0.316 0.355 0.315 0.357 0.317 0.359 0.318 0.362 1.120 0.789 0.357 0.403 0.339 0.370 2.428 1.236 0.399
0.439 0.314 0.355

|0.684 0.538&lt;br&gt;0.702 0.541&lt;br&gt;0.738 0.569&lt;br&gt;0.735 0.558|0.897 0.589&lt;br&gt;0.864 0.584&lt;br&gt;0.816 0.585&lt;br&gt;0.768 0.589|0.891 0.587&lt;br&gt;0.850 0.583&lt;br&gt;0.853 0.594&lt;br&gt;0.879 0.616|0.985 0.604&lt;br&gt;0.872 0.600&lt;br&gt;0.926 0.614&lt;br&gt;0.899 0.624|0.815 0.560&lt;br&gt;0.900 0.606&lt;br&gt;0.906 0.602&lt;br&gt;0.866 0.619|1.032 0.620&lt;br&gt;1.176 0.676&lt;br&gt;1.199 0.718&lt;br&gt;1.373 0.832|0.734 0.578&lt;br&gt;0.723 0.594&lt;br&gt;0.750 0.590&lt;br&gt;0.760 0.592|1.205 0.678&lt;br&gt;1.159 0.670&lt;br&gt;1.197 0.689&lt;br&gt;1.583 0.784|0.743 0.577&lt;br&gt;0.750 0.588&lt;br&gt;0.764 0.606&lt;br&gt;0.801 0.634|0.762 0.567&lt;br&gt;0.785 0.588&lt;br&gt;0.767 0.594&lt;br&gt;0.800 0.627|
|---|---|---|---|---|---|---|---|---|---|


96 0.684 0.538 0.897 0.589 0.891 0.587 0.985 0.604 0.815 0.560 1.032 0.620 0.734 0.578 1.205 0.678 0.743 0.577 0.762
0.567 0.819 0.566 192 0.702 0.541 0.864 0.584 0.850 0.583 0.872 0.600 0.900 0.606 1.176 0.676 0.723 0.594 1.159 0.670
0.750 0.588 0.785 0.588 0.845 0.586 336 0.738 0.569 0.816 0.585 0.853 0.594 0.926 0.614 0.906 0.602 1.199 0.718 0.750
0.590 1.197 0.689 0.764 0.606 0.767 0.594 0.834 0.595 720 0.735 0.558 0.768 0.589 0.879 0.616 0.899 0.624 0.866 0.619
1.373 0.832 0.760 0.592 1.583 0.784 0.801 0.634 0.800 0.627 0.867 0.616

h2 →m1

Avg. 0.714 0.552 0.836 0.586 0.868 0.595 0.920 0.610 0.871 0.596 1.195 0.711 0.741 0.588 1.286 0.705 0.764 0.601 0.778
0.594 0.841 0.590

|0.231 0.315&lt;br&gt;0.284 0.338&lt;br&gt;0.338 0.369&lt;br&gt;0.433 0.419|0.225 0.310&lt;br&gt;0.283 0.342&lt;br&gt;0.340 0.373&lt;br&gt;0.429 0.418|0.228 0.311&lt;br&gt;0.283 0.341&lt;br&gt;0.343 0.376&lt;br&gt;0.437 0.424|0.235 0.316&lt;br&gt;0.287 0.346&lt;br&gt;0.361 0.391&lt;br&gt;0.444 0.433|0.288 0.345&lt;br&gt;0.344 0.375&lt;br&gt;0.438 0.425&lt;br&gt;0.611 0.588|0.821 0.634&lt;br&gt;1.732 1.018&lt;br&gt;2.587 1.393&lt;br&gt;3.034 1.452|0.261 0.347&lt;br&gt;0.313 0.370&lt;br&gt;0.401 0.431&lt;br&gt;0.487 0.472|0.244 0.324&lt;br&gt;0.331 0.374&lt;br&gt;0.386 0.405&lt;br&gt;0.485 0.458|0.327 0.414&lt;br&gt;0.450 0.485&lt;br&gt;0.526 0.526&lt;br&gt;0.806 0.652|0.264 0.366&lt;br&gt;0.394 0.452&lt;br&gt;0.506 0.513&lt;br&gt;0.822 0.655|
|---|---|---|---|---|---|---|---|---|---|


96 0.231 0.315 0.225 0.310 0.228 0.311 0.235 0.316 0.288 0.345 0.821 0.634 0.261 0.347 0.244 0.324 0.327 0.414 0.264
0.366 0.226 0.315 192 0.284 0.338 0.283 0.342 0.283 0.341 0.287 0.346 0.344 0.375 1.732 1.018 0.313 0.370 0.331 0.374
0.450 0.485 0.394 0.452 0.289 0.348 336 0.338 0.369 0.340 0.373 0.343 0.376 0.361 0.391 0.438 0.425 2.587 1.393 0.401
0.431 0.386 0.405 0.526 0.526 0.506 0.513 0.339 0.372 720 0.433 0.419 0.429 0.418 0.437 0.424 0.444 0.433 0.611 0.588
3.034 1.452 0.487 0.472 0.485 0.458 0.806 0.652 0.822 0.655 0.433 0.422

h2 →m2

Avg. 0.322 0.359 0.319 0.360 0.322 0.363 0.331 0.371 0.420 0.433 2.043 1.124 0.365 0.405 0.361 0.390 0.527 0.519 0.496
0.496 0.321 0.364

1st Count 27 8 4 0 0 0 0 0 0 0 5

#### 4) MLP-based Methods:

• DLinear (Zeng et al., 2023) explores the application of linear layers in time series tasks and achieves efficient time
series prediction.

• TiDE (Das et al., 2023) designs an encoder-decoder structure based on MLP, which can achieve comparable performance
with Transformers while requiring less computation burdens.

• TimeMixer (Wang et al., 2024) downsamples the time series signal into multiple-scale inputs for ensemble predictions
in the MLP model.

Additionally, we compare with early time series forecasting methods:

• N-BEATS (Oreshkin et al., 2019b) is the first to apply deep learning models to time series forecasting tasks and
designed a deep model based on residual links and Fourier series, achieving better performance than traditional
statistical methods.

• N-HiTS (Challu et al., 2022) finds that N-BEATS becomes slow as the forecast length increases and reduces the time
series data length by downsampling the input series, thereby effectively improving the model’s inference speed.

#### A.2. Benchmark Datasets

In this paper, we evaluate our proposed LLM-PS on the Electricity Transformer Temperature (ETT) (Zhou et al., 2021a),
Weather (Wu et al., 2021), Traffic (Wu et al., 2021), Illness (ILI), (Wu et al., 2021), Electricity (Trindade, 2015), M4
(Makridakis et al., 2018), and Electrocardiography (ECG) (Moody & Mark, 2001) datasets. These datasets are sourced from
different domains, such as finance and meteorology. In general, the time series in these datasets exhibit distinct
patterns, as shown in Fig. 5. The details of these time series datasets are as follows. ETT records measurements from an
electricity transformer over an extended period, primarily focusing on seven variables,

Figure 5: Visualization of the time series belongs to the Weather dataset and the finance subset in the M4 dataset. The
temperature readings measured by the meteorological station are generally stable, whereas stock prices in financial
markets fluctuate rapidly around the average value.

including the target variable “oil temperature” and six power load features. The ETT dataset contains four subsets
(ETTh1, ETTh2, ETTm1, and ETTm2) sampled at different frequencies (hourly and minutely) from various locations. The
training, validation, and test sets of ETT contain data sampled over 12, 4, and 4 months, respectively. Weather is a
meteorological dataset used for climate modeling and environmental research, which records 21 meteorological indicators
every 10 minutes throughout the entire year of 2020, such as air temperature and humidity. Traffic is commonly employed
for traffic flow forecasting, which predicts the spatio-temporal traffic volume by considering historical traffic data
and additional features from adjacent locations. This dataset captures traffic volume measurements every 15 minutes at
862 sensor locations situated along two main highways from July 1, 2016, to July 2, 2018. Illness includes weekly data
from the Centers for Disease Control and Prevention in the United States from 2002 to 2021, recording the ratio of
patients with influenza to the total number of patients. Illness contains seven variables, such as age and number of
providers. Electricity consists of 321 variables related to energy utilities for energy production and usage in the
United States from over 2,000 U.S. utilities in 2017. M4 comprises 100,000 time series utilized in the fourth edition of
the Makridakis Forecasting Competition. This dataset encompasses yearly, quarterly, monthly, and various other datasets.
ECG is a collection of biomedical datasets primarily used for research in electrocardiogram (ECG) analysis. This dataset
contains 48 half-hour two-channel ambulatory ECG recordings from 47 subjects with a sampling rate of 360 Hz.

#### A.3. Metrics of Time Series Forecasting

In this paper, we mainly employ five widely used metrics to assess model performance, including Mean Squared Error
(MSE), Mean Absolute Error (MAE), Mean Absolute Scaled Error (MSAE), Symmetric Mean Absolute Percentage Error (SMAPE),
and Overall Weighted Average (OWA). MSE measures the average of the squared differences between the predicted and actual
values. MSE gives more weight to more significant errors because the errors are squared, thereby sensitive to outliers.
Given T time steps ground-truth time series signal Y = {xH+1, . . . , xH+T } ∈ RT ×V and prediction ˆY = {ˆxH+1, . . . ,
ˆxH+T } ∈ RT ×V , MSE is calculated as:

T X

MSE = 1

i=1 (xH+i − ˆxH+i)2 . (11)

T

MAE quantifies the average squared differences between predicted and actual values. It is less sensitive to outliers
than MSE because it does not square the errors, treating all errors linearly. MAE is computed as:

T X

MAE = 1

i=1 |xH+i − ˆxH+i| . (12)

T

15

Title Suppressed Due to Excessive Size

MSAE evaluates the accuracy of a model by scaling the absolute error relative to the actual values, which is calculated
as:

T X

|xH+i − ˆxH+i|

MSAE = 1

|xH+i| . (13)

T

i=1

SMAPE aims to provide a more balanced error with a symmetry formula, especially when the actual values approach zero.
This helps mitigate the instability in MAPE when the actual values are small. SMAPE is calculated as:

T X

2|xH+i − ˆxH+i|

SMAPE = 1

xH+i + |ˆxH+i| . (14)

T

i=1

OWA is commonly used for multiple tasks or criteria, and the importance of each task or metric differs. It allows for a
balanced evaluation by adjusting the contribution of each task based on its relative importance. OWA is computed as:

N X

i=1 wi · vi, (15)

OWA =

where N is the number of tasks (metrics), wi is the weight assigned to the i-th task (metric) and PN i=1 wi = 1, vi is
the evaluation result of the i-th task (metric).

### B. Model Analysis

#### B.1. Wavelet Transform

Our MSCNN employs the DauBechies 4 (DB4) wavelet transform with four vanishing moments to decouple low-frequency
components Wb low and high-frequency components {Wb high i}w i=1. The DB4 wavelet transform is widely used in time
series signal processing due to its compact support and smoothness. Specifically, the DB4 wavelet transform decomposes
the input time series via sequential decomposition and downsampling steps. In the decomposition step, the time series X
= {x1, . . . , xH} ∈ RH×V is progressively decomposed into multiple frequency bands with w levels, encompassing both
low-frequency and high-frequency components. At i-th decomposition level, the time series X is passed through low-pass
filter f i low and high-pass filter f i high to generate the low-frequency approximation coefficients ai and high-
frequency detail coefficients di:

L−1 X

L−1 X

k=0 f i high[k] · X[2n − k], n = 1, 2, . . . , H + L − 1

 , (16)

k=0 f i low[k] · X[2n − k], di[n] =

ai[n] =

2

where L = 8 denotes the filter length. After w decomposition levels, the outputs from these filters undergo downsampling
by a factor of 2. The final approximation aw, which represents the smoothest features of the signal, serves as the low-
frequency components Wb low. Meanwhile, the detail coefficients {di}w i=1, which capture diverse high-frequency details
or changes in the input time series, constitute the high-frequency components {Wb high i}w i=1.

Conversely, the inverse DB4 wavelet transform reconstructs the original signal through upsampling and reversed-order
filtering. During the upsampling step, the approximation coefficients and detail coefficients are expanded by a factor
of 2 to recover the original signal length. In the filtering step, the upsampled coefficients are convolved with the
same low-pass and high-pass filters used during the decomposition in reverse order to reconstruct the time series
signal:

w X

 ˜ai + ˜di  . (17)

X =

i=1

Here, ˜ai and ˜di represent reconstruction coefficients and details, which are calculated as follows:

n ˜f i low[m − 2n] · ai[n], ˜di[m] = X

n ˜f i high[m − 2n] · di[n], m = 1, 2, . . . , H. (18)

˜ai[m] = X

˜f i low (low-pass) and ˜f i high (high-pass), derived as time-reversed versions of the decomposition filter f i low
(low-pass) and f i high, as follows: ˜f i low[k] = f i low[L − 1 − k], ˜f i high[k] = f i high[L − 1 − k], k = 0, 1, . .
. , L − 1. (19)

16

Title Suppressed Due to Excessive Size

Original Time Series Signal

2.5 5.0 7.5

0.0 0.2 0.4 0.6 0.8 1.0

Short-Term Pattern Decoupled by Fourier Transform

2.5

0.0

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Short-Term Pattern Decoupled by Average Pooling

1 0 1

0.0 0.2 0.4 0.6 0.8 1.0

Short-Term Pattern Decoupled by Wavelet Transform (Ours)

0.0

0.0 0.2 0.4 0.6 0.8 1.0 2.5

Long-Term Pattern Decoupled by Fourier Transform

7.5

5.0

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Long-Term Pattern Decoupled by Average Pooling

7.5

5.0

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Long-Term Pattern Decoupled by Wavelet Transform (Ours)

7.5

5.0

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Figure 6: Visualization results of temporal patterns decoupled by our proposed wavelet-transform-based decoupling
technique and other widely-used decoupling methods based on Fourier transform and average pooling.

#### B.2. Visualizations of Decoupled Temporal Patterns

In this section, we compare our proposed wavelet-transform-based temporal pattern decoupling technique with other
methods that employ Fourier transform and average pooling. Fig. 6 visualizes the decomposed short-term and long-term
patterns. We can observe that our decoupling method achieves significantly better results than other decoupling methods.
In particular, our method accurately captures short-term patterns that mirror the periodic fluctuations of the original
time series, while effectively representing the overall trend in the long-term patterns.

#### B.3. Receptive Field of MSCNN

In CNNs, the receptive field refers to the region of the input associated with each output in a convolutional layer. The
size of the receptive field determines how much of the input the network observes during convolution operations. In
general, the receptive field of the output for a convolutional layer depends on the kernel size k, stride s, and the
receptive field of the input Rin, which is formulated as: Rout = Rin + (k − 1) × s. (20)

In our MSCNN block, the features are divided into multiple branches and processed by parallel 3×3 convolutional layers
with an identical stride s=1. Given the features {¯F1, . . . , ¯FB} obtained from B branches in Eq. (1), we assume that
the receptive field of input features Rin=1 and B=4, and the receptive fields of multiple-branch features are [3, 5, 7,
9]. Here, features with a small receptive field focus on short-term patterns, namely the periodic fluctuations, while
features with a

17

Title Suppressed Due to Excessive Size

#### T2T Encoder

#### T2T Decoder T2T Decoder T2T Decoder

...

... Similarity-Based Semantics Filtering

Multi-Head

Attention

Add & Layer Norm

Forward

Add & Layer Norm

Input

× N

Multi-Head

Attention

Add & Layer Norm

Forward

Add & Layer Norm

Input

× N

Masked Patch Unmasked Patch Predicted Patch ...

Hidden States

Original Label Predicted Label

Filtered Embedding (a) Proposed T2T Architecture (b) Semantic filtering

Figure 7: The diagram of our proposed Time-to-Text (T2T) module. The input time series is first divided into P patches,
with some patches randomly masked. These patches are then fed into the encoder and decoder to facilitate T2T’s accurate
reconstruction and encourage T2T to learn meaningful semantic information by encouraging it to accurately predict the
semantic labels of the time patches.

large receptive field focus on long-term patterns, namely the overall trend. Therefore, our proposed MSCNN can
effectively capture both short-term and long-term patterns.

#### B.4. Time-to-Text Module

Filtered Embedding Original Embedding

Filtered Embedding

Text Index: Time, Trend, Sequence,

Step, Lag, Period

|0.50&lt;br&gt;MSE&lt;br&gt;0.48 MAE MAE&lt;br&gt;Used by LLM&amp;#45;PS&lt;br&gt;0.46&lt;br&gt;and&lt;br&gt;0.44&lt;br&gt;MSE&lt;br&gt;0.42&lt;br&gt;of&lt;br&gt;0.40&lt;br&gt;Mean&lt;br&gt;0.38&lt;br&gt;0.36&lt;br&gt;0.0001 0.001 0.01 0.1 1&lt;br&gt;Prediction Horizon|MSE&lt;br&gt;MAE&lt;br&gt;Used|by LLM&amp;#45;PS|
|---|---|---|

Figure 8: Parametric sensitivities of λ in Eq. 9.

P , and O denotes the overlap length between consecutive patches. Semantic Filtering. In the LLM word embeddings, there
are numerous inrelevant semantic information for Time Series Forecasting (TSF). As shown in Fig. 7, the original word
embeddings Eori ∈ RW ×D (where W and D are the length and dimension of Eori) of the GPT2 model includes embedding with
corresponding word such as “Thro”, “Magn”, and “belt”, which are irrelevant to TSF. To facilitate T2T in extracting
valuable semantic information for TSF, we filter the LLM word embeddings based on text indices relevant to TSF.

Specifically, given the text indices {ti}I i (I denotes the number of text indices) corresponding to the TSF, the
similarities

Title Suppressed Due to Excessive Size

between their word embeddings ˆE = {Eori[ti]}I i=1 and LLM word embeddings Eori are computed as follows:

s = ˆE · Eori ∥ˆE∥2∥Eori∥2 . (21)

Then, we select the top 100 most similar word embeddings as the final word embeddings E for T2T training.

#### B.5. Parameter Sensitivity Analysis

The LLM-PS equation (Eq. (9)) in our model has one tuning parameter, denoted as λ. We analyze its sensitivities on the
ETTh1 dataset by varying λ within the range of {0.0001, 0.001, 0.01, 0.1, 1} and observe the mean MSE/MAE across predict
lengths spanning {96, 192, 336, 720}, as shown in Fig. 8. Despite the large fluctuations in λ, the MSE/MAE curve of our
LLM-PS remains relatively stable. These results demonstrate the robustness of our LLM-PS against parameter variations.
Furthermore, our LLM-PS achieves its best performance when λ = 0.01, so we adopt this parameter configuration in our
method.