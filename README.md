#  **A multimodal deep reinforcement learning framework for multi-period inventory decision-making under demand uncertainty**
The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper by Yu-Xin Tian and Chuan Zhang. 
## Abstract
We study the problem of multi-period inventory decision-making using multisource multimodal data and propose a deep reinforcement learning (DRL) method—Word Embedding and Transformer-enhanced Twin Delayed Deep Deterministic Policy Gradient (WET-TD3)—that integrates multimodal environmental perception with policy optimization to produce end-to-end replenishment decisions for each period. First, we design multimodal feature-aware agent neural networks that incorporate word embeddings and Transformer modules based on demand-related structured features and unstructured customer review texts from multiple sources. This design constructs a state space that adapts to dynamic market environments. Second, we integrate into the TD3 algorithm a multimodal Actor-Critic architecture tailored for high-dimensional heterogeneous inputs. Additionally, we introduce delayed policy updates, experience replay, and exploration noise mechanisms to improve training stability. Finally, experiments based on real-world data show that the WET-TD3 method significantly outperforms benchmarks, achieving an average cost reduction of over 53.69%. The method dynamically adjusts replenishment strategies in response to changes in the relative magnitude of unit holding and underage costs, maintaining stable performance under varying cost structures. These findings highlight the importance of deeply integrating unstructured textual reviews and structured features from multiple sources for high-accuracy replenishment. They also show that the DRL framework effectively supports long-term optimization goals in uncertain and dynamic demand environments.

## Environment Requirements
**The computer conditions:**

-   System: `X64 Windows 11`;

-   Memory: `16GB`;

-   CPU: `12th Gen Intel(R) Core(TM) i7-12700H 2.30 GHz`;

-   GPU: `NVIDIA GeForce RTX 3070 Ti Laptop`. 

To run the code, you will need to make sure that you have the following dependencies installed: 

`Python 3.11`,  `Cuda 11.2.128`,  `PyTorch 2.0.1`, `numpy`, `pandas`, `jupyter`, `notebook`, `selenium`, `requests`, `calendar`, `xlwt`, `urllib`, `json`, `scipy`, `scikit-learn`, `jieba`, `matplotlib`, `docplex`, `matplotlib`, `xlsxwriter`

## Experimental details

