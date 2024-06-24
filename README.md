#  **A Transformer-based deep reinforcement learning method driven by multi-source heterogeneous data for the replenishment problem with carried-over inventory**
The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper <u>A Transformer-based deep reinforcement learning method driven by multi-source heterogeneous data for the replenishment problem with carried-over inventory</u> by Yu-Xin Tian and Chuan Zhang. 
## Abstract
We propose a novel end-to-end deep reinforcement learning (DRL) inventory control method that integrates Word embedding, Transformer networks, and the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm to address multi-period replenishment problems with carried-over inventory. Utilizing both numerical demand-related features and textual online reviews, our model enhances demand estimation and decision-making in uncertain environment where the distribution of demand is unknown. The proposed end-to-end DRL method amalgamates the TD3 algorithm with hybrid deep neural network structures, enabling the direct derivation of multi-period replenishment quantities. Empirical evaluation using real-world data demonstrates that our method significantly reduces costs by at least 28.7% compared to the latest benchmarks. It excels in scenarios with increasing unit stockout costs relative to unit inventory costs, showcasing adaptability. In comparison to non-end-to-end approaches involving pre-prediction and subsequent decision-making, our model exhibits remarkable adaptability to situations across various cost parameters, ultimately resulting in more economically efficient replenishment decisions. Additionally, our model outperforms benchmarks utilizing solely numerical features or relying exclusively on textual reviews, underscoring the paramount importance of integrating multi-source heterogeneous data.

## Environment Requirements
**The computer conditions:**

-   System: `X64 Windows 11`;

-   Memory: `16GB`;

-   CPU: `12th Gen Intel(R) Core(TM) i7-12700H 2.30 GHz`;

-   GPU: `NVIDIA GeForce RTX 3070 Ti Laptop`. 

To run the code, you will need to make sure that you have the following dependencies installed: 

`Python 3.11`,  `Cuda 11.2.128`,  `PyTorch 2.0.1`, `numpy`, `pandas`, `jupyter`, `notebook`, `selenium`, `requests`, `calendar`, `xlwt`, `urllib`, `json`, `scipy`, `scikit-learn`, `jieba`, `matplotlib`, `docplex`, `matplotlib`, `xlsxwriter`

## Data and dataset division

### *Data source*

| Data                   | Source         | URL                                                          | Valid date range |
| ---------------------- | -------------- | ------------------------------------------------------------ | ---------------- |
| Search engine data     | Baidu Index    | https://index.baidu.com/v2/index.html#/                      | 2014.01~2024.02  |
| Macroeconomic data     | Wind           | https://www.wind.com.cn/                                     | 2014.01~2024.02  |
| Textual online reviews | Auto Home      | https://www.autohome.com.cn/grade/carhtml/A.html             | 2014.12~2024.02  |
| History sales data     | *Chezhuzhijia* | https://xl.16888.com/s/57415/ <br>https://xl.16888.com/s/117870/ <br>https://xl.16888.com/s/126615/ <br> | 2014.12~2024.03  |

### *Dataset division*

We used data from January 2015 to March 2022 (87 samples in total) as the training set. Data from April 2022 to March 2023 served as the validation set for determining hyperparameter values. Subsequently, data from April 2023 to March 2024 formed the test set for evaluating the method's performance and conducting experimental comparisons. The historical sales were used as historical demand during training. 

## Folder structure
The project folder structure and its comments are as follows:   

```
.
│  README.md    # Introduction of data and code.
│
├─Analysis
│  │  Analysis_Benchmarks.ipynb    # Compare with other methods, and draw the diagrams and tables in the paper
│  │  utils.py    # Some tool functions.
│  │
│  └─Paper_outputs\    # Save the pictures and tables in the paper.
├─Collect_data
│  ├─Get_reviews
│  │  │  Get_reviews.ipynb    # Used to crawl online review data.
│  │  │  UserAgent_set.pickle    # Necessary components.
│  │  │
│  │  └─Reviews\    # Save textual review data tables.
│  ├─Get_search_data
│  │  │  Get_search.ipynb    #  Used to crawl Baidu index data.
│  │  │
│  │  ├─Merged_data\    # Save the merged and sorted search engine data tables.
│  │  └─tmp\    # Save intermediate process files.
│  └─Other_features
│      │  Macro_90.wet    # Template for downloading macroeconomic data from Wind database.
│      │  Macro_data.xlsx    # Downloaded and processed macroeconomic data.
│      │
│      └─Sales_data\    # Downloaded and processed history sales data.
├─Methods
│  │  DNN_Afshin2019.py    # Deep neural network (DNN) (Oroojlooyjadid et al., 2019).
│  │  E2E_Tian2023.py    # The E2E method proposed by Tian and Zhang (2023).
│  │  EDD.py    # Empirical demand distribution.
│  │  Forecast_TransX.py    # Use our network structure to forecast demand as a benchmark.
│  │  HyperParams.py    # Selected hyperparameter combinations.
│  │  KO_Ban2018.py    # Kernel optimization (KO) (Ban and Rudin, 2018).
│  │  LML_Ban2018.py    # Linear machine learning (LML) (Ban and Rudin, 2018).
│  │  Roberts_TD3_MLP.py    # Network structures of the ablation method without Transformer, and the TD3 algorithm.
│  │  Roberts_TD3_Trans.py    # Network structures of our method, and the TD3 algorithm.
│  │  RUN.py    # The main program of running above all methods.
│  │  TransX_Run.py    # Our method.
│  │  Trans_Run.py    # The ablation method without numerical features.
│  │  utils.py    # Some tool functions.
│  │  X_Run.py    # The ablation method without textual reviews.
│  │
│  ├─checkpoints\    # Save the trained models.
│  └─Logs\    # Curves of training processes.
├─Outputs\    # Save the test results.
└─Preprocess_data
    │  Describe_sales_data.ipynb    # Make descriptive statistics on sales data.
    │  Numerical_feature_engineering.ipynb    # Select numerical features by time difference correlation analysis.
    │  Preprocess_data.ipynb    # Combine and process numerical and textual feature data to form the input required by subsequent programs.
    │
    ├─cache\    # Save intermediate process files.
    ├─Filter_results\    # Time difference correlation analysis results.
    └─Inputs_data\    # Save the generated model inputs.
```

## Workflow

| **Which results  to reproduce**                              | **Data File**                                                | **Code File**                                                | **Expected  output**                                         | **Run time at  the above-specified computer conditions** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| Get search index data                                        | -                                                            | `Get_search.ipynb` in `Collect_data/Get_search_data/`.       | `S_Lavida.xlsx`,  `S_HavalH6.xlsx`, and `S_Emgrand.xlsx` in  `Collect_data/Get_search_data/Merged_data/`. | About 34 minutes.                                        |
| Get textual online review data                               | -                                                            | `Get_reviews.ipynb` in  `Collect_data/Get_reviews/`.         | `R_Lavida.xls`,  `R_HavalH6.xls`, and `R_Emgrand.xls` in `Collect_data/Get_reviews/Reviews/`. | About 45 minutes.                                        |
| Select numerical features by time difference  correlation analysis, and generate inputs required by subsequent programs. | (1) All files in  `Collect_data/Get_search_data/Merged_data/`; (2) `Macro_data.xlsx` in  `Collect_data/Other_features/`; (3) All files in  `Collect_data/Other_features/Sales_data/`; (4) All files in  `Collect_data/Get_reviews/Reviews/`. | `Numerical_feature_engineering.ipynb`  and `Preprocess_data.ipynb` in `Preprocess_data/`. | Dataset files in `Preprocess_data/Inputs_data/`.             | About 73.6 seconds.                                      |
| Determine hyperparameters by repeated  experiments.          | All files in  `Preprocess_data/Inputs_data/`.                | `DNN_Afshin2019.py`, `E2E_Tian2023.py`,  `Forecast_TransX.py`, `Trans_Run.py`, `X_Run.py`, and `TransX_Run.py` in `Methods/`. | `HyperParams.py` in `Methods/`.                              | About 3 days.                                            |
| Train our method.                                            | All files in  `Preprocess_data/Inputs_data/`.                | `TransX_Run.py` and `RUN.py`  in `Methods/`.                 | Saved models in `Methods/checkpoints/`.                      | Total 13979.25 seconds (Three experiments).              |
| Test our method.                                             | All files in  `Preprocess_data/Inputs_data/`.                | `TransX_Run.py` and `RUN.py`  in `Methods/`.                 | Result files in `Outputs/`.                                  | Total 75 seconds (Three experiments).                    |
| Train the Deep neural network (DNN)  (Oroojlooyjadid et al., 2019) method. | All files in  `Preprocess_data/Inputs_data/`.                | `DNN_Afshin2019.py` and  `RUN.py` in `Methods/`.             | Saved models in `Methods/checkpoints/`.                      | Total 115.23 seconds (Three experiments).                |
| Test the Deep neural network (DNN)  (Oroojlooyjadid et al., 2019) method. | All files in  `Preprocess_data/Inputs_data/`.                | `DNN_Afshin2019.py` and  `RUN.py` in `Methods/`.             | Result files in `Outputs/`.                                  | Total 62.19 seconds (Three experiments).                 |
| Train the E2E method proposed by Tian and Zhang  (2023).     | All files in  `Preprocess_data/Inputs_data/`.                | `E2E_Tian2023.py` and `RUN.py`  in `Methods/`.               | Saved models in `Methods/checkpoints/`.                      | Total 205.27 seconds (Three experiments).                |
| Test the E2E method proposed by Tian and Zhang  (2023).      | All files in  `Preprocess_data/Inputs_data/`.                | `E2E_Tian2023.py` and `RUN.py`  in `Methods/`.               | Result files in `Outputs/`.                                  | Total 65.3 seconds (Three experiments).                  |
| Train the ablation method that uses our network  structure to forecast demand and then decision. | All files in  `Preprocess_data/Inputs_data/`.                | `Forecast_TransX.py` and  `RUN.py` in `Methods/`.            | Saved models in `Methods/checkpoints/`.                      | Total 23.95 seconds (Three experiments).                 |
| Test the ablation method that uses our network  structure to forecast demand and then decision. | All files in  `Preprocess_data/Inputs_data/`.                | `Forecast_TransX.py` and  `RUN.py` in `Methods/`.            | Result files in `Outputs/`.                                  | Total 8.29 seconds (Three experiments).                  |
| Train the ablation method without numerical  features.       | All files in  `Preprocess_data/Inputs_data/`.                | `Trans_Run.py` and `RUN.py`  in `Methods/`.                  | Saved models in `Methods/checkpoints/`.                      | Total 17121.9 seconds (Three experiments).               |
| Test the ablation method without numerical  features.        | All files in  `Preprocess_data/Inputs_data/`.                | `Trans_Run.py` and `RUN.py`  in `Methods/`.                  | Result files in `Outputs/`.                                  | Total 62.52 seconds (Three experiments).                 |
| Train the ablation method without textual  reviews.          | All files in  `Preprocess_data/Inputs_data/`.                | `X_Run.py` and `RUN.py` in  `Methods/`.                      | Saved models in `Methods/checkpoints/`.                      | Total 1511.22 seconds (Three experiments).               |
| Test the ablation method without textual reviews.            | All files in  `Preprocess_data/Inputs_data/`.                | `X_Run.py` and `RUN.py` in  `Methods/`.                      | Result files in `Outputs/`.                                  | Total 70.7 seconds (Three experiments).                  |
| Execute the Empirical demand distribution (EDD)  method.     | All files in  `Preprocess_data/Inputs_data/`.                | `EDD.py` in `Methods/`.                                      | Result files in `Outputs/`.                                  | < 1second.                                               |
| Execute the Kernel optimization (KO) (Ban and  Rudin, 2018) method. | All files in  `Preprocess_data/Inputs_data/`.                | `KO_Ban2018.py` in `Methods/`.                               | Result files in `Outputs/`.                                  | Total 38.58 seconds (Three experiments).                 |
| Execute the Linear machine learning (LML) (Ban  and Rudin, 2018) method. | All files in  `Preprocess_data/Inputs_data/`.                | `LML_Ban2018.py` in `Methods/`.                              | Result files in `Outputs/`.                                  | Total 20.46 seconds (Three experiments).                 |
| Compare and visually analyze above all outputs.              | All files in `Outputs/`.                                     | `Analysis_Benchmarks.ipynb`.                                 | All the figures and  tables in sections 4.3 and 4.4 of the paper are generated in the folder  `Analysis/Paper_outputs/`. | < 1second.                                               |

## Results

### **Literature comparison**

Box plots of costs across all test scenarios for different literature methods.

<center class="half">
    <img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/CompareBox_Lavida.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/CompareBox_Emgrand.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/CompareBox_HavalH6.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/CompareBox_All_products.png?raw=true" width="200"/>
</center>

Box plots of costs across all test scenarios for different ablation methods.

<center class="half">
    <img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/AblationBox_Lavida.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/AblationBox_Emgrand.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/AblationBox_HavalH6.png?raw=true" width="200"/><img src="https://github.com/Jesse-tien/WET_TD3_Inventory/blob/main/Figures/AblationBox_All_products.png?raw=true" width="200"/>
</center>
