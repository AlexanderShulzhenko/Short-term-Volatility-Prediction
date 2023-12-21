# Short-term volatility prediction with LightGBM
This repo introduces a framework with implemented LightGBM model and feature engineering scripts to predict short-term volatility fluctuations of financial assets.

Emphasizing cryptocurrency markets, known for their heightened volatility compared to traditional asset markets, the repo concentrates on engineering predictive features crucial for volatility estimation. The comprehensive methodology delineates the creation of a feature set employing a modular structure. Additionally, it briefly outlines the process of selecting the most informative features. The repo meticulously examines the predictive quality of the generated forecasts while dissecting samples where the model gives wrong predictions. Beyond the developed model, an evaluation of existing models pertaining to this task is conducted.

The structure of this repo is outlined on the fugure below (marked parts are presented in the repo):

<img width="657" alt="Снимок экрана 2023-12-21 в 15 54 41" src="https://github.com/AlexanderShulzhenko/Short-term-Volatility-Prediction/assets/80621503/82c234fb-60df-4627-ae98-c39ed3681007">

## Repo structure

The repo contains **4 main feature modules**:
- Candlestick features;
- Tech. indicators features;
- Exchange data features;
- Stochastic process features.

Repo also contains **module combination** part, which merges all the features and trains the model using LightGBM. 
