# Overview

The work in this repo is based on my classwork in IEOR 4735: Structured and Hybrid Products at Columbia University Fall 2024. The course was broadly theoretical covering numerous continuous time models in order to derive and examine equity models (including stochastic and local vol) and rates models (such as ATSMs and HJM framework). While this repo began as a playground for me to better understand course material in practice, I have left a copy of my final project to give a taste of the type of work done throughout the class and problems I'd be interested in tackling in the future.

# Final Project
### Summary
- Designed and implemented robust pricing routine for a custom product which protects the holder against European market - US forward rate correlation over variable time horizons
- Calibrated Hull-White (extended Vasicek) and lognormal SX5E model by calculating implied volatilities from equity and rate derivatives market data as well as computing parameters through historical regressions
- Forecasted forward rates and stock performance using Monte Carlo methods and Cholesky decomposition to allow for correlated Brownian motions

### Files
The pricing report can be found [here](final_project/IEOR_4735_Final_Project.pdf) which presents the product term sheet, explains the modeling choices, and summarizes results/next steps. The pricing routine can be viewed in [a Jupyter Notebook](final_project/routine.ipynb). The code is lightly commented and is best read in tandem with the report itself.
