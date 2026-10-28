Figure S6G source and method

The 1,794 sequence-exclusive held-out pairs are grouped using the 12 shared −NCP intervals in Figure S4D.
The −NCP-only baseline is a five-member MLPRegressor ensemble using only Opposite_Charge_Number (−NCP). It uses the frozen 13-feature base predictor's MLP architecture, hyperparameters, and seeds, with scaling fit on the 917 development pairs. It is trained on those 917 pairs only.
The comparison series is the frozen full 13-feature base predictor's base_prediction on the held-out pairs; the optional residual correction is not included. Test targets were used only to calculate held-out RMSE.
Points show within-bin RMSE. Error bars are percentile 95% intervals from 5,000 two-way cluster-bootstrap replicates resampling RG/RGG and AHD identities independently within each bin (seed 20261018). Pair counts are shown as translucent bars on the right-hand y-axis. Bins use the S4D convention (lower, upper], except the first includes its lower bound. The 12 bin counts were checked against the S4D held-out audit and sum to 1,794.
No model selection, tuning, or fitting used held-out targets.
