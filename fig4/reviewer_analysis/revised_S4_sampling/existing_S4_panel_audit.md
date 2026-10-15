# Existing S4 Panel Audit

Historical panels were verified from executable code/data in `supp4.py`, `25_pairs_features_data.csv`, and `initial_set_features_data.csv`.
The old panels are retained as sampling-design/provenance analyses rather than independent biological validation.

| panel | purpose | x variable | y variable | n | source file | script/function | recommendation |
|---|---|---|---|---:|---|---|---|
| S4A | Pilot-stage relationship between B22 and charge complementarity. | Charge_Complementarity | B22_corr1_log10 | 25 | /Users/yuktikhanna/rtads-rgg-interactions/fig4/25_pairs_features_data.csv | plot_old_panels/S4A | retain with relabelling as sampling-design/provenance analysis |
| S4B | Pilot-stage relationship between B22 and mean pair NCPR. | Average_NCPR | B22_corr1_log10 | 25 | /Users/yuktikhanna/rtads-rgg-interactions/fig4/25_pairs_features_data.csv | plot_old_panels/S4B | retain with relabelling as sampling-design/provenance analysis |
| S4C | Pilot-stage relationship between B22 and average lambda. | Average_lambda | B22_corr1_log10 | 25 | /Users/yuktikhanna/rtads-rgg-interactions/fig4/25_pairs_features_data.csv | plot_old_panels/S4C | retain with relabelling as sampling-design/provenance analysis |
| S4D | Early simulated relationship between B22 and net-charge product/OCN that motivated later OCN-bin sampling. | Opposite_Charge_Number | B22_corr1_log10 | 595 | /Users/yuktikhanna/rtads-rgg-interactions/fig4/initial_set_features_data.csv | plot_old_panels/S4D | retain with relabelling as sampling-design/provenance analysis |
