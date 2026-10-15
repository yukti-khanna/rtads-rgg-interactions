# Preparation status

The workflow is prepared for local execution. After your instruction, no analysis, tests or rendering were run. Numerical results, regenerated figures and their visual verification are therefore pending. Earlier read-only inspection included one RDF pickle, which was a two-row pandas DataFrame with five individual `run_1`–`run_5` columns. No input data or historical output was edited.

The two reference repositories were located from the actual workspace:

- `/Users/yuktikhanna/idr-interaction-analysis`: commit `80b07345c7a50e12e5118a72ec461f4fe5db5eb0`; Git reports untracked `fig4/` and no tracked modifications. The untracked convergence script corroborates 7.5 nm, 2 nm tail width and replica means using existing helpers.
- `/Users/yuktikhanna/idr-interaction-predict`: commit `af534d380250e6e9e89ec0c9f1c544106f67cb2b`; Git reports untracked `models_old/` and `sample_rdfs/`, with no tracked modifications. Those additions are not asserted as the supplied new run.

No applicable `AGENTS.md` was found in either reference repository, the working folder or inspected parent directories. No reference repository was modified.

Read-only inspection recounted 20 Table_S8 rows and nine numerical Kd values. The new QC summary reports 73 predictions and zero rejected pairs; the runtime script independently recounts these and the RDFs rather than hard-coding them. The numerical RDF reproduction, old/new metrics, full sequence audit and manifest hash checks are left to the local run.

The current documents were accessible at:

- `/Users/yuktikhanna/writing/rbp-tfs/cell_reports/final_cell_reports.docx`
- `/Users/yuktikhanna/writing/rbp-tfs/cell_reports/ptp_response.docx`
- `/Users/yuktikhanna/rtads-rgg-interactions/supplementary/supplementary_figures.docx`

Document text was inspected for context without edits or rendering. The manuscript describes 0.1 nm bins, an L/2 cutoff, a 2 nm tail window and nm³ units. The reviewer response names outcome/affinity validation Figure 5C/5D and discusses the prior eight-pair exclusion. The final assembled panel layout was not verified. Supplementary exports use descriptive filenames to avoid occupied S7/S8 numbering.

Key preparation decisions:

- Old-panel predictions come from `Table_S8.csv`, exactly as read by `make_fig6_panels_final.py`. The two historical hybrid tables have discrepant predictions and older construct lengths; neither is silently substituted.
- Frozen new final and base predictions are copied directly. Predictor calibration, residual gates and lambda are untouched.
- Both AHD/RGG identities are checked with oriented pair identifiers; reversed historical feature roles are explicit. Full new sequence/hash/length auditing is prepared. Nearby old FASTAs remain candidate evidence unless linked to the old panel run.
- Legacy sums and numerical nm³ integrals are exported separately. The legacy mask-only tail integral can have shorter coverage than its 2 nm divisor; that behavior is retained for reproduction and documented. Actual overlap agreement will decide whether the legacy prediction/simulation identity panel can be exported.
- Primary old/new statistics include all eligible fitted interactions and FOXO4–G3BP1. Exact-sequence, changed-construct, uncertain-identity and historical-exclusion analyses are distinct. A small row-bootstrap difference cannot establish superiority with shared constructs.

Unresolved items are the old-panel prediction run’s exact FASTA linkage, precise supplied new-model training-target provenance, independent RDF sequence/replica-job provenance and final panel-letter assignment. These do not prevent preparing or running the independent inventory, frozen prediction update, diagnostic comparison or separately labelled B22 calculations.

Run `python3 run_validation_workflow.py` in `fig6/`. The local run produces `updated_validation/RESULTS_REPORT.md` with actual metrics and reproduction status, followed by your inspection of every exported PNG. Proposed manuscript/legend changes are documented in that generated report; the DOCX files remain unchanged.
