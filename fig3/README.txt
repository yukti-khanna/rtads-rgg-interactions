Figure 3 / Fig. S3 reorganized scripts

Folder structure expected:
fig3/
  inputs/
    fasta/
      modified_headers_disordered_regions.fasta
      RGG_disordered_regions.fasta
      combined_rg_regions_ll_7_all.fasta
    tables/
      combined_rg_regions_ll_7_data.csv
      Table_S4.xlsx
    metadata/
      rbp_rgs_list.txt
  outputs/
    main/
    supplement/
    cache/
  scripts/

Scripts:
- plot_fig3_rg_region_properties.py
  Creates main Fig. 3A, 3C, and 3D panels.
- plot_figS3_rg_compact_tract_supplement.py
  Creates Supplementary Fig. S3A-D from Table_S4.xlsx

Notes:
- Panel B is not included because you assemble it in PowerPoint.
- No per-panel headers/titles are drawn in these outputs.
- Grey points in Fig. 3D are darkened for print visibility.
