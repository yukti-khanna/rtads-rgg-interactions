# Numerical conventions

Legacy: -2*pi*sum((g-c)*r^2), without dr; c=trapezoid(masked tail)/delta. This reproduces repository corr1/Hummer-labelled targets and preserves the shorter bin-centre tail span. It is not a physical nm^3 integral. Historical cutoff evidence is 7.5 nm, pending numerical reproduction.

Integrated: -2*pi*trapezoid((g-c)*r^2,r), with exact window endpoints and constant extension over outer half bins. r is in nm, so these outputs are in nm^3. The full tail window avoids the legacy constant-RDF offset artifact. 7.5 nm isolates numerical changes; L/2 additionally implements the manuscript cutoff. Neither calculation replaces the frozen target scale or generates a Kd.

Five stored run columns are retained as five RDF replicas; independence follows their run labels and is not independently established from trajectories. A mean-only payload has n_replicas=0 (unknown), n_rdf_curves=1 and no uncertainty. Negative mean B22 is transformed after averaging. Positive/zero means are retained with a missing log.

See historical_reproduction.csv and historical_panel_reproduction.csv for actual local-run agreement. The supplied frozen-model target scale is inferred from local historical code; exact new training provenance remains unresolved.
