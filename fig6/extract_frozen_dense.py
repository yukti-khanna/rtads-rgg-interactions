#!/usr/bin/env python3
"""Read saved Keras Dense weights without loading or modifying a model."""
import io
import json
import sys
import zipfile
import h5py
import numpy as np
from pathlib import Path
from workflow_common import inside

archive, output = inside(sys.argv[1]), inside(sys.argv[2])
z = zipfile.ZipFile(archive)
config = json.loads(z.read('config.json'))
result = {'config': np.array(json.dumps(config))}
with h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r') as handle:
    for name, layer in handle['layers'].items():
        for key, value in layer['vars'].items():
            result[name + '/' + key] = np.asarray(value)
np.savez(output, **result)
