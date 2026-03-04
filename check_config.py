import json, sys
print(f'moshi from: ', end='')
import moshi
print(moshi.__file__)

from moshi.models import loaders

if not hasattr(loaders, 'CheckpointInfo'):
    print('ERROR: This moshi (personaplex fork) does not have CheckpointInfo.')
    print('Run with upstream kyutai moshi on the path to use this script.')
    print('Or just check: does /workspace/personaplex/moshi/moshi/models/loaders.py have depformer_weights_per_step=True in _lm_kwargs?')
    sys.exit(1)

ci = loaders.CheckpointInfo.from_hf_repo('nvidia/personaplex-7b-v1')
print(json.dumps(ci.lm_config, indent=2))
