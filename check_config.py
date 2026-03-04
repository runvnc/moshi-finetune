from moshi.models import loaders
import json
ci = loaders.CheckpointInfo.from_hf_repo('nvidia/personaplex-7b-v1')
print(json.dumps(ci.lm_config, indent=2))
