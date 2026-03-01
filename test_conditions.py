import json
from finetune.data.interleaver import InterleavedTokenizer, Interleaver, Sample
from moshi.conditioners import ConditionAttributes

# Let's just check if Sample accepts a dict or if it needs ConditionAttributes
s = Sample(codes=None, condition_attributes={"description": "test"})
print(s)
