#export OPENBLAS_NUM_THREADS=1
import torch

ckpt_path = 'models/mbart50-many-to-many/model_fixed.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model_state = checkpoint['model']
embeddings = model_state['encoder.embed_tokens.weight']

from fairseq.data.dictionary import Dictionary

dict_path = 'models/mbart50-many-to-many/dict.en_XX.txt'
d = Dictionary.load(dict_path)

pt_idx = d.index('__pt_XX__')
targets = ['__tr_TR__', '__my_MM__','__ml_IN__','__gl_ES__','__tl_XX__','__te_IN__','__mk_MK__']

target_indices = [d.index(tag) for tag in targets]
for idx in target_indices:
    embeddings[idx] = embeddings[pt_idx].clone()
torch.save(checkpoint, 'models/mbart50-many-to-many/model_modified.pt')
