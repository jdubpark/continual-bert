
## Note

Current implementation establishes layerwise connection after the Add & Norm of each sublayer. One can explore other possible implementations, e.g. before the Add & Norm, just one layerwise connection after FFN output, before activation, etc.

In Layer i-1 of KB, hidden states are extracted from:
- MHA after dense and before dropout and norm
- FFN after the second linear layer, before dropout and norm

## Adapted from

**BERT**
- [Original Code](https://github.com/google-research/bert)
- [Huggingface BERT](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py)
- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/bert.py)
- [SciBERT](https://github.com/allenai/scibert)

**BERTSUM***
- [Original Code](https://github.com/nlpyang/PreSumm)

**Elastic Weight Consolidation (EWC)**
- [Original Code](https://github.com/ariseff/overcoming-catastrophic/blob/master/model.py)
- [Continual Learner](https://github.com/GMvandeVen/continual-learning/blob/master/continual_learner.py)
- [PyTorch EWC](https://github.com/kuc2477/pytorch-ewc/blob/master/model.py)

## Some interesting pre-trained models
- [BioBERT](https://github.com/dmis-lab/biobert)
- [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)

***My Notes on:***
- [Transformer](https://mydl.life/)
- [BERT](https://mydl.life/)
- [EWC](https://mydl.life/)
