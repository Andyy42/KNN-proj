# KNN Project


To clone with other submodules run this command:

```
git clone --recursive https://github.com/Andyy42/KNN-proj.git
```

Project structure:
* `utils/wespeaker`: Git submodule for WeSpeaker fork.
* `utils/misc`: Misc utils for data/format conversion to h5 files
* `src/classifier/glc`: Simple GLC Classifier
* `src/diarization/`: Diarization


## `utils` folder


* `utils/misc`: Misc utils for data/format conversion to h5 files
* `utils/wespeaker`: [WeSpeaker](https://github.com/wenet-e2e/wespeaker) submodule mainly focuses on speaker embedding learning, with application to the speaker verification task. We support online feature extraction or loading pre-extracted features in kaldi-format. The example for VoxLingua ResNet-18 and WavLM.

### wespeaker

git submodule of a forked WeSpeaker project

All files with suffix `_V2` are customized version of the files without the suffix to support WavLM. The WavLM model is from [microsoft unilm wavlm](https://github.com/microsoft/unilm/tree/master/wavlm). The MHFA backend and Last layer ASTP was adapted from (SLT22_MultiHead-Factorized-Attentive-Pooling)[https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling].

We implemented custom VoxLingua107 example which also contains parts of code for NAKI dataset. Our VoxLingua107 example: [utils/wespeaker/examples/voxlingua/v2](utils/wespeaker/examples/voxlingua/v2)

All the WeSpeaker configuration for the ML models is in [utils/wespeaker/examples/voxlingua/v2/conf](utils/wespeaker/examples/voxlingua/v2/conf/)


#### How does training work

Run the training with:

[wespeaker/examples/voxlingua/v2/run_WavLM.sh](utils/wespeaker/examples/voxlingua/v2/run_WavLM.sh)


Which runs the `train_V2.py`:

[wespeaker/wespeaker/bin/train_V2.py](utils/wespeaker/wespeaker/bin/train_V2.py)

This does:
* Initialize model with `get_speaker_model` from [wespeaker/wespeaker/models/speaker_model.py](utils/wespeaker/wespeaker/models/speaker_model.py)
* Create the `Dataset` object from: [wespeaker/wespeaker/dataset/dataset_V2.py](utils/wespeaker/wespeaker/dataset/dataset_V2.py)

---

WavLM overview:
* Initialize model with `get_speaker_model` from [wespeaker/wespeaker/models/speaker_model.py](utils/wespeaker/wespeaker/models/speaker_model.py)
* WavLM model init with pooling backends picker: [wespeaker/wespeaker/models/Transformer_WavLM.py](utils/wespeaker/wespeaker/models/Transformer_WavLM.py)
* WavLM default config: [WavLMConfig](utils/wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L162)
* Pooling backends: [wespeaker/wespeaker/models/ssl_backend.py](utils/wespeaker/wespeaker/models/ssl_backend.py)
* WavLM model: [wespeaker/wespeaker/models/ssl/WavLM.py](utils/wespeaker/wespeaker/models/ssl/WavLM.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)

* WavLM modules: [wespeaker/wespeaker/models/ssl/modules.py](utils/wespeaker/wespeaker/models/ssl/modules.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)


##### WavLM model breakdown

WavLM model: [wespeaker/wespeaker/models/ssl/WavLM.py](utils/wespeaker/wespeaker/models/ssl/WavLM.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)

Extracting features from WavLM with [WavLM.extract_features](utils/wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L320):
1. Use CNN feature extractor: [ConvFeatureExtractionModel.forward](utils/wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L483) (is learnable if `feature_grad_mult > 0`, frozen if ` = 0`) NOTE: 1D convolution. The temporal convolutions have 512 channels with strides
(5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2), resulting
in each output representing about 25ms of audio strode by
20ms.
2. Pass features through `torch.nn.LayerNorm`
3. Pass normalized features through encoder: [TransformerEncoder.forward](utils/wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L574)
4. Return `cnn_outs` tensor and `layer_results` tensors list (dim: `T x B x C`)



