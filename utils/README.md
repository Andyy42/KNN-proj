# utils `README.md`

## misc

Misc utils for data/format conversion to h5 files

## wespeaker

git submodule of a forked Wespeaker project


VoxLingua107 example: [wespeaker/examples/voxlingua/v2](wespeaker/examples/voxlingua/v2)


### WavLM version

#### How does training work

Run the training with:

[wespeaker/examples/voxlingua/v2/run_WavLM.sh](wespeaker/examples/voxlingua/v2/run_WavLM.sh)


Which runs the `train_V2.py`:

[wespeaker/wespeaker/bin/train_V2.py](wespeaker/wespeaker/bin/train_V2.py)

This does:
* Initialize model with `get_speaker_model` from [wespeaker/wespeaker/models/speaker_model.py](wespeaker/wespeaker/models/speaker_model.py)
* Create the `Dataset` object from: [wespeaker/wespeaker/dataset/dataset_V2.py](wespeaker/wespeaker/dataset/dataset_V2.py)

---

WavLM overview:
* Initialize model with `get_speaker_model` from [wespeaker/wespeaker/models/speaker_model.py](wespeaker/wespeaker/models/speaker_model.py)
* WavLM model init with pooling backends picker: [wespeaker/wespeaker/models/Transformer_WavLM.py](wespeaker/wespeaker/models/Transformer_WavLM.py)
* WavLM default config: [WavLMConfig](wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L162)
* Pooling backends: [wespeaker/wespeaker/models/ssl_backend.py](wespeaker/wespeaker/models/ssl_backend.py)
* WavLM model: [wespeaker/wespeaker/models/ssl/WavLM.py](wespeaker/wespeaker/models/ssl/WavLM.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)

* WavLM modules: [wespeaker/wespeaker/models/ssl/modules.py](wespeaker/wespeaker/models/ssl/modules.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)


##### WavLM model breakdown

WavLM model: [wespeaker/wespeaker/models/ssl/WavLM.py](wespeaker/wespeaker/models/ssl/WavLM.py) from the original [wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)

Extracting features from WavLM with [WavLM.extract_features](wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L320):
1. Use CNN feature extractor: [ConvFeatureExtractionModel.forward](wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L483) (is learnable if `feature_grad_mult > 0`, frozen if ` = 0`) NOTE: 1D convolution. The temporal convolutions have 512 channels with strides
(5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2), resulting
in each output representing about 25ms of audio strode by
20ms.
2. Pass features through `torch.nn.LayerNorm`
3. Pass normalized features through encoder: [TransformerEncoder.forward](wespeaker/wespeaker/models/ssl/WavLM.py?plain=1#L574)
4. Return `cnn_outs` tensor and `layer_results` tensors list (dim: `T x B x C`)

#### How does inference work


