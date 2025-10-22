# Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement
### Ye-Xin Lu, Yang Ai, Zhen-Hua Ling
In our [paper](https://arxiv.org/abs/2305.13686), we proposed MP-SENet: a TF-domain monaural SE model with parallel magnitude and phase spectra denoising.<br>
A [long-version](https://arxiv.org/abs/2308.08926) MP-SENet was extended to the speech denoising, dereverberation, and bandwidth extension tasks.<br>
Audio samples can be found at the [demo website](http://yxlu-0102.github.io/MP-SENet).<br>
We provide our implementation as open source in this repository.

---

## ⚙️ Modifications and Experiments

This implementation extends the original **MP-SENet** by adding support for alternative block types defined in the configuration file `config.json`.  
The type of block used in the model is set via the `block_type` field and can take one of the following values:
- `"mamba"`
- `"xlstm"`
- `"mixed"` (combination of xLSTM and Mamba blocks)
- `None` (defaults to the baseline Transformer-based architecture) 

Experiments were performed using baseline architecture, Mamba blocks and a **mixed architecture** (xLSTM blocks for frequency spectrum + Mamba for time spectrum). RA working implementation of the **xLSTM block** was also developed. However, due to GPU memory limitations, it was not possible to run full validation or inference. The code itself is functional and can be successfully trained and evaluated on more powerful GPUs. More about it can be found in `experiments.ipynb` notebook. Overall, the mixed architecture turn out to be pretty successful with **3.434 pesq** when trained with `learning_rate = 0.0008` for only 20 epochs. This modification of original architecture seems to be a good compromise between speed and quality, as the model works pretty fast (especially compared to the modification using purely xLSTM blocks). More on results of the experiments can be found in `experiments.ipynb` notebook.

Recent works — [MP-SENet-Mamba (Zhao et al., 2025)](https://arxiv.org/pdf/2501.06146v2) and [MP-SENet-xLSTM (Liu et al., 2024)](https://arxiv.org/pdf/2405.06573) — demonstrate that integrating these advanced sequence modeling blocks can further enhance speech enhancement performance.  
Given the results obtained here, with more powerful hardware and longer training (more epochs), similar or higher metrics are achievable, compared to the ones in articles mentioned.

---

## 🧠 Metrics

In addition to standard metrics (PESQ, STOI, SI-SDR, etc.), the **evaluation metrics** from the [URGeNT Challenge 2026](https://github.com/urgent-challenge/urgent2026_challenge_track1/blob/main/evaluation_metrics/) were implemented in `cal_metrics/compute_metrics.py` for a more comprehensive assessment of model performance.

---

## ⚠️ Note
There is a small bug in our code, but it does not affect the overall performance of the model. 
If you intend to retrain the model, it’s **strongly recommended** to set `batch_first=True` in the `MultiHeadAttention` module inside [transformer.py](models/transformer.py), which can significantly reduce the memory usage of the model.

## Pre-requisites
1. Python == 3.13.
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](https://github.com/yxlu-0102/MP-SENet/blob/main/requirements.txt).
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). Resample all wav files to 16kHz, and move the clean and noisy wavs to `VoiceBank+DEMAND/wavs_clean` and `VoiceBank+DEMAND/wavs_noisy`, respectively. You can also directly download the downsampled 16kHz dataset [here](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l).

## Training
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --config config.json
```
Checkpoints and copy of the configuration file are saved in the `cp_mpsenet` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

## Inference
```
python inference.py --checkpoint_file [generator checkpoint file path]
```
You can also use the pretrained best checkpoint files we provide in the `best_ckpt` directory.
<br>
Generated wav files are saved in `generated_files` by default.
You can change the path by adding `--output_dir` option.<br>
Here is an example:
```
python inference.py --checkpoint_file best_ckpt/g_best_vb --output_dir generated_files/MP-SENet_VB
```

## Model Structure
![model](Figures/model.png)

## Comparison with other SE models
![comparison](Figures/table.png)

## Acknowledgements
We referred to [HiFiGAN](https://github.com/jik876/hifi-gan), [NSPP](https://github.com/YangAi520/NSPP) 
and [CMGAN](https://github.com/ruizhecao96/CMGAN) to implement this.

## Citation
```
@inproceedings{lu2023mp,
  title={{MP-SENet}: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={3834--3838},
  year={2023}
}

@article{lu2023explicit,
  title={Explicit estimation of magnitude and phase spectra in parallel for high-quality speech enhancement},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  journal={Neural Networks},
  volume = {189},
  pages = {107562},
  year={2025}
}
```
