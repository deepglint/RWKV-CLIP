# RWKV-CLIP: A Robust Vision-Language Representation Learner


> **[RWKV-CLIP: A Robust Vision-Language Representation Learner](https://arxiv.org/abs/2406.06973)** <br>
Tiancheng Gu,</span>
<a href="https://kaicheng-yang0828.github.io">Kaicheng Yang</a>,</span>
Xiang An,</span>
Ziyong Feng,</span>
Dongnan Liu,</span>
<a href="https://weidong-tom-cai.github.io/">Weidong Cai</a>,</span>
<a href="https://jiankangdeng.github.io">Jiankang Deng</a></span>


## 📣 News
- [2024/06/11]:✨The paper of [RWKV-CLIP](https://arxiv.org/abs/2406.06973) is submitted to arXiv.
  
## 💡 Introduction
We introduce a diverse description generation framework that can leverage Large Language Models(LLMs) to synthesize and refine content from web-based texts, synthetic captions, and detection tags. Beneficial form detection tags, more semantic information can be introduced from images, which in turn further constrains LLMs and mitigates hallucinations.

![teaser](figure/Diverse_description_generation_00.png)

We propose RWKV-CLIP, the first RWKV-driven vision-language representation learning model that combines the effective parallel training of transformers with the efficient inference of RNNs.

![teaser](figure/RWKV_architecture_00.png)


## 🎨 In-Progress
- [ ] Release training code
- [ ] Release pretrain model weight
- [ ] Release the generated diverse descriptions of YFCC15M



## 📖 Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@misc{gu2024rwkvclip,
      title={RWKV-CLIP: A Robust Vision-Language Representation Learner}, 
      author={Tiancheng Gu and Kaicheng Yang and Xiang An and Ziyong Feng and Dongnan Liu and Weidong Cai and Jiankang Deng},
      year={2024},
      eprint={2406.06973},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgements
This project is based on [RWKV](https://github.com/BlinkDL/RWKV-LM), [VisionRWKV](https://github.com/OpenGVLab/Vision-RWKV), [RAM++](https://github.com/xinyu1205/recognize-anything), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [vllm](https://github.com/vllm-project/vllm), [OFA](https://github.com/OFA-Sys/OFA), and [open_clip](https://github.com/mlfoundations/open_clip), thanks for their works.

### License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=deepglint/RWKV-CLIP&type=Date)](https://star-history.com/#deepglint/RWKV-CLIP&Date)