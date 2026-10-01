# ai-image-edit-qwen

Qwen-Image-2.1 inference for [AI Image Edit](https://github.com/pulb/ai_image_edit):
the diffusers pipeline loader, the ZeroGPU duration budget, and the
optional AOTI-compiled kernels. AI Image Edit's `qwen_image` backend
imports this package; nothing here imports AI Image Edit.

```
ai_image_edit_qwen/
  pipeline.py   # load() and generate(): pipeline setup and the @spaces.GPU diffusion call
  aoti.py       # AOTI kernels for the transformer blocks and VAE decoder
```

## Install

Install the CUDA build of torch you need first, then:

```bash
pip install git+https://github.com/pulb/ai_image_edit_qwen
```

This pulls in the pinned diffusers commit (`QwenImage21Pipeline` is not in
a released diffusers version yet), transformers, accelerate, torchvision,
pillow and `spaces`.

## API

```python
from ai_image_edit_qwen import pipeline

pipeline.load(aoti_repo="hugging-apps/qwen-image-2-1-aoti", aoti_token=None, use_aoti=True)
image = pipeline.generate(
    prompt, image_paths, negative_prompt, true_cfg_scale,
    num_inference_steps, seed, resolution, width, height,
)  # -> PIL.Image
```

`load()` is called once and needs a CUDA device. `generate()` runs under
`spaces.GPU` on ZeroGPU and as a plain call elsewhere.

## License

The code in this repository is under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE) (SPDX:
`LicenseRef-Qwen-Research-License-Agreement`), not the GPL. It is derived
from the reference Space
[`hugging-apps/qwen-image-2-1`](https://huggingface.co/spaces/hugging-apps/qwen-image-2-1).
`NOTICE` carries the attribution that agreement requires.

The agreement grants rights **for non-commercial purposes only**
(research or evaluation); commercial use needs a separate license from
Alibaba. The same agreement governs the Qwen-Image-2.1 weights.
