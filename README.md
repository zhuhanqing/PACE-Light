# PACE

By [Hanqing Zhu](https://github.com/zhuhanqing), Wenyan Cong, Guojin Chen, Ray T. Chen, Jiaqi Gu and David Z. Pan.

This repo is the official implementation of "[PACE: Pacing Operator Learning to Accurate Optical Field Simulation for Complicated Photonic Devices](https://arxiv.org/abs/2411.03527)".


# Introduction

**PACE* is a state-of-the-art, open-source neural operator designed to revolutionize electromagnetic field simulation for photonic devices. Traditional numerical methods, while accurate, often present significant bottlenecks due to their high computational costs, limiting the scalability and speed of photonic circuit design.

**PACE** addresses the challenges faced by existing neural operator approaches, such as NeurOLight, which struggles to predict high-fidelity fields for complex real-world photonic devices, achieving only a 0.38 normalized mean absolute error. By introducing a novel cross-axis factorized PACE operator, we empower simulations with enhanced long-distance modeling capabilities, connecting intricate local device structures with full-domain electromagnetic field patterns.

Inspired by human learning, **PACE** adopts a two-stage "divide and conquer" approach to solve extremely challenging cases: the first stage learns an initial solution, while the second model refines it to achieve unprecedented accuracy. On benchmark tests for complex photonic devices, **PACE** delivers a 73% reduction in error compared to prior approaches, using 50% fewer parameters.

In terms of speed, **PACE** achieves a remarkable **154-577×** simulation speedup over traditional numerical solvers like scipy, and a **12×** speedup over optimized solvers like pardiso, making it the ideal solution for high-fidelity and fast photonic device simulations.

![PACE-Light Flow](figs/Neurips_pace.jpg)

### Key Features:
- **Cross-axis factorized operator**: Efficiently models long-range interactions in photonic simulations.
- **Two-stage refinement**: Significantly improves prediction fidelity, even for the most intricate cases.
- **Unparalleled speedup**: Up to **577× faster** than conventional methods, enabling ultra-fast simulations.
- **Parameter-efficient design**: Achieves higher accuracy with fewer parameters than existing solutions.


# Dependencies
* Python >= 3.6
* pyutils >= 0.0.1. See [pyutils](https://github.com/JeremieMelo/pyutility) for installation.
* pytorch-onn >= 0.0.5. See [pytorch-onn](https://github.com/JeremieMelo/pytorch-onn) for installation.
* Python libraries listed in `requirements.txt`
* NVIDIA GPUs and CUDA >= 10.2

# Structures
* configs/: configuration files
* core/
    * models/
        * layers/
            * pace_conv2d: PACE block definition
        * pace_cnn.py: NeurOLight model definition
        * fno_cnn.py: PACE model definition
        * pde_base.py: base model definition
        * utils.py: utility functions
        * constant.py: constant definition
    * builder.py: build training utilities
    * utils.py: customized loss function definition
* scripts/: contains experiment scripts
* data/: MMI simulation dataset
* train.py: training logic
* test.py: inference logic
* refine.py: refine logic

# Usage

TO BE UPDATED

# Citing NeurOLight
```
@inproceedings{zhu2024Pace,
  title={PACE: Pacing Operator Learning to Accurate Optical Field Simulation for Complicated Photonic Devices},
  author={Hanqing Zhu, Wenyan Cong, Guojin Chen, Shupeng Ning, Ray Chen, Jiaqi Gu, and David Z. Pan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
