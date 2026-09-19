# Radix Embodied Stack

Universal infrastructure for embodied AI: model quantization, inference runtime, and robot control for domestic chips.

## Overview

Radix Embodied Stack enables **cross-model, cross-chip, cross-platform** deployment of embodied AI models (VLA, VLM+Policy, Diffusion Policy, World-Action Models) on domestic computing platforms.

**Key Features:**
- 🔧 Model quantization and optimization
- ⚡ Inference runtime with hardware adaptation  
- 🤖 Multi-robot control interface
- 🎯 Support for domestic chips

## Supported Platforms

**Domestic chips:**
TPU/GPU/NPU/IPU

**Robots:**
- Humanoid robots 
- Quadruped robots
- Mobile manipulators
- Robotic arms
- Dexterous hands 

## Architecture

```
Model Layer     →  VLA / VLM+Policy / Diffusion Policy
    ↓
Infra Layer     →  Quantization / Runtime / Control
    ↓
Hardware Layer  →  Domestic Chips / Robot Platforms
```

## Roadmap

- [ ] Model quantization toolkit
- [ ] Inference runtime for domestic chips
- [ ] Robot control adapters
- [ ] Benchmarks and performance profiles
- [ ] Documentation and tutorials

## Contributing

Contributions welcome! This project is part of the [RadixRootMind](https://github.com/RadixRootMind) open-source community.

## License

TBD

---

**Project Status:** 🚧 Under active development
