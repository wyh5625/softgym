# SoftGym
<a href="https://sites.google.com/view/softgym/home">SoftGym</a> is a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope. It is built on top of the Nvidia FleX simulator and has standard Gym API for interaction with RL agents. A number of RL algorithms benchmarked on SoftGym can be found in <a href="https://github.com/Xingyu-Lin/softagent">SoftAgent</a>

## SoftGym Environment for Pushing
|Image|Name|Description|
|----------|:-------------|:-------------|
|![Gif](./examples/ClothPush.gif)|[PushCloth](softgym/envs/cloth_push.py) | RRT* planning for pushing cloth on the floor|
|![Gif](./examples/PantsPush.gif)|[PushPants](softgym/envs/pants_push.py) | RRT* planning for pushing pants on the floor|

## User Guide
Try pushing cloth with GUI:

```
python examples/Manual_ClothPush.py
```

Test RRT* planning by running: 

```
python examples/Control_ClothPush.py
```

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{corl2020softgym,
 title={SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation},
 author={Lin, Xingyu and Wang, Yufei and Olkin, Jake and Held, David},
 booktitle={Conference on Robot Learning},
 year={2020}
}
```

## References
- Instruction for installation of SoftGym engine: https://github.com/Xingyu-Lin/softgym
