# ZeRO
### Lets Understand this First
The memory consumption during model training can be divided into two main parts:

1. **Majority memory occupied by Model States**
    - Include optimizer states (such as momentum and variances in Adam), gradients and parameters.
2. **Remaining memory occupied by Residual States**
    - Include activation, temporary buffers, and unusable fragmented memory.

### How DeepSpeed Optimize Memory
To be exact DeepSpeed implements **Zero Redundancy Optimizer** (**ZeRO**), a novel memory optimization technology for large-scale distributed deep learning.
**Optimizing Model State Memory**
For this optimization, **Zero-powered data parallelism** (**ZeRO-DP**) is introduced which combines:

- Communication Efficiency of Data Parallelism (DP) with
- Memory Efficiency of Model Parallelism (MP).

ZeRO-DP has three main optimization stages (as shown in Figure 1):

![](./images/zero-dp-stages.png)

![](./images/memory-communication-of-zero.png)
**Optimizing Residual State Memory**

**ZeRO-R** is developed to optimize residual memory consumed by activation, temporary buffers, and unusable fragmented memory respectively.

1. Activations (stored from forward pass for backward pass) are optimized through activation partitioning to remove replication in existing MP approaches. Offloading activations to CPU when appropriate also helps in optimizing activation memory.
2. Temporary buffers have an appropriate size defined by ZeRO-R to balance memory and computation efficiency effectively.
3. Fragmented memory during training due to varying lifetimes of tensors can lead to allocation failures even with enough free space. ZeRO-R proactively manages this fragmentation issue based on tensor lifetimes.
# Visualize ZeRO-DP
| Steps | Visualization | Comment |
| :--: | :--- | :--- |
| 1 | ![](./Zero-visualized/1.png) |  |
| 2 | ![](./Zero-visualized/2.png) | **Split the whole data into 4 equal parts for each GPU to process.** |
| 3 | ![](./Zero-visualized/3.png) |  |
| 4 | ![](./Zero-visualized/4.png) | **GPU Memory Used by Model Parameters, Gradients and Optimizers** |
| 5 | ![](./Zero-visualized/5.png) |  |
| 6 | ![](./Zero-visualized/6.png) |  |
| 7 | ![](./Zero-visualized/7.png) | **To effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.** |
| 8 | ![](./Zero-visualized/8.png) |  |
| 9 | ![](./Zero-visualized/9.png) |  |
| 10 | ![](./Zero-visualized/10.png) | **Forward Pass starts from this step** |
| 11 | ![](./Zero-visualized/11.png) | **Broadcasts model parameters of GPU 0 to other GPUs.** |
| 12 | ![](./Zero-visualized/12.png) |  |
| 13 | ![](./Zero-visualized/13.png) | **Once Forward Pass is completed, delete model parameters of GPU 0 stored in GPU 1, 2, and 3.** |
| 14 | ![](./Zero-visualized/14.png) | **Replicate the process performed in steps 11, 12, and 13 on the remaining GPUs as well.** |
| 15 | ![](./Zero-visualized/15.png) |  |
| 16 | ![](./Zero-visualized/16.png) | **Once forward pass is completed, calculate loss on each GPU for its respective dataset.** |
| 17 | ![](./Zero-visualized/17.png) | **Backward Propagation from this step** |
| 18 | ![](./Zero-visualized/18.png) |  |
| 19 | ![](./Zero-visualized/19.png) |  |
| 20 | ![](./Zero-visualized/20.png) |  |
| 21 | ![](./Zero-visualized/21.png) | **Replicate the process performed in steps 18, 19, and 20 on the remaining GPUs as well.** |
| 22 | ![](./Zero-visualized/22.png) |  |
| 23 | ![](./Zero-visualized/23.png) |  |
| 24 | ![](./Zero-visualized/24.png) | **Parameter Update start from this step** |
| 25 | ![](./Zero-visualized/25.png) |  |
| 26 | ![](./Zero-visualized/26.png) |  |
| 27 | ![](./Zero-visualized/27.png) | **Training Iteration is completed** |
