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
| ---- | ---- | ---- |
|  |  |  |