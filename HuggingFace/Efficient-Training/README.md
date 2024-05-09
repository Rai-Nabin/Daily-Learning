# Methods and Tools for Efficient Training on a Single GPU
*Resources*

| S.N | Topics | Link |
| :--: | ---- | :--: |
| 1 | Methods and Tools for Efficient Training on a Single GPU | [link](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#gradient-accumulation) |
| 2 | Performance and Scalability: How to Fit a Bigger Model and Train It Faster | [link](https://huggingface.co/docs/transformers/v4.18.0/en/performance) |


**When training large models, there are two aspects that should be considered at the same time:**
- Data Throughput/Training Time
- Model Performance

**What is Data Throughput?**

Data Throughput means "the speed at which training data is fed into the model during training process". Higher data throughput means that the model can process more training examples in a given amount of time, leading to faster training.

**How to maximize Data Throughput and minimize Training Time?**

- Utilize the GPU as much as possible
- Use techniques like Gradient Accumulation when the desired batch sized exceeds GPU memory limits, but avoid them if the batch size fits comfortably within memory to prevent from slow training time.

![Techniques comparison](./Images/Overview.png)
# Gradient Accumulation
**What is Gradient Accumulation?**

Gradient accumulation is a technique used in training neural networks to overcome memory limitations and support larger batch sizes.

> **How mini-batch stochastic gradient descent updates model parameters?**
> 
> The model parameters are updated after processing each individual batch of training data.

![Gradient Accumulation](./Images/gradient-accumulation.png)

In Gradient Accumulation, the gradients  are accumulated over multiple batches before updating the model parameters. Specifically, the gradients are computed for each mini-batch, but the parameter updates are only performed after a certain number of mini-batches have been processed (e.g. every 4 batches).
![Advantages of Gradient Accumulation](./Images/g-a-advantages.png)

**References:**
- [Gradient Accumulation](https://www.hopsworks.ai/dictionary/gradient-accumulation)
- [What is Gradient Accumulation in Deep Learning?](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
# Gradient Checkpointing
**How Vanilla Back Propagation Works?**
1. Forward pass to compute and store activations of all layers
2. Backward pass to compute gradients at each layer

![Vanilla Back Propagation](./Images/vanilla-back-prop.png)

In vanilla back propagation, the required memory grows linearly with the number of layers _n_ in the neural network. This is because all nodes from the forward pass are being kept in memory (until all their dependent child nodes are processed).

**How Gradient Checkpointing Works?**

**Forward Pass:**
- Only certain intermediate activations (checkpoints) are stored in memory during the forward pass.
- The selection of these checkpoints can be based on various strategies, such as every 'n' layers or based on specific memory constraints.

**Backward Pass:**
- Some intermediate activations were not stored (due to checkpointing), they need to be recomputed before calculation. This recomputation involves redoing parts of the forward pass from the last checkpoint to the desired layer.
- The gradient is then computed using the recomputed activations.
![Gradient Checkpointing](./Images/gradient-checkpointing.png)

**Benefits of Gradient Checkpointing:**
- **Reduced Memory Usage**: By recomputing checkpointed activations, gradient checkpointing reduces the memory required to store the gradients, which can be particularly useful when training large models or processing large datasets.
- **Increased Effective Batch Size**: Gradient checkpointing allows for training with larger effective batch sizes, which can improve model performance and convergence.
- **Improved Training Stability**: By reducing the memory footprint, gradient checkpointing can help stabilize the training process, particularly when working with batch sizes that are too large to fit into memory.

**References:**
- [Gradient Checkpointing Demo](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/gradient-checkpointing-nin.ipynb)
- [Gradient Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/#:~:text=(accumulated_gradients)-,Gradient%20Checkpointing,-Gradient%20checkpointing%20is)
- [Fitting Larger Networks into Memory](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
