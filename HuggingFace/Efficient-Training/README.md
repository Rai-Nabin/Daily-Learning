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