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
