# continual-learning-tasks

`cltask` is a package containing a series of continual learning tasks implemented by Pytorch.

## Installation

### Install from Source Code

From [Github](https://github.com/AllenYolk/continual-learning-tasks):
```shell
git clone https://github.com/AllenYolk/continual-learning-tasks.git
cd continual-learning-tasks
pip install .
```

## Task List

* **Sequential Nonlinear Regression**: (Camp et al., 2020; Flennerhag et al., 2020)
* **Permuted Benchmark (e.g. MNIST)**: a.k.a. "Shuffled Benchmark" (Goodfellow et al., 2015; Kirkpatrick et al., 2017; Masse et al., 2018; Wu et al., 2022; Zenke et al., 2017; Yang et al., 2022)
* **Split10 CIFAR100**: (Zenke et al., 2017)

## TODO

* [x] Refine the interface of `plot_sequential_regression()`.
* [x] Implement **Permuted Benchmark** task.
* [x] Refactor permuted benchmark task to get better efficiency.
* [x] Modify the docs for permuted benchmark.
* [ ] Add new continual learning tasks.
* [x] Add installation guide.

## References

* Camp, B., Mandivarapu, J. K., & Estrada, R. (2020). Continual Learning with Deep Artificial Neurons (arXiv:2011.07035). arXiv. 
* Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., & Hadsell, R. (2020). Meta-Learning with Warped Gradient Descent (arXiv:1909.00025). arXiv. 
* Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2015). _An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks_ (arXiv:1312.6211). arXiv. 
* Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. _Proceedings of the National Academy of Sciences_, _114_(13), 3521–3526. 
* Masse, N. Y., Grant, G. D., & Freedman, D. J. (2018). Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. _Proceedings of the National Academy of Sciences_, _115_(44). 
* Wu, Y., Zhao, R., Zhu, J., Chen, F., Xu, M., Li, G., Song, S., Deng, L., Wang, G., Zheng, H., Ma, S., Pei, J., Zhang, Y., Zhao, M., & Shi, L. (2022). Brain-inspired global-local learning incorporated with neuromorphic computing. _Nature Communications_, _13_(1), 1–14.
* Yang, Y., Xu, M., Pei, J., Li, P., Li, G., Wu, S., & Li, H. (2022). Bio-realistic and versatile artificial dendrites made of anti-ambipolar transistors. Wiley-VCH.
* Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. _Proceedings of the 34th International Conference on Machine Learning_, 3987–3995.

