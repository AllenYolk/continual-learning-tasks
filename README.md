# continual-learning-tasks

`cltask` is a package containing a series of continual learning tasks implemented by Pytorch.

## Task List

* **Sequential Nonlinear Regression**: (Camp et al., 2020; Flennerhag et al., 2020)
* **Permuted Benchmark (e.g. MNIST)**: a.k.a. "Shuffled Benchmark" (Goodfellow et al., 2015; Kirkpatrick et al., 2017; Masse et al., 2018; Wu et al., 2022; Zenke et al., 2017)

## TODO

* [x] Refine the interface of `plot_sequential_regression()`.
* [x] Implement **Permuted Benchmark** task.
* [ ] Add new continual learning tasks.

## References

* Camp, B., Mandivarapu, J. K., & Estrada, R. (2020). Continual Learning with Deep Artificial Neurons (arXiv:2011.07035). arXiv. https://doi.org/10.48550/arXiv.2011.07035
* Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., & Hadsell, R. (2020). Meta-Learning with Warped Gradient Descent (arXiv:1909.00025). arXiv. https://doi.org/10.48550/arXiv.1909.00025
* Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2015). _An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks_ (arXiv:1312.6211). arXiv. https://doi.org/10.48550/arXiv.1312.6211
* Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. _Proceedings of the National Academy of Sciences_, _114_(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114
* Masse, N. Y., Grant, G. D., & Freedman, D. J. (2018). Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. _Proceedings of the National Academy of Sciences_, _115_(44). https://doi.org/10.1073/pnas.1803839115
* Wu, Y., Zhao, R., Zhu, J., Chen, F., Xu, M., Li, G., Song, S., Deng, L., Wang, G., Zheng, H., Ma, S., Pei, J., Zhang, Y., Zhao, M., & Shi, L. (2022). Brain-inspired global-local learning incorporated with neuromorphic computing. _Nature Communications_, _13_(1), 1–14.
* Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. _Proceedings of the 34th International Conference on Machine Learning_, 3987–3995. https://proceedings.mlr.press/v70/zenke17a.html

