# Introduction

Convex optimization is a fundamental problem in many applications, where the goal is to minimize the sum of a smooth function and a nonsmooth function. The traditional approach for solving these problems, known as proximal gradient descent (ProxGD), requires evaluating the gradient of the smooth function and the prox operator of the nonsmooth function in each iteration. However, in practical applications, the cost of evaluating the prox operator can be significantly higher than that of evaluating the gradient. This motivates the development of a more efficient method that minimizes the number of expensive prox evaluations. In this paper, the authors propose ProxSkip, a simple yet highly efficient method that allows for skipping the prox operator in most iterations. The key idea behind ProxSkip is to achieve an effective acceleration of communication complexity in federated learning scenarios, where the evaluation of the prox operator corresponds to costly communication between devices. Unlike existing methods, ProxSkip provides a substantial improvement in communication complexity without any assumptions on the heterogeneity of data.

# Problem Statement
The paper focuses on the problem defined as

$$\min_{x\in\mathbb{R}^d}f(x) +\psi(x)$$

where $f\ :\ \mathbb{R}^d\rightarrow\mathbb{R}$ and $\psi\ :\ \mathbb{R}^d\rightarrow\mathbb{R}\cup\{+\infty\}$ is a proper, closed and convex regularizer

The problem appears in a huge variety of domains. Authors address a particular case of that problem when calculation of the $\psi$ is computationally expensive.

## Federated Learning

Federated Learning (FL) is a widely used approach in distributed machine learning settings where devices collaboratively train a shared model without sharing raw data. In FL, communication between devices is a major bottleneck due to the high cost of transmitting data. To address this issue, various methods have been proposed to reduce communication, such as delayed communication where devices perform multiple local steps independently based on their local objective. However, when each device has data drawn from a different distribution, these local steps introduce client drift, leading to convergence issues. Several methods, including Scaffold, S-Local-GD, and FedLin, have been developed to mitigate client drift and improve communication complexity. However, despite their empirical success, their theoretical communication complexity does not surpass that of vanilla gradient descent (GD). Consequently, it remains challenging to establish theoretically that performing independent local updates improves communication complexity compared to GD.

Despite the empirical advantages of methods like Scaffold, S-Local-GD, and FedLin over vanilla gradient descent (GD) in federated learning, their theoretical communication complexity does not surpass that of GD. This implies a fundamental gap in our understanding of local methods. Extensive efforts by the FL community have been made to bridge this gap, but establishing a theoretical improvement in the communication complexity of local updates compared to GD has proven challenging. In contrast, accelerated gradient descent (without local steps) can achieve the optimal communication complexity of $O\big(\sqrt{\kappa} \log \frac{1}{\varepsilon}\big)$ . This leads to the important question of whether the limitation in communication complexity is inherent to local gradient-type methods. Or in other words the question is whether it is possible to reach a complexity better than $O\big(\kappa \log \frac{1}{\varepsilon}\big)$ for simple local gradient-type methods without relying on explicit acceleration mechanisms.

# Related Work

## Basis of the Paper

The work is based on a classical proximal gradient descent with a certain extension. The main idea of the extension is to be able to skip prox function computations. The approach introduces a control variate that helps to achieve provable convergence in a convex setting with an ability to skip an amount of steps sufficient for reaching forementioned compexity.

# Contribution

Our contribution is using Scaffnew method in training of deep neural network. We trained ResNet-18 model with PyTorch framework using our own implementation of Scaffnew. We simulated a multi-device environment and explored different settings for $p$. For these experiments we additionally used learning rate decaying and mini-batch loss calculation.

To reproduce the training run:

```bash
cd proxskip

python resnet18/train.scaffnew.py
```

We have saved and shared weights of the experiments. They are stored in proxskip/resnet18/assets. You can evaluate a certain checkpoint using the following command: 

```bash
cd proxskip

python resnet18/eval.py -w <PATH-TO-WEGHTS>
```

Also, to have a result to compare with, we have implemented a training loop of ResNet-18 with SGD algorithm (with no momentum). The algorithm is runnable using the following command:

```bash
cd proxskip

python resnet18/sgd.py
```

Files tagged with losses, such are `resnet18.losses.*` are losses saved during the training of each of experiments. They are used to produce plots.

