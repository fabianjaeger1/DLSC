# Deep Learning in Scientific Computing

This repository contains the machine learning projects completed for the class "Deep Learning in Scientific Computing" taught at [ETH](https://ethz.ch/en.html) jointly by Siddhartha Mishra and Benjamin Moseley in Spring 2024. The description of the tasks can be found in the PDFs.

### Project A

The main objective of the project is to apply machine learning algorithms to solve various tasks related to the preliminary design of a thermal energy storage. In total there are three Tasks.

- Task1: aims at solving a PDE describing the evolution of temperatures of the solid $T_s$ and fluid phase $T_f$ using PINNs. The aim is given a set of input coordinates $(t, x)$ to predict $(T_s, T_f)$.
- Task2: is a PDE constrained inverse Problem where given the fluid temperature and input coordinates $(t,x)$ we want to infer the solid temperature for a set of four different phases of two cycles. This problem is similarly solved using PINNs
- Task3: is a time-series problem. Given a set of noiseless temperature measurements of solid and fluid phase of previous time steps we want to infer the temperatures at a later time step. This problem was solved using Neural Operators, more precisely Fourier Neural Operators (FNO).

### Objective of Course

The objective of this course will be to introduce students to advanced applications of deep learning in scientific computing. The focus will be on the design and implementation of algorithms as well as on the underlying theory that guarantees reliability of the algorithms. We will provide several examples of applications in science and engineering where deep learning based algorithms outperform state of the art methods.

### Content of the Course

1. Issues with traditional methods for scientific computing such as Finite Element, Finite Volume etc, particularly for PDE models with high-dimensional state and parameter spaces.

2. Introduction to Deep Learning: Artificial Neural networks, Supervised learning, Stochastic gradient descent algorithms for training, different architectures: Convolutional Neural Networks, Recurrent Neural Networks, ResNets.

3. Theoretical Foundations: Universal approximation properties of the Neural networks, Bias-Variance decomposition, Bounds on approximation and generalization errors.

4. Supervised deep learning for solutions fields and observables of high-dimensional parametric PDEs. Use of low-discrepancy sequences and multi-level training to reduce generalization error.

5. Uncertainty Quantification for PDEs with supervised learning algorithms.

6. Deep Neural Networks as Reduced order models and prediction of solution fields.

7. Active Learning algorithms for PDE constrained optimization.

8. Recurrent Neural Networks and prediction of time series for dynamical systems.

9. Physics Informed Neural networks (PINNs) for the forward problem for PDEs. Applications to high-dimensional PDEs.

10. PINNs for inverse problems for PDEs, parameter identification, optimal control and data assimilation.

All the algorithms will be illustrated on a variety of PDEs: diffusion models, Black-Scholes type PDEs from finance, wave equations, Euler and Navier-Stokes equations, hyperbolic systems of conservation laws, Dispersive PDEs among others.
