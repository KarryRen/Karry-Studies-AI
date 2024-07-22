# Networks

## Finance Transformer (FiT)

> A new **Plug-and-Play module** for **Finance Cross-Section Feature** Extraction.

### What is Cross-Section Feature ?

Cross-Section feature refers to the observed values of the **same feature of a group of samples** reflecting an aggregate **at the same time cross-section**, and is one of the common types of sample data.

Cross-section data are generally characterized by the following:
- It is characterized at one point in time, with **no temporal information**.
- The order of the samples on the cross-section is changeable, i.e., the **sample order contains no information**.
- Some samples **may not contain information**.

<img src="./README.assets/What is Cross-Section Feature.png" alt="What is Cross-Section Feature.png" style="zoom:60%;" />

### Why we need FiT ?

Since the cross-section data has several features as described above, we want to have an information extraction module that satisfies the following properties:
- The extracted information of the module is **not affected by the order of the samples**.
- Samples without information **can be ignored**.

These two properties are at the heart of why Finance Transformer was proposed !

### What is the FiT ?

The framework of FiT is shown below.

<img src="./README.assets/FiT_Framework.png" alt="FiT_Framework" style="zoom:50%;" />

### How to use FiT ?

