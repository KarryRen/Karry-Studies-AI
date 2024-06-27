# LSTM (Long Short-Term Memory)

> 有关 LSTM 的基础理解，已经都放在了主项目的 [**README**](https://github.com/KarryRen/Karry-Studies-AI/blob/main/LiMu/HandsAI/README.md#lstm-long-short-term-memory) 中，在此不对其基础含义做过多赘述。在此主要关注 `torch` 中的具体参数实现形式，实现和理论的对应。



## Core Computation

<img src="https://d2l.ai/_images/lstm-3.svg" style="zoom:100%;" />

Mathematically, suppose that there are $h$ hidden units, the batch size is $n$, and the number of inputs is $d$. Thus, the input is and  $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ the hidden state of the previous time step is $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$. Correspondingly, the gates at time step $t$ are defined as follows: the input gate is $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, the forget gate is $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$, and the output gate is $\mathbf{O}_t \in \mathbb{R}^{n \times h}$. They are calculated as follows:
$$
\begin{split}\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{b}_\textrm{xi} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{hi}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{b}_\textrm{xf} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{hf}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{b}_\textrm{xo} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{ho}),
\end{aligned}\end{split}
$$
where $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$ and $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$ are weight parameters and $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$ are bias parameters. Use **sigmoid functions ($\sigma=\textrm{sigmoid}$)** to map the input values to the **interval $(0, 1)$ .** **So they just like the weight !**

Next we design the memory cell. Since we have not specified the action of the various gates yet, **we first introduce the *input node* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$**. Its computation is similar to that of the three gates described above, but uses a function with a value range for $(-1, 1)$ as the activation function. This leads to the following equation at time step $t$:
$$
\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{b}_\textrm{xc} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{hc}),
$$
where $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$ and $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$ are weight parameters and $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$ is a bias parameter.

In LSTMs, the input gate $\mathbf{I}_t$ governs how much we take new data into account via $\tilde{\mathbf{C}}_t$ and the forget gate $\mathbf{F}_t$ addresses how much of the old cell internal state $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ we retain. Using the Hadamard (elementwise) product operator $\odot$ we arrive at the following update equation:
$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.
$$
If the forget gate is always 1 and the input gate is always 0, the memory cell internal state $\mathbf{C}_{t-1}$ will **remain constant forever,** passing unchanged to each subsequent time step. However, input gates and forget gates give the model the flexibility of being able to learn when to keep this value unchanged and when to perturb it in response to subsequent inputs. **In practice, this design alleviates the vanishing gradient problem, resulting in models that are much easier to train, especially when facing datasets with long sequence lengths.**

Last, we need to define how to compute the output of the memory cell, i.e., the hidden state $\mathbf{H}_t \in \mathbb{R}^{n \times h}$, as seen by other layers. This is where the output gate comes into play. In LSTMs, we first apply $\tanh$ to the memory cell internal state and then apply another point-wise multiplication, this time with the output gate. This ensures that the values of $\mathbf{H}_t$ are always in the interval $(-1, 1)$:
$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).
$$
Whenever the output gate is close to 1, we allow the memory cell internal state to impact the subsequent layers uninhibited, whereas for output gate values close to 0, we prevent the current memory from impacting other layers of the network at the current time step. Note that a memory cell can accrue information across many time steps without impacting the rest of the network (as long as the output gate takes values close to 0), and then suddenly impact the network at a subsequent time step as soon as the output gate flips from values close to 0 to values close to 1.



## Parameter Analyze

With the above introduction, it is easy to realize that the parameter list of a **single-layer** LSTM is actually the following 16:

- $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}}, \mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$
- $\mathbf{b}_\textrm{xi}, \mathbf{b}_\textrm{xf}, \mathbf{b}_\textrm{xo}, \mathbf{b}_\textrm{xc} \in \mathbb{R}^{1 \times h}$
- $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}}, \mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$
- $\mathbf{b}_\textrm{hi}, \mathbf{b}_\textrm{hf}, \mathbf{b}_\textrm{ho}, \mathbf{b}_\textrm{hc} \in \mathbb{R}^{1 \times h}$

A multilayer LSTM is nothing more than a stack of single-layer LSTM, the input $\mathbf{X}_t$ of the $l\textrm{-th}$ layer ($l \ge 2$) is the hidden state $\mathbf{H}^{l-1}_t$ of the previous layer multiplied by **dropout**. When $l \ge 2$, the list of parameters is the same 16, but the shape changes.

- $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}}, \mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{h \times h}$
- $\mathbf{b}_\textrm{xi}, \mathbf{b}_\textrm{xf}, \mathbf{b}_\textrm{xo}, \mathbf{b}_\textrm{xc} \in \mathbb{R}^{1 \times h}$
- $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}}, \mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$
- $\mathbf{b}_\textrm{hi}, \mathbf{b}_\textrm{hf}, \mathbf{b}_\textrm{ho}, \mathbf{b}_\textrm{hc} \in \mathbb{R}^{1 \times h}$

In total, an LSTM with l layers would have $12\times l$ parameters.



## `Torch` Implementation

In `torch`, we define a one-layer LSTM using the following simple line of code:

```python
lstm_layer = nn.LSTM(input_size=16, hidden_size=30, num_layers=1, batch_first=True)
```

The `lstm_layer._parameters` shows that there are **four parameters**:

```
- weight_ih_l0, # shape=(120, 16)
- bias_ih_l0, # shape=(120)
==================================
- weight_hh_l0, # shape=(120, 30)
- bias_hh_l0, # shape=(120)
```

The actual implementation of `torch` differs significantly from the theoretical mathematical formulation: All weights and bias are **concatenated** and **run in parallel**. **How exactly is the bottom calculated ?** Here we try to reproduce it by handwritten LSTM to verify the derivation.



