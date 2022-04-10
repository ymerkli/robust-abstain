# mrevadv

Models in this directory are trained with abstain training with loss function

```math
(1 - \sigma(f(x_{nat}))_y) \cdot CE(f_\theta(x_{nat}), \arg\max_{k \neq F(x_{nat})} f_\theta(x_{adv})_k) + \beta \cdot TRADES(f_\theta, x_{nat}, x_{adv}, y)
```

where:
- $`(x_{nat}, y)`$ is the ground truth sample
- $`x_{adv} = x_{nat} + \delta, \delta = \arg\max_\{||\delta||_p \leq \epsilon} l(f_\theta(x_{nat}+\delta), y)`$ is a corresponding adversarial samples
- $`CE()`$ is the crossentropy loss
- $`TRADES()`$ is the Trades loss presented by Zhang et. al. (https://arxiv.org/abs/1901.08573),
- $`\sigma()`$ is the softmax function
- $`f(x)`$ is the soft classifer returning logits
- $`F(x) = \arg\max_k f(x)_k`$ is the hard classifier returning top1 label from the output of $`f(x)`$