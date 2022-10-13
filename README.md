<h1 align='center'>Initial LinearWarmup and Max LR Decay <br>for PyTorch CosineAnnealingWarmRestarts</h3>


Simple PyTorch CosineAnnealingWarmRestarts LR scheduler modification that adds both an initial linear warmup and max learning rate decay. 
Inherits from the base PyTorch _LRScheduler class and supports standard flags like <code>verbose</code>. 

<b>Note:</b> It is not a package currently and is primarily given here as a resource for people to use.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/195468405-40448b88-4c2d-4d20-a11b-39a4e6d70d25.png" alt="latent variable schematic" /></p>
<p align='center'>Fig 1. Learning rate values over 120k iterations with a <code>T_0</code> of 10k and a <code>decay</code> of 0.75.</p>

## Usage
The new class args added are:
<ul>
<li>warmup_steps (int): how many batch steps are warmed up over</li>
<li>decay (float): decay factor of the max LR decay (default=1)</li>
</ul>

Example: <code>cyclic_scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(encoder_optim, T_0=10000, T_mult=1, warmup_steps=2000, decay=0.75)</code>

## Limitations
Currently this implementation removes the ability to specify the "epoch" argument to the step function as the base PyTorch implementation has. It is a non-issue for most users, however if there is an interest in adding this capability, please submit an issue!

## Credits
This repository serves a similar purpose to the wonderful repo: <url>https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup</url>. However, it is distinct in that the warmup period is only applied to the first N global iterations rather than to the iterations of each restart. The implementation done here was completed and used before knowledge of this repository.

The max LR decay implementation is sourced from the StackOverflow answer here: <url>https://stackoverflow.com/questions/62427719/decrease-the-maximum-learning-rate-after-every-restart</url>, however it solely deals with the decay and no warmup. As well, it has at-base inheritance issues with attributes. Part of the reason for this repository is that combining different schedulers (e.g. LinearLR) with the solution here caused undesired issues and it was easier to combine both functionalities into the base CosineAnnealingWarmRestarts scheduler.
