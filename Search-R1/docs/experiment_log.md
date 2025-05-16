
## Experiment log

### Preliminary results 

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-open)


The preliminary experiment is conducted only on natural question (NQ) dataset (+ PPO) with a small number of training steps.


### v0.1

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train), [docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa), [scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.1)


We extend the experiments from NQ to seven datasets with both PPO and GRPO methods. The studies are still on a small number of training steps with a big learning rate warm up ratio.


### v0.2

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-v0.2), [docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa), [scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.2)


We fix several bugs including [retrieved token masking](https://github.com/PeterGriffinJin/Search-R1/pull/21) and [GRPO sample indexing](https://github.com/PeterGriffinJin/Search-R1/commit/9ec2fa9892fbf0315d0c67b4dc08ae8f6cf5f378). 
The former can largely improve the stablity of RL training. 
Then we adjust the training scripts, increasing the number of training steps and decreasing the learning rate warm up ratio, to obtain a better performance, and conduct experiments on different scale of LLMs (3B, 7B, 14B).


### v0.3

Ongoing, stay tuned!

