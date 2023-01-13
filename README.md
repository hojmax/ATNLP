# ATNLP: Paper Implementation

## 🏄‍♂️ Usage

### Local

For local use, simply run the `main.ipynb` notebook.

### Colab

When using colab, upload the `main.ipynb` notebook. Afterwards, add a cell to the top of the notebook with the following content:

```python
!git clone https://git_token@github.com/hojmax/Individual-ATNLP.git
%cd /content/Individual-ATNLP
```

The `git_token` should be replaced by your personal access token, and is required since the repo is private. You can generate a token by going to:

Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)

When you push to the repository, you need to run `!git pull` and restart the runtime for the changes to take effect.

## 🏋️ Weights & Biases

You can access [our W&B team here](https://wandb.ai/project-group-1).

You will need an API key when connecting. This can be found in your settings.

## 📙 Resources

[Research Paper](https://arxiv.org/pdf/1711.00350.pdf)

[SCAN Data](https://github.com/brendenlake/SCAN)

[Pytorch Language Translation Tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html)

[Pytorch Seq2seq Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)


walk = WALK
look = LOOK
run = RUN
jump = JUMP
turn left = LTURN
turn right = RTURN
u left = LTURN u
u right = RTURN u
turn opposite left = LTURN LTURN turn opposite right = RTURN RTURN
u opposite left = turn opposite left u
u opposite right = turn opposite right u
turn around left = LTURN LTURN LTURN LTURN
turn around right = RTURN RTURN RTURN RTURN
u around left = LTURN u LTURN u LTURN u LTURN u u around right = RTURN u RTURN u RTURN u RTURN u x twice = x x
x thrice = x x x
x1 and x2 = x1 x2
x1 after x2 = x2 x1