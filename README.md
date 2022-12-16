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

[Pytorch Language Translation Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

[Pytorch Seq2seq Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
