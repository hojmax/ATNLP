# Advanced Topics in NLP Final Project

My report can be found [here](https://github.com/hojmax/ATNLP/blob/main/Paper.pdf).

## 📝 Description

This is my final project for the UCPH course [Advanced Topics in Natural Language Processing](https://kurser.ku.dk/course/NDAK19001U/2022-2023) (ATNLP). The project goal was two fold. The first part was a reproduction study of the paper *Generalization without Systematicity* by Lake and Baroni. The second part was free-form, where i chose to investigate in-context learning for experiment 3.

## 🏄‍♂️ Usage

You will need a [Weights & Biases](https://www.wandb.com/) account to log the experiments. You will be prompted to enter your API key when running the notebook. You should change the `wandb_project` and `wandb_entity` variables at the top of the notebook to match your account.

### Local

The reproduction part can be executed by running the `main.ipynb` notebook. In order to execute the in-context learning  script you can run:
```bash
python in-context.py --model facebook/opt-350m --dataset random8.json --max-examples 1000
```
The model and dataset can be varied. See [the report](https://github.com/hojmax/ATNLP/blob/main/Paper.pdf) for more information on this. All the datasets are available in the `/prompts` folder. The script expects cuda to be available.

### Colab

When using colab, upload the `main.ipynb` notebook. Afterwards, add a cell to the top of the notebook with the following content:

```python
!git clone https://github.com/hojmax/ATNLP.git
%cd /content/ATNLP
```

You can then run the notebook and the in-context script as described above.

## 📙 Resources

[Research Paper](https://arxiv.org/pdf/1711.00350.pdf)

[SCAN Data](https://github.com/brendenlake/SCAN)

[Pytorch Seq2seq Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
