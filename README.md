# OpenDungeon
Code and stuff for open-source AI Dungeon clone.

Check models and datasets here: https://huggingface.co/OpenDungeon

To train models use our version of CRD3 with ```train_opendungeon.py```. You will need around 12GB of vRAM or around 8GB of vRAM if you will use 8bit mode.

Side-by-Side of base GPT-J with finetuned on CRD3:

|                | Like         | Fluency      | No Contradiction | Reaction     |
|----------------|--------------|--------------|------------------|--------------|
| basemodel | 46.02\%      | 44.57\%      | 48.10\%          | 44.90\%      |
| LoRA      | 53.98\%| 55.43\% | 52.90\%     | 55.10\% |

Answers are weighted according to probabilities of being correctly labeled provided by Toloka.


### Inference:

Demo: [![Watch the video](https://github.com/AIshutin/opendungeon/blob/master/Screenshot%20from%202023-05-17%2015-20-54.png?raw=true)](https://github.com/AIshutin/opendungeon/blob/master/screen-capture.webm)

- [Colab/Your GPU](https://github.com/AIshutin/opendungeon/blob/master/notebooks/OAID_Inference.ipynb)
- [HuggingFace Spaces](https://huggingface.co/spaces/tafxle/Bloom_chat)
- [Petals + Colab/Your GPU](https://github.com/AIshutin/opendungeon/blob/master/notebooks/Petals_OAID_Inference.ipynb)


### Authors

In alphabetic order:
- [aishutin](https://github.com/AIshutin)
- [Graf-D](https://github.com/Graf-D)
- [justheuristic](https://github.com/justheuristic)
- [SvetlanaShir](https://github.com/SvetlanaShir)
- [tafxle](https://github.com/tafxle)

Please, ignore the authorship of commits.
