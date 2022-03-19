# DARER
This repository contains the PyTorch source Code for our paper: **[DARER: Dual-task Temporal Relational Recurrent Reasoning Network
for Joint Dialog Sentiment Classification and Act Recognition](https://arxiv.org/abs/2203.03856)**.

**[Bowen Xing](https://scholar.google.com/citations?hl=zh-CN&user=DpBvmGQAAAAJ)** and **[Ivor W. Tsang](https://scholar.google.com/citations?user=rJMOlVsAAAAJ&hl=zh-CN)**.

***ACL 2022 (Findings)***.

## Architectures

DARER's Architecture:

<img src="img/model.pdf">


## Requirements
Our code relies on Python 3.6 and following libraries:
- transformers==1.1.0
- torch-geometric==1.7.0
- torch-geometric==1.5.0
- tqdm==4.60.0
- transformers==3.3.1
- numpy==1.19.2
- scikit-learn==0.24.2

## Run: // LSTM-based Encoder  
``` shell script
    # Mastodon //glove
    python -u main.py -lr 1e-3 -l2 1e-8 -dd dataset/mastodon -hd 128 -mc 2 -dr 0.2 -sn 3
  
    # DailyDialog // glove
    python -u main.py -ne 50 -hd 300 -lr 1e-3 -l2 1e-8 -dd dataset/dailydialogue -rnb 10 -sn 2 -mc 5 -dr 0.5
    # DailyDialog // train random word vector 
    python -u main.py -ne 50 -hd 256 -lr 1e-3 -l2 1e-8 -dd dataset/dailydialogue -sn 1 -mc 1e-05 -dr 0.3 -rw

```
## Run: // PTLM(pre-trained language model)-based Encoder 
``` shell script
    # Mastodon // BERT
    python -u main.py -pm bert -bs 16 -sn 4 -dr 0.3 -hd 768 -l2 0.01 -blr 1e-05 -mc 1
    # Mastodon // RoBERTa
    python -u main.py -pm roberta -bs 16 -sn 4 -dr 0.14 -hd 768 -l2 0.0 -blr 1e-05 -mc 1
    # Mastodon // XLNet
    python -u main.py -pm bert -bs 12 -sn 4 -dr 0.2 -hd 256 -l2 0.0 -blr 1e-05 -mc 1
```
We recommend you search the hypter-parameters on your service to obtain the best performances in your own experiment envronment.

## Citation
If you use our source code in this repo in your work, please cite the following paper. 
The bibtex are listed below:

<pre>
@article{xing2022darer,
  title={DARER: Dual-task Temporal Relational Recurrent Reasoning Network for Joint Dialog Sentiment Classification and Act Recognition},
  author={Xing, Bowen and Tsang, Ivor W},
  journal={arXiv preprint arXiv:2203.03856},
  year={2022}
}
</pre>
