# Class-Incremental Learning with Multi-Teacher Knowledge Distillation

<center>
<img src="./assets/4.png" alt="abc" height="380" style="vertical-align:middle">
</center>

**Soft computing lab**
---

<strong> Implementation for Class-Incremental Learning with Multi-Teacher Knowledge Distillation </strong>
<center>
<img src="./assets/5.png" alt="abc" height="380" style="vertical-align:middle">
</center>
Abstract

- [Installation](#installation)
- [Run](#run-experiment)
- [Results](#results)



### Installation
**Dependencies**
* Python 3.8
* torch 1.8.1
* torchvision 0.6.0
* tqdm
* numpy
* scipy
* quadprog
* POT


### Run
```
python main.py --config ./exps/ours.json
```
******

### Results
Results will be saved in `logs/Ours/$dataset$/$InitTasks-Increment$/$Increment$/$prefix$_$seed$_$backbone$_*` for each episode.
**MNIST-5tasks**
<center>
<img src="./assets/3.png" alt="abc" height="380" style="vertical-align:middle">
</center>
<center>Average accuracy</center>

|![](./assets/1.png)|![](./assets/2.png)|
|:---:|:---:|
|Test accuracy|Class accuracy|