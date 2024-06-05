# Class-Incremental Learning with Knowledge Distillation
**Soft computing lab**
Implementation for Continual learning via Knowledge Distillation


****
- [Run](#run-experiment)
- [Installation](#installation)
- [Results](#results)





### Installation
---
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
---
```
python main.py --config ./exps/ours.json
```
******

### Results
---
**MNIST-5tasks**
<center>
<img src="./assets/3.png" alt="abc" height="380" style="vertical-align:middle"> <br> Average accuracy
</center>

|![](./assets/1.png)|![](./assets/2.png)|
|:---:|:---:|
|Test accuracy|Class accuracy|