# FSPT-FSCIL
Repository for "Brain-Inspired Fast- and Slow-Update Prompt Tuning for Few-Shot Class-Incremental Learning"

## How to Install
 
This project is built upon a **modified** version of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). so you need to install the **modified** version of `dassl` first. 

**Please follow these steps for installation:**

```bash
# Clone this repo
git clone https://github.com/qihangran/FSPT-FSCIL.git

# install modified Dassl
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).

## Datasets

We follow the [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training. Put all datasets under the same folder (say `$DATA`) and follow the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– cifar100/
|–– cub200/
|–– miniimagenet/
```
For CIFAR100, we do **not** directly use the original CIFAR100. Instead, we have organized CIFAR-100 according to the FSCIL paradigm with staged splits. The staged CIFAR-100 images can be downloaded from [here](https://drive.google.com/file/d/1OAjxYXmqhycEvHxDuCRvMEaJjE5dTiQu/view?usp=share_link).


For miniImageNet and cub200, you can download the dataset [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing).  

For FGVCAircraft and DTD, you can download the dataset [FGVCAircraft](#fgvcaircraft), [DTD](#dtd)




## How to Run
### CIFAR

For CIFAR-100 base class training and test, run
```
CUDA_VISIBLE_DEVICES = 1 bash run_cifar100_base.sh
```


For CIFAR-100 incemental class training and test, run
```
CUDA_VISIBLE_DEVICES = 1 bash run_cifar100_inc.sh
```
Result:

|Session|0|1|2|3|4|5|6|7|8|
|-|-|-|-|-|-|-|-|-|-|
|FSPT-FSCIL|88.0|85.1|83.9|81.9|81.6|81.3|80.6|80.4|79.7|

### cub200
Refer to CIFAR
### miniimagenet
Refer to CIFAR
### DTD
Refer to CIFAR
### FGVCAircraft
Refer to CIFAR
