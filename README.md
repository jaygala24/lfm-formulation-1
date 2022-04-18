# Learning from Mistakes (LFM) - A Framework for Neural Architecture Search

The unofficial pytorch implementation for [Learning from Mistakes - A Framework for Neural Architecture Search](https://arxiv.org/abs/2111.06353).

### Architecture Search

- **DARTS:**
    ```
    CIFAR-10/100: cd darts-LFM && python train_search.py --gpu 0 \\
    --is_cifar100 0/1 --img_encoder_arch 18 --batch_size 96 --save <exp_name>
    ```
    Append `--unrolled` argument for training using 2nd order approximation

- **PDARTS:**
    ```
    CIFAR-10: cd pdarts-LFM && python train_search.py --add_layers 6 \\
    --add_layers 12 --dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
    --batch_size 96 --save <exp_name>
    ```
    ```
    CIFAR-100: cd pdarts-LFM && python train_search.py --add_layers 6 \\
    --add_layers 12 --dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
    --cifar100 --batch_size 96 --save <exp_name>
    ```

- **PC-DARTS:**
    
    Data preparation: Please first sample 10% and 2.5% images for each class as the training and validation set, which is done by `pcdarts-LFM/sample_images.py`.

    ```
    CIFAR-10: cd pcdarts-LFM && python train_search.py --gpu 0 \\
    --set cifar10 --img_encoder_arch 18 --batch_size 96 --save <exp_name>
    ```
    ```
    CIFAR-100: cd pcdarts-LFM && python train_search.py --gpu 0 \\
    --set cifar100 --img_encoder_arch 18 --batch_size 96 --save <exp_name>
    ```
    ```
    ImageNet: cd pcdarts-LFM && python train_search_imagenet.py \\
    --img_encoder_arch 18 --batch_size 96 --save <exp_name>
    ```


### Architecture Evaluation

- **DARTS:**
    ```
    CIFAR-10/100: cd darts-LFM && python train.py --cutout --auxiliary \\
    --gpu 0 --is_cifar100 0/1 --arch <arch_name> --batch_size 96 --save <exp_name>
    ```
    ```
    ImageNet: cd darts-LFM && python train_imagenet.py --auxiliary \\
    --arch <arch_name> --batch_size 96 --save <exp_name>
    ```

- **PDARTS:**
    ```
    CIFAR-10: cd pdarts-LFM && python train.py --cutout --auxiliary \\
    --arch <arch_name> --batch_size 96 --save <exp_name>
    ```
    ```
    CIFAR-100: cd pdarts-LFM && python train.py --cutout --auxiliary \\
    --cifar100 --arch <arch_name> --batch_size 96 --save <exp_name>
    ```
    ```
    ImageNet: cd pdarts-LFM && python train_imagenet.py --auxiliary \\
    --arch <arch_name> --batch_size 96 --save <exp_name>
    ```

- **PC-DARTS:**
    ```
    CIFAR-10: cd pcdarts-LFM && python train.py --cutout --auxiliary \\
    --gpu 0 --set cifar10 --arch <arch_name> --batch_size 96 --save <exp_name>
    ```
    ```
    CIFAR-100: cd pcdarts-LFM && python train.py --cutout --auxiliary \\
    --gpu 0 --set cifar100 --arch <arch_name> --batch_size 96 --save <exp_name>
    ```
    ```
    ImageNet: cd pcdarts-LFM && python train_imagenet.py --auxiliary \\
    --arch <arch_name> --batch_size 96 --save <exp_name>
    ```

### Related Work

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://github.com/yuhuixu1993/PC-DARTS)

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://github.com/chenxin061/pdarts)

[Differentiable Architecture Search](https://github.com/quark0/darts)

### Citations

```
@article{Garg2021LearningFM,
  title={Learning from Mistakes - A Framework for Neural Architecture Search},
  author={Bhanu Garg and Li Lyna Zhang and Pradyumna Sridhara and Ramtin Hosseini and Eric P. Xing and Pengtao Xie},
  journal={ArXiv},
  year={2021},
  volume={abs/2111.06353}
}
```
