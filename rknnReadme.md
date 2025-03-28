
# RKNN Model
## Convert to rknn and test on ubuntu(x86-64 amd64)

### Installation

you need to download [rknn-tookit2](https://github.com/airockchip/rknn-toolkit2),and install rknn-tookit2 on your ubuntu(x86-64,amd64)

## export rknn and test
You need to download model from [PySOT Model Zoo](MODEL_ZOO.md).
```
cd to rootPath(pysot_rknn)
python rknn/siamrpn_alex_dwxcorr.py --config experiments/siamrpn_alex_dwxcorr/config.yaml --snapshot experiments/siamrpn_alex_dwxcorr/model.pth --video demo/bag.avi
```
The file ***siamrpn_alex_dwxcorr.py*** can export rknn model, and test the result on your ubuntu(x86-64,amd64).  

Detail about **siamrpn_alex_dwxcorr** model.  
**backbone of exemplar**.  
>Target img:  
>>input size: torch.Size([1, 3, 127, 127])  
>>outputsize: torch.Size([1, 256, 6, 6])  

**backbone of instance**.  
>Original img:  
>>input size: torch.Size([1, 3, 287, 287])  
>>outputsize: torch.Size([1, 256, 26, 26])   

**rpn Head**
>rpn head input size: [z_f,x_f]  
>>z_f shape: torch.Size([1, 256, 6, 6])  
>>x_f shape: torch.Size([1, 256, 26, 26])  
>output size: output[[cls],[loc]]  
>>cls shape: (1, 10, 21, 21)  
>>loc shape: (1, 20, 21, 21)  

z_f is the output of **backbone of exemplar**, x_f is the output of **backbone of instance**
  
## RUN the demo on RK3588
Install rknn-toolkit-lite2 on your rk device, I run the demo on RK3588.  
Copy folders and files on the projict to your rk device, and your need to maintain the structure of the project.
>pysot_rknn(root path)
>>demo/  
>>tools/  
>>rknn/  
>>experiments/  
>>pysot/  

```
python tools/runRKNNLite.py
```
The mode inference speed on rk3588 is about 10 FPS.

