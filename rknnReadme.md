try convert to rknn

siamrpn_alex_dwxcorr.py is used to export rknn and test tknn model on Ubuntu


cd pysot

python rknn/siamrpn_alex_dwxcorr.py --config experiments/siamrpn_alex_dwxcorr/config.yaml --snapshot experiments/siamrpn_alex_dwxcorr/model.pth --video demo/bag.avi

config.yaml 
    EXEMPLAR_SIZE: 127
    INSTANCE_SIZE: 287
input size: torch.Size([1, 3, 127, 127])
input size: torch.Size([1, 3, 287, 287])

rpn head
z_f shape: torch.Size([1, 256, 6, 6])
x_f shape: torch.Size([1, 256, 26, 26])

rpn head output
output[[cls],[loc]]
#cls np shape  : (1, 10, 21, 21) 
#loc np shape  : (1, 20, 21, 21) 
