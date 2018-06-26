"""Discriminator model for ADDA."""

from torch import nn
import torch


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims,output_dims,len_feature_map,size_in):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        
        strides=[2**(k+1) for k in range(len_feature_map-1)]
        padding=[2**(k) for k in range(len_feature_map-1)]
        
        self.avg_pool=nn.ModuleList()
        for i in range(len(strides)):
            self.avg_pool+=[nn.AvgPool2d(strides[i]+1,stride=strides[i],padding=padding[i])]
        self.size_in=size_in
        self.input_filters=128
        self.layer_1=nn.Sequential(nn.Conv2d(input_dims, self.input_filters, kernel_size=1,stride=1),nn.ReLU())
        self.layer_2=nn.Sequential(nn.Linear(self.input_filters*size_in*size_in, 4096),
                                   nn.ReLU(),
                                   nn.Linear(4096, 2048),
                                   nn.ReLU(),
                                   nn.Linear(2048, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, output_dims))

#         self.layer=nn.ModuleList()
#         block=[]
#         hidden_dim_last=input_dims
#         for hidden_dim in hidden_dims:
#             block+=[
#             nn.Linear(hidden_dim_last, hidden_dim),
#             nn.ReLU()]
#             hidden_dim_last=hidden_dim
            
#         block+=[nn.Linear(hidden_dim_last, output_dims)]
        
#         self.layer=nn.Sequential(*block)
#         self.layer = nn.Sequential(
#             nn.Linear(input_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, output_dims)
# #             ,
# #             nn.LogSoftmax()
#         )

    def forward(self, feature_maps):
        """Forward the discriminator."""
        discri_input=feature_maps[0]

        for feature_map,avg_pool in zip(feature_maps[1:],self.avg_pool):
            feature_map=avg_pool(feature_map)
            discri_input=torch.cat((discri_input,feature_map),dim=1)


        out = self.layer_1(discri_input)

        out=out.view(-1, self.input_filters *self.size_in*self.size_in)

        out=self.layer_2(out)

        return out