from FoTF_module import *
from utils import *
from conf.dataset.HierarchicalModel_bk_SERVIR import *

class HC_model(nn.Module):
    def __init__(self, shape_in, hid_S=256, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8): #moving
    #def __init__(self, shape_in, hid_S=512, hid_T=512, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8): #taxiing
    #def __init__(self, shape_in, hid_S=8, hid_T=8, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8):

        super(HC_model, self).__init__()
        T, C, H, W = shape_in

        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))

        self.fotf_encoder = FoTF(shape_in=shape_in)
        self.skip_conneciton = ConvolutionalNetwork.skip_connection(shape_in=shape_in)
        self.latent_projection = Encoder(C, hid_S, N_S)
        self.enc = Encoder(C, hid_S, N_S)
        #self.TeDev_block = TeDev(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups) #
        self.dec = Decoder(hid_S, C, N_S)
        #self.hierarchicalModel=HierarchicalModel(shape_in,input_dim=2304,output_dim=2304,model_dim=8,embed='timeF',freq=12,dropout=0.1)#2024-10-17-moving
        #self.hierarchicalModel=HierarchicalModel(shape_in,input_dim=256,output_dim=256,model_dim=512,embed='timeF',freq=10,dropout=0.1) 2024-10-17
        self.hierarchicalModel=HierarchicalModel(shape_in,input_dim=256,output_dim=256,model_dim=256,embed='timeF',freq=10,dropout=0.1) #2024-10-17-moving

    def forward(self, input_st_tensors):
        # Residual Connection Component
        B, T, C, H, W = input_st_tensors.shape #W 64 B5 C 1 H 64 T 10
        skip_feature = self.skip_conneciton(input_st_tensors) #(5,10,1,64,64)
        spatial_feature = self.fotf_encoder(input_st_tensors)  #(5,10,1,64,64)
        #Global $\&$ Local Spatio Component

        spatial_feature = spatial_feature.reshape(-1, C, H, W)#(5*10,1,64,64) 将batch,len->batch*Len
        spatial_embed, spatial_skip_feature = self.latent_projection(spatial_feature) #spatial_skip_feature(50,512,64,64) spatial_embed (50,512,64,64)
        _, C_, H_, W_ = spatial_embed.shape # BxT, D h w
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_) # B:5, T 10 , D 512  ,h 16, w 16

        # {Global $\&$ Local Temporal Component


        # Mamba TeDev
        spatial_embed_T=spatial_embed.view(B,C_,T,H_*W_)
        Hierarchical_embed=self.hierarchicalModel(spatial_embed_T)
        Hierarchical_embed = Hierarchical_embed.reshape(B*T, C_, H_, W_) # B:5, T 10 , c 512  ,h 16, w 16

        predictions = self.dec(Hierarchical_embed, spatial_skip_feature)
        # Temporal-spatial Hybrid Decoder

        # Decoder
        #predictions = self.dec(spatialtemporal_embed, spatial_skip_feature)
        predictions = self.dec(Hierarchical_embed, spatial_skip_feature).reshape(B, T, C, H, W)

        predictions = 0.1 * predictions + skip_feature
        
        return predictions

if __name__ == '__main__':
    x = torch.randn((1, 10, 1, 64, 64))
    y = torch.randn((1, 10, 1, 64, 64))
    model1 = HC_model(shape_in=(10, 1, 64, 64))
    output = model1(x)
    print("input shape:", x.shape)
    print("output shape:", output.shape)

    def model_memory_usage_in_bytes(model):
        total_bytes = 0
        for param in model.parameters():
            num_elements = np.prod(param.data.shape)
            total_bytes += num_elements * 4  
        return total_bytes
    
    total_bytes = model_memory_usage_in_bytes(model1) 
    mb = total_bytes / 1048576
    print(f'Total memory used by the model parameters: {mb} MB')
