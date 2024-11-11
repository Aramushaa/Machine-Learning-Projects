#%%
import torch
from tqdm.auto import tqdm
from plotly import express as px
def swissRoll(samples=1000, noise=0.0):
    t = 1.5 * 3.14 * (1 + 2 * torch.rand(samples, 1))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    X = torch.concat([x, y], axis=1)

    if noise > 0:
        X += noise * torch.randn(samples, 2)
    
    return X

# %%
def makeBlock(in_features,out_features):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features,out_features),
        torch.nn.BatchNorm1d(out_features),
        torch.nn.ReLU()
    )

def getCoefficients(t):
    alpha=torch.cos(3.14*t/2)
    beta=torch.sin(3.14*t/2)
    return alpha,beta

decoder=torch.nn.Sequential(
    makeBlock(2,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    torch.nn.Linear(64,2)
).cuda()

optimizerD=torch.optim.Adam(decoder.parameters(),lr=1e-3)
#%%
loss=0.0
#Training Loop
for i in (pbar:=tqdm(range(2000))):
    sample=swissRoll(1024,0.5).cuda()
    t=torch.rand(sample.shape[0],1).cuda()
    alpha,beta=getCoefficients(t)
    noise=torch.randn_like(sample)
    latent=alpha*sample+beta*noise

    optimizerD.zero_grad()
    prediction=decoder(latent)
    loss=torch.nn.functional.mse_loss(prediction,sample)
    loss.backward()
    optimizerD.step()
    
    if i%100==0:
        pbar.set_postfix({'Loss':loss.item()})

#%%
sR=swissRoll(500,0.5)
latent=torch.randn_like(latent)
t=torch.Tensor([1.0]).cuda()
for i in range(20):
    alpha,beta=getCoefficients(t)
    latent=decoder(latent)
    t-=0.05
    latent=alpha*latent+beta*torch.randn_like(latent)

generatedData=latent.cpu().detach().numpy()
px.scatter(x=sR[:,0],y=sR[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
px.scatter(x=generatedData[:,0],y=generatedData[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
# %%
