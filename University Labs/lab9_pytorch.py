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
    
encoder=torch.nn.Sequential(
    makeBlock(2,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    torch.nn.Linear(64,4)
).cuda()

decoder=torch.nn.Sequential(
    makeBlock(2,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    torch.nn.Linear(64,2)
).cuda()

def sp(latent):
    mean=latent[:,:2]
    logVar=latent[:,2:]
    std=torch.exp(0.5*logVar)
    latentSample=mean+std*torch.randn_like(std)
    kl=mean**2+logVar.exp()-logVar
    return latentSample,kl.mean()

optimizerD=torch.optim.Adam(decoder.parameters(),lr=1e-3)
optimizerE=torch.optim.Adam(encoder.parameters(),lr=1e-3)

loss=0.0
#Training Loop
for i in (pbar:=tqdm(range(2000))):
    optimizerD.zero_grad()
    optimizerE.zero_grad()
    sample=swissRoll(1024,0.5).cuda()

    latent=encoder(sample)
    latentSample,klLoss=sp(latent)
    y=decoder(latentSample)

    reconstructionLoss=torch.nn.functional.mse_loss(y,sample)
    loss=reconstructionLoss + 1*klLoss.mean()
    loss.backward()
    optimizerE.step()
    optimizerD.step()
    if i%100==0:
        pbar.set_postfix({'Loss':loss.item()})

#%%
sR=swissRoll(500,0.5)
latent=encoder(sR.cuda())
latent,_=sp(latent)
latent=torch.randn_like(latent)
generatedData=decoder(latent).cpu().detach()

px.scatter(x=sR[:,0],y=sR[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
px.scatter(x=generatedData[:,0],y=generatedData[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
# %%
