#%%
import torch
import torchsummary
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
def makeBlock(in_features,out_features,bn=True):
    if bn:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features,out_features),
            torch.nn.BatchNorm1d(out_features),
            torch.nn.ReLU()
        )
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features,out_features),
            torch.nn.ReLU()
        )

def hinge(x,maximize=True):
    if maximize:
        return torch.relu(1-x)
    else:
        return torch.relu(1+x)
    
encoder=torch.nn.Sequential(
    makeBlock(2,64,bn=False),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    torch.nn.Linear(64,1)
).cuda()

decoder=torch.nn.Sequential(
    makeBlock(2,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    makeBlock(64,64),
    torch.nn.Linear(64,2)
).cuda()

optimizerD=torch.optim.Adam(decoder.parameters(),lr=5e-4)
optimizerE=torch.optim.Adam(encoder.parameters(),lr=1e-3)

#%%
loss=0.0
#Training Loop
for i in (pbar:=tqdm(range(8000))):
    sample=swissRoll(1024,0.5).cuda()
    latent=torch.randn_like(sample)

    optimizerD.zero_grad()
    sampleFake=decoder(latent)
    generatorLoss=hinge(encoder(sampleFake),maximize=True).mean()
    generatorLoss.backward()
    optimizerD.step()

    optimizerE.zero_grad()
    discriminatorLoss= ( hinge(encoder(sampleFake.detach()),maximize=False) + hinge(encoder(sample),maximize=True) ).mean()
    discriminatorLoss.backward()
    optimizerE.step()
    
    if i%100==0:
        pbar.set_postfix({'Generator Loss':generatorLoss.item()})

#%%
sR=swissRoll(500,0.5)
latent=torch.randn_like(latent)
generatedData=decoder(latent).cpu().detach()

px.scatter(x=sR[:,0],y=sR[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
px.scatter(x=generatedData[:,0],y=generatedData[:,1],width=512,height=512,template='plotly_dark',range_x=[-15,15],range_y=[-15,15]).show()
# %%
