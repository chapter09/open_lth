# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
#read model file
#get layer weight
#plot different layers
for i in range(13):
    folder_path = "replicate_1/level_{}/main".format(i)

    model_path = os.path.join(folder_path, "model_ep100_it0.pth")
    pic_path = os.path.join(folder_path, "pics")
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    weight_dict = torch.load(model_path, map_location=torch.device('cpu'))
    for name, weight in weight_dict.items():
        if 'weight' in name:
            #print(name)
            plt.spy(weight.numpy())
            plt.savefig(os.path.join(pic_path, "{}.png".format(name)))