# -*- coding: utf-8 -*-
import json
import csv
import os
import matplotlib.pyplot as plt
import matplotlib

remain_rate = {}
accuracy_dict={}
total_accuracy=[]

line_style = ['-','--',':']
for i in range(13):
    sparsity_path=f"replicate_1/level_{i}/main/sparsity_report.json"
    src=f"replicate_1/level_{i}/main/logger"
    log_path=f"replicate_1/level_{i}/main/logger.csv"
    data_save_path=f"data.json"

    #os.rename(src,log_path)
  
    with open(sparsity_path) as f:

        sparsity_data = json.load(f)

        rate = round(sparsity_data['unpruned']/sparsity_data['total'],2)
        remain_rate['level_{}'.format(i)]= rate
    

    with open(log_path, 'r+') as cf:
        csv_reader = csv.reader(cf, delimiter=' ')
        accuracy_list = []
        step = 1
        for row in csv_reader:
            if(step%3==2):
                data = row[0].split(",")
                accuracy_list.append(data[2])
                total_accuracy.append(data[2])
            step = step+1


        accuracy = [round(float(item)*100,2) for item in accuracy_list]
        size = len(accuracy)
        x = list(range(0, size))
        #if(rate == 1.0 or rate ==0.07 or rate ==0.09):
        plt.plot(x, accuracy, line_style[i%3],label=rate)

        accuracy_dict['level_{}'.format(i)] = accuracy_list

#with open(data_save_path, 'w')as pf:
    #json.dump(accuracy_dict,pf)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(95,100)
#plt.xlim(0,100)
plt.legend(loc='lower right',fontsize='x-small',prop={'size': 6})

plt.savefig('pic_4.png')



                
            

        
        
