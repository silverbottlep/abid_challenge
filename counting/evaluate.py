import json
import numpy as np

with open('../dataset/counting_val.json') as f:        
    val_list = json.loads(f.read())

n = 0
perclass_correct = np.zeros(6) 
perclass_dist = np.zeros(6) 
perclass_N = np.zeros(6)

with open('counting_result.txt') as f:
    for line in f:
        pred = int(line)
        gt = int(val_list[n][1])
        perclass_correct[gt] = perclass_correct[gt] + int(pred==gt) 
        perclass_dist[gt] = perclass_dist[gt] + np.power(pred-gt,2)
        perclass_N[gt] = perclass_N[gt] + 1
        n = n+1

print('accuracy')
print('%d/%d (%f)' %(perclass_correct.sum(), perclass_N.sum(), perclass_correct.sum()/perclass_N.sum()))
print('RMSE(Root mean squared error)')
print(np.sqrt(perclass_dist.sum()/perclass_N.sum()))
print('Per class accuracy')
print(perclass_correct/perclass_N)
print('Per class RMSE')
print(np.sqrt(perclass_dist/perclass_N))
