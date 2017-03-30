import json
import numpy as np

with open('../dataset/obj_verification_val.json') as f:        
    val_list = json.loads(f.read())

n = 0
correct = 0

with open('obj_verification_result.txt') as f:
    for line in f:
        pred = int(line)
        gt = int(val_list[n][2])
        correct = correct + int(pred==gt) 
        n = n+1

print('accuracy')
print('%d/%d (%f)' %(correct, n, float(correct)/n))
