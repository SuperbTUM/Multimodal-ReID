import scipy.io
import numpy as np

data = scipy.io.loadmat('Market-1501_Attribute/market_attribute.mat')['market_attribute']['train'][0,0]
ids = [int(x[0]) for x in data['image_index'][0][0].flatten()]

attrs_list = []
for i in range(len(ids)):
    attr_vec = []
    attr_vec.append(1.0 if data['gender'][0][0].flatten()[i] == 2 else 0.0)
    
    age = data['age'][0][0].flatten()[i]
    attr_vec.extend([1.0 if age == 1 else 0.0,
                     1.0 if age == 2 else 0.0,
                     1.0 if age == 3 else 0.0,
                     1.0 if age == 4 else 0.0])
    
    attr_vec.append(1.0 if data['hair'][0][0].flatten()[i] == 2 else 0.0)
    attr_vec.append(1.0 if data['hat'][0][0].flatten()[i] == 2 else 0.0)
    attr_vec.append(1.0 if data['clothes'][0][0].flatten()[i] == 2 else 0.0)
    
    attr_vec.append(1.0 if data['backpack'][0][0].flatten()[i] == 2 else 0.0)
    attr_vec.append(1.0 if data['bag'][0][0].flatten()[i] == 2 else 0.0)
    attr_vec.append(1.0 if data['handbag'][0][0].flatten()[i] == 2 else 0.0)
    
    for c in ['upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow']:
        attr_vec.append(1.0 if data[c][0][0].flatten()[i] == 2 else 0.0)
        
    for c in ['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow']:
        attr_vec.append(1.0 if data[c][0][0].flatten()[i] == 2 else 0.0)
        
    attrs_list.append((ids[i], attr_vec))

attrs_list.sort(key=lambda x: x[0])
attr_tensor = np.array([x[1] for x in attrs_list], dtype=np.float32)
np.save("market_train_attrs.npy", attr_tensor)
print(f"Saved attribute tensor of shape {attr_tensor.shape} to market_train_attrs.npy")
