import scipy.io

data = scipy.io.loadmat('Market-1501_Attribute/market_attribute.mat')['market_attribute']['train'][0,0]

# Extract attributes
ids = [str(x[0]) for x in data['image_index'][0][0].flatten()]

prompts = []
for i in range(len(ids)):
    pid = ids[i]
    
    # 1=male, 2=female
    gender = "man" if data['gender'][0][0].flatten()[i] == 1 else "woman"
    
    # 1=young, 2=teenager, 3=adult, 4=old
    age_val = data['age'][0][0].flatten()[i]
    age = "young" if age_val == 1 else "teenage" if age_val == 2 else "adult" if age_val == 3 else "old"
    
    # hair: 1=short, 2=long
    hair = "short hair" if data['hair'][0][0].flatten()[i] == 1 else "long hair"
    
    # hat: 1=no, 2=yes
    hat_str = " and a hat" if data['hat'][0][0].flatten()[i] == 2 else ""
    
    # clothes: 1=dress, 2=pants
    clothes = "dress" if data['clothes'][0][0].flatten()[i] == 1 else "pants"
    
    # backpacks, bags
    bag_str = []
    if data['backpack'][0][0].flatten()[i] == 2:
        bag_str.append("backpack")
    if data['bag'][0][0].flatten()[i] == 2:
        bag_str.append("bag")
    if data['handbag'][0][0].flatten()[i] == 2:
        bag_str.append("handbag")
    
    bag_desc = ", carrying a " + " and ".join(bag_str) if bag_str else ""
    
    # Colors
    up_colors = []
    for c in ['upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow']:
        if data[c][0][0].flatten()[i] == 2:
            up_colors.append(c[2:]) # remove 'up'
    up_color = " and ".join(up_colors) if up_colors else "unknown color"
    
    down_colors = []
    for c in ['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow']:
        if data[c][0][0].flatten()[i] == 2:
            down_colors.append(c[4:])
    down_color = " and ".join(down_colors) if down_colors else "unknown color"
    
    prompt = f"X X X X A photo of a {age} {gender} with {hair}{hat_str}, wearing a {up_color} top and {down_color} {clothes}{bag_desc}."
    prompts.append((pid, prompt))

# Sort by pid just in case, because standard ReID dataset loaders sort train identities by ID
prompts.sort(key=lambda x: int(x[0]))

with open("prompts_market1501_attributes.txt", "w") as f:
    for pid, p in prompts:
        f.write(f"{pid}:{p}\n")

print(f"Generated {len(prompts)} prompts in prompts_market1501_attributes.txt")
