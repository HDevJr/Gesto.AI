import numpy as np

data = np.load("data/interim/landmarks_raw/oi/oi_001.npy")

print("Shape:", data.shape)

first_frame = data[0]

left_hand = first_frame[:63]
right_hand = first_frame[63:126]
pose = first_frame[126:258]

print("Soma mão esquerda:", left_hand.sum())
print("Soma mão direita:", right_hand.sum())
print("Soma pose:", pose.sum())