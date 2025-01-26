# import tensorflow as tf
# from tensorflow.keras import layers, models

# # Define U-Net Model
# def unet_model(input_shape):
#     inputs = layers.Input(input_shape)

#     # Contracting path
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)

#     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)

#     c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
#     p3 = layers.MaxPooling2D((2, 2))(c3)

#     c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
#     c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
#     p4 = layers.MaxPooling2D((2, 2))(c4)

#     # Bottleneck
#     c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
#     c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

#     # Expansive path
#     u6 = layers.UpSampling2D((2, 2))(c5)
#     u6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)

#     # Ensure that u6 and c4 have compatible spatial dimensions
#     u6_resized = layers.Resizing(height=c4.shape[1], width=c4.shape[2])(u6)  # Resize u6 to match c4
#     u6 = layers.Concatenate(axis=-1)([u6_resized, c4])

#     c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
#     c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

#     u7 = layers.UpSampling2D((2, 2))(c6)
#     u7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)

#     u7_resized = layers.Resizing(height=c3.shape[1], width=c3.shape[2])(u7)  # Resize u7 to match c3
#     u7 = layers.Concatenate(axis=-1)([u7_resized, c3])

#     c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
#     c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

#     u8 = layers.UpSampling2D((2, 2))(c7)
#     u8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)

#     u8_resized = layers.Resizing(height=c2.shape[1], width=c2.shape[2])(u8)  # Resize u8 to match c2
#     u8 = layers.Concatenate(axis=-1)([u8_resized, c2])

#     c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
#     c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

#     u9 = layers.UpSampling2D((2, 2))(c8)
#     u9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)

#     u9_resized = layers.Resizing(height=c1.shape[1], width=c1.shape[2])(u9)  # Resize u9 to match c1
#     u9 = layers.Concatenate(axis=-1)([u9_resized, c1])

#     c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
#     c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

#     outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(c9)

#     model = models.Model(inputs=[inputs], outputs=[outputs])
#     return model

# # Define input shape based on the generated data
# input_shape = (20, 8, 1)  # 20 rows (4 users + 16 IRS elements), 8 antennas, 1 channel

# # Create the U-Net model
# model = unet_model(input_shape)
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()


import tensorflow as tf
from tensorflow.keras import layers, models

# Define U-Net Model
def unet_model(input_shape):
    inputs = layers.Input(input_shape)

    # Contracting path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2), padding='same')(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2), padding='same')(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2), padding='same')(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2), padding='same')(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Expansive path
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)

    # Resize u6 to match c4's dimensions
    u6_resized = layers.Resizing(height=c4.shape[1], width=c4.shape[2])(u6)
    u6 = layers.Concatenate(axis=-1)([u6_resized, c4])

    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)

    # Resize u7 to match c3's dimensions
    u7_resized = layers.Resizing(height=c3.shape[1], width=c3.shape[2])(u7)
    u7 = layers.Concatenate(axis=-1)([u7_resized, c3])

    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)

    # Resize u8 to match c2's dimensions
    u8_resized = layers.Resizing(height=c2.shape[1], width=c2.shape[2])(u8)
    u8 = layers.Concatenate(axis=-1)([u8_resized, c2])

    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)

    # Resize u9 to match c1's dimensions
    u9_resized = layers.Resizing(height=c1.shape[1], width=c1.shape[2])(u9)
    u9 = layers.Concatenate(axis=-1)([u9_resized, c1])

    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Define input shape based on the generated data
input_shape = (20, 8, 1)  # 20 rows (4 users + 16 IRS elements), 8 antennas, 1 channel

# Create the U-Net model
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()