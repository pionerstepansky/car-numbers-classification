from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
# transform = transforms.Compose(
#     [
#         # Hint: this might not be the best way to resize images
#         transforms.ToPILImage(),
#         transforms.GaussianBlur(5),
#         transforms.Resize((150, 300)),
#         transforms.ToTensor(),
#         # Hint: this might not be the best normalization
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]
# )

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(5, p=0.5),
        A.Cutout(num_holes=3, max_h_size=20, max_w_size=20),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)
