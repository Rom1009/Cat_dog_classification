import glob
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

import streamlit as st

from PIL import Image

import NerualNetwork
from customise_data.customized import Cat_Dog_Dataset


model = NerualNetwork.CNN()
model.load_state_dict(torch.load("model.pt"))


# Hàm để xử lý hình ảnh
def process_image(path):
    for file_path in glob.glob(
        "**/*", recursive=True
    ):  # Tìm trong tất cả các thư mục và tệp
        if path in file_path:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return [image]
    return None


# Tiêu đề của ứng dụng
st.title("Upload and Process Image")

# Tải ảnh lên từ người dùng
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Xử lý ảnh
    st.write(uploaded_file.name)
    processed_image = process_image(uploaded_file.name)
    if processed_image == None:
        st.write(
            "<h2 style='text-align: center; color: white;'>Please put image into folder code</h2>",
            unsafe_allow_html=True,
        )
    else:
        # Đọc ảnh đã tải lên
        image = Image.open(uploaded_file)

        # Hiển thị ảnh gốc
        st.image(image, caption="Uploaded Image", use_column_width=True)

        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        test_data = Cat_Dog_Dataset(images=processed_image, transform=transforms_test)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)
        # Hiển thị ảnh đã xử lý
        iter_test = iter(test_loader)
        img_test = next(iter_test)

        preds_test = model(img_test)
        img_test_permuted = img_test.permute(0, 2, 3, 1)
        rounded_preds = preds_test.round()

        types = rounded_preds[0].item()

        if types == 1:
            st.write(
                "<h2 style='text-align: center; color: white;'>This is a dog </h2>",
                unsafe_allow_html=True,
            )
        elif types == 0:
            st.write(
                "<h2 style='text-align: center; color: white;'>This is a cat</h2>",
                unsafe_allow_html=True,
            )
