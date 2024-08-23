import torch
from PIL import Image
from torchvision import transforms
import json

model_script = None
bird_info = None


def predict_jit(image_path):
    global model_script, bird_info

    if model_script is None:
        model_script = torch.jit.load("model20200824.pt")

    # 加载图片
    image = Image.open(image_path)

    # 定义数据预处理操作
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # 对图片进行预处理
    image = transform(image)

    # 将图片转换为批量张量
    image = image.unsqueeze(0)

    # 进行推断
    output = model_script.forward(image)

    _, predicted = torch.max(output.data, 1)

    # 读取文件内容
    if bird_info is None:
        with open('birdinfo.json', 'r') as f:
            data = f.read()

        # 解码JSON格式的数据
        bird_info = json.loads(data)

    # 输出解码后的数据
    return bird_info[predicted.item()][0]


if __name__ == '__main__':
    print(predict_jit("images/白腰文鸟-A9_05972-clear.jpg"))
