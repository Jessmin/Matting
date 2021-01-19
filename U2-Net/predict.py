import os
import torch
from torchvision import transforms
from PIL import Image
from model.u2net import U2NET


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def main():
    model_dir = '/home/zhaohoj/Documents/checkpoint/U2net/u2net.pth'
    filename = '/home/zhaohoj/Pictures/getImage.png'
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        net.to('cuda')
    net.eval()
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_batch)
    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    predict_np = pred.squeeze().cpu().data.numpy()
    img = Image.fromarray(predict_np * 255).convert('RGB')
    img.show()


if __name__ == '__main__':
    main()
