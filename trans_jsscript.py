from show_emo import load_model
import torch
from lib.msra_resnet import get_pose_net

def main():
    to_coords_model_pt("./pose_coords_model2.pt")


def to_model_pt(path):
    vgcnn = load_model()
    vgcnn.eval()  ## put models in eval mode
    x = torch.rand(1, 3, 224, 224)
    m = torch.jit.trace(vgcnn, x)
    torch.jit.save(m, path)

    # torch.save(vgcnn, path)



def to_coords_model_pt(path):

    num_layers = 50
    # heads = {'hm': 16, 'depth': 16}
    model = get_pose_net(num_layers)

    load_model = './models/fusion_3d_var.pth'
    # checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(load_model, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()

    for key in state_dict:
        if not 'deconv_layers' in key:
            model_state_dict[key] = state_dict[key]
        elif '1' in key:
            model_state_dict[key.replace('deconv_layers.1.', 'deconv_layers.0.bn.')] = state_dict[key]
        elif '4' in key:
            model_state_dict[key.replace('deconv_layers.4.', 'deconv_layers.1.bn.')] = state_dict[key]
        elif '7' in key:
            model_state_dict[key.replace('deconv_layers.7.', 'deconv_layers.2.bn.')] = state_dict[key]

    model_state_dict['deconv_layers.0.deconv.weight'] = state_dict['deconv_layers.0.weight']
    model_state_dict['deconv_layers.1.deconv.weight'] = state_dict['deconv_layers.3.weight']
    model_state_dict['deconv_layers.2.deconv.weight'] = state_dict['deconv_layers.6.weight']


    model.load_state_dict(model_state_dict)

    model.eval()  ## put models in eval mode
    x = torch.rand(1, 3, 256, 256)
    m = torch.jit.trace(model, x)
    torch.jit.save(m, path)


if __name__ == '__main__':
    # main()
    model = torch.jit.load("./pose_coords_model.pt")
    output = model(torch.ones(1, 3, 256, 256))
    print(output)








