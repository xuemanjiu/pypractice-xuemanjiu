import torch
from src.banknote_classifier.models import VGG16


def test_vgg16_instantiation():
    model = VGG16()
    assert isinstance(model, torch.nn.Module)


def test_vgg16_output_shape():
    model = VGG16()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 999)


def test_vgg16_output_is_tensor():
    model = VGG16()
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, torch.Tensor)
