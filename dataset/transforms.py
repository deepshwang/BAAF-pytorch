from torchvision import transforms
import torch

class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

T_S3DIS = transforms.Compose(
	[
		PointcloudToTensor()
	]
	)