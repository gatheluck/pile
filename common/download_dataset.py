import torch
import torchvision

import argparse

def main(opt):
	dataset = torchvision.datasets.CIFAR10(root=opt.download_path, download=True)
	dataset = torchvision.datasets.CIFAR100(root=opt.download_path, download=True)
	dataset = torchvision.datasets.ImageNet(root=opt.download_path, split="train", download=True)
	dataset = torchvision.datasets.ImageNet(root=opt.download_path, split="val", download=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--download_path", type=str, required=True, help="dataset download path.")
	opt = parser.parse_args()

	main(opt)

