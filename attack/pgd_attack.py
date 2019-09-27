import random

import torch
from torch import nn

__all__ = [
	'PGDAttack'
]

class PGDAttack(AttackWrapper):
	def __init__(self, nb_its, eps_max, step_size, resol, norm='linf', rand_init=True, scale_each=False):
		"""
		Parameters:
			nb_its (int):				Number of iterations.
			eps_max (float):		The max norm, in pixel space.
			step_size (float):	The max step size, in pixel space.
			resol (int):				Side length of the image.
			norm (str):					Either 'linf' or 'l2'
			rand_init (bool):		Whether to init randomly in the norm ball.
			scale_each (bool):	Whether to scale eps for each image in a batch separately.
		"""
		super().__init__(resol)
		self.nb_its = nb_its
		self.eps_max = eps_max
		self.step_size = step_size
		self.resol = resol
		self.norm = norm
		self.rand_init = rand_init
		self.scale_each = scale_each

		self.criterion = nn.CrossEntropyLoss().cuda()

	def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
		# compute base_eps and step_size
		if scale_eps:
			if self.scale_each:
				rand = torch.rand(pixel_image.size(0), device='cuda')
			else:
				rand = random.random() * torch.ones(pixel_image.size(0), device='cuda')
			base_eps  = rand.mul(self.eps_max)
			step_size = rand.mul(self.step_size)
		else:
			base_eps  = self.eps_max   * torch.ones(pixel_image.size(0), device='cuda')
			step_size = self.step_size * torch.ones(pixel_image.size(0), device='cuda')

		pixel_inp = pixel_img.detach()
		pixel_inp.requires_grad = True
		
		# adversarial loop 
		delta = self._init(pixel_inp.size(), base_eps)
		if self.nb_its > 0:
			delta = self._run_one(pixel_model, pixel_inp, delta, target, base_eps, step_size, avoid_target=avoid_target)
		else:
			# clamp between 0 ~ 255
			delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
		pixel_result = pixel_inp + delta
		return pixel_result