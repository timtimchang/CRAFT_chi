import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

#from pytictoc import TicToc
from torchvision import transforms
from models import network
from models import ohem

from dataset import SynthText
from dataset.dataloader import TBrain
from dataset import transutils

import craft_utils
from craft import CRAFT
from collections import OrderedDict
import shutil
from torch.autograd import Variable
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def copyStateDict(state_dict):
	if list(state_dict.keys())[0].startswith("module"):
		start_idx = 1
	else:
		start_idx = 0
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = ".".join(k.split(".")[start_idx:])
		new_state_dict[name] = v
	return new_state_dict
	
def cycle(iterable):
	while True:
		for x in iterable:
			yield x

def run():
	#print(TicToc.format_time(), "begining.......")
	print("begining.......")
	vgg = CRAFT()	 # initialize
	vgg = vgg.to("cuda")
	vgg.load_state_dict(copyStateDict(torch.load("/datasets/data-nfs-if-fin-brain/tim/CRAFT/model/craft_mlt_25k.pth")))
	vgg = torch.nn.DataParallel(vgg)
	print("finish load")
	
	vgg.eval()
	train_dataset = TBrain(vgg, '/datasets/data-nfs-if-fin-brain/tim/', target_size=640, viz=False)
	train_dataLoader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=12,
		shuffle=True,
		num_workers=0,
		drop_last=True,
		pin_memory=True)
	train_batch_syn = iter(cycle(train_dataLoader))

	# val_dataset = TBrain(vgg, '/datasets/data-nfs-if-fin-brain/tim/', target_size=640, viz=False, isTrain=False)
	# val_dataLoader = torch.utils.data.DataLoader(
	#	 val_dataset,
	#	 batch_size=2,
	#	 shuffle=False,
	#	 num_workers=0,
	#	 pin_memory=True)
	# val_batch_syn = iter(cycle(val_dataLoader))

	optimizer = torch.optim.Adam(vgg.parameters(), lr = 0.001)
	loss_fn = ohem.MSE_OHEM_Loss()
	loss_fn = loss_fn.to("cuda")
	#print(TicToc.format_time(), " training.........")
	print("training.........")

	for e in range(2000):
		# train
		total_loss = 0.0
		char_loss = 0.0
		aff_loss = 0.0

		for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(tqdm(train_dataLoader)):
			syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(train_batch_syn)
			images = torch.cat((syn_images,real_images), 0)
			gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
			gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
			mask = torch.cat((syn_mask, real_mask), 0)
			#affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)


			images = Variable(images.type(torch.FloatTensor)).cuda()
			gh_label = gh_label.type(torch.FloatTensor)
			gah_label = gah_label.type(torch.FloatTensor)
			gh_label = Variable(gh_label).cuda()
			gah_label = Variable(gah_label).cuda()
			mask = mask.type(torch.FloatTensor)
			mask = Variable(mask).cuda()
			# affinity_mask = affinity_mask.type(torch.FloatTensor)
			# affinity_mask = Variable(affinity_mask).cuda()

			out, _ = vgg(images)
			predict_r = out[:,:,:,0]
			predict_l = out[:,:,:,1]

			targets_r = gh_label
			targets_l = gah_label

			optimizer.zero_grad()
			loss_r = loss_fn(predict_r, targets_r)
			loss_l = loss_fn(predict_l, targets_l)

			loss = loss_r + loss_l #+ loss_cls
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			char_loss += loss_r.item()
			aff_loss += loss_l.item()
		
		val_total_loss = 0.0
		val_char_loss = 0.0
		val_aff_loss = 0.0

		# for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(val_dataLoader):
		#	 #real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
		#	 syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(val_batch_syn)
		#	 images = torch.cat((syn_images,real_images), 0)
		#	 gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
		#	 gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
		#	 mask = torch.cat((syn_mask, real_mask), 0)
		#	 #affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)


		#	 images = Variable(images.type(torch.FloatTensor)).cuda()
		#	 gh_label = gh_label.type(torch.FloatTensor)
		#	 gah_label = gah_label.type(torch.FloatTensor)
		#	 gh_label = Variable(gh_label).cuda()
		#	 gah_label = Variable(gah_label).cuda()
		#	 mask = mask.type(torch.FloatTensor)
		#	 mask = Variable(mask).cuda()
		#	 # affinity_mask = affinity_mask.type(torch.FloatTensor)
		#	 # affinity_mask = Variable(affinity_mask).cuda()

		#	 out, _ = vgg(images)
		#	 predict_r = out[:,:,:,0]
		#	 predict_l = out[:,:,:,1]

		#	 targets_r = gh_label
		#	 targets_l = gah_label

		#	 loss_r = loss_fn(predict_r, targets_r)
		#	 loss_l = loss_fn(predict_l, targets_l)

		#	 loss = loss_r + loss_l #+ loss_cls

		#	 val_total_loss += loss.item()
		#	 val_char_loss += loss_r.item()
		#	 val_aff_loss += loss_l.item()
		
		print("Train ", e, total_loss, char_loss, aff_loss)
		print("Val ", e, val_total_loss, val_char_loss, val_aff_loss)

		if e%5 == 0:
		  str_val_total_loss = str(total_loss)[:6]
		  torch.save(vgg.state_dict(), "./model/vgg_ICDAR_{0}e_{1}.pkl".format(e, str_val_total_loss))


if __name__ == "__main__":
	run()

