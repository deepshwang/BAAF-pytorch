from torch.utils.data import DataLoader
import pdb
from tqdm import tqdm
import torch
import os
import numpy as np
import wandb

from utils.argparse import argument_parser
from utils.tools import parse_yaml
from dataset.S3DIS import S3DISDataset, S3DISDatasetWholeScene
from dataset.transforms import T_S3DIS
from model.baaf import BilateralAugmentation, BAAFNet


class Trainer():
	def __init__(self, args, config, train_dataset, test_dataset):
		self.args = args
		self.config = config
		self.label2names = config['dataset']['s3dis']['label2names']
		
		# Training configuration
		self.device = config['device']
		self.aug_loss_weights = list(map(float, config['train']['aug_loss_weight'].split(",")))
		self.epochs = config['train']['epochs']
		self.num_classes = config['num_classes']
		
		# Attributes Initialization
		self.model = self._initialize_model()
		self.optimizer = self._initialize_optimizer()
		self.lr_scheduler = self._initialize_lr_scheduler()
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.train_dataloader = self._initialize_dataloader(split='train')
		self.test_dataloader = self._initialize_dataloader(split='test')



	def train(self):
		best_oa = 0
		running_loss = 0
		for e in range(self.epochs):
			# Initialize wandb log
			if self.config['wandb']:
				wandb.init(project='baaf-s3dis', config=self.config)
				wandb.watch(self.model)

			######### TRAIN #########
			self.model.train()
			for i, (point, label) in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader))):
				point = point.to(self.device)
				label = label.to(self.device)
				logits, p_tilde_layers, p_layers = self.model(point[:,:, :3], point[:, :, 3:])
				loss = self.getLoss(logits, p_tilde_layers, p_layers, label)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()


				running_loss += loss.item()
				if (i+1) % 20 == 0:    
					print('[%d, %5d / %d] loss: %.3f' %
						(e + 1, i + 1, len(self.train_dataloader) ,running_loss/20))
					if self.config['wandb']:
						wandb.log({"Train Batch Loss": running_loss/20})

					running_loss = 0.0

			self.lr_scheduler.step()

			######### VALIDATION #########
			with torch.no_grad():
				num_batches = len(self.test_dataloader)
				total_correct = 0
				total_seen = 0
				loss_sum = 0
				labelweights = np.zeros(self.num_classes)
				total_seen_class = [0 for _ in range(self.num_classes)]
				total_correct_class = [0 for _ in range(self.num_classes)]
				total_iou_deno_class = [0 for _ in range(self.num_classes)]
				self.model = self.model.eval()
				print("==== Epoch ", e, ' Evaluation =====')
				for i, (point, label) in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), smoothing=0.9):
					point = point.to(self.device)
					label = label.to(self.device)
					logits, p_tilde_layers, p_layers = self.model(point[:,:, :3], point[:, :, 3:])				
					pred_val = logits.contiguous().cpu().numpy()
					loss = self.getLoss(logits, p_tilde_layers, p_layers, label)
					loss_sum += loss.item()
					pred_val = np.argmax(pred_val, 2)
					correct = np.sum((pred_val == label))
					total_correct += correct
					label = label.contiguous().cpu().numpy()
					total_seen += (self.config['test']['batch_size'] * self.config['n_points'])
					tmp, _ = np.histogram(label, range(self.num_classes + 1))
					labelweights += tmp

					for l in range(self.num_classes):
						total_seen_class[l] += np.sum((label == l))
						total_correct_class[l] += np.sum((pred_val == l) & (label == l))
						total_iou_deno_class[l] += np.sum(((pred_val == l) | (label == l)))

				labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
				mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
				print('eval mean loss: %f' % (loss_sum / float(num_batches)))
				print('eval point avg class IoU: %f' % (mIoU))
				print('eval point accuracy: %f' % (total_correct / float(total_seen)))
				print('eval point avg class acc: %f' % (
				    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

				iou_per_class_str = '------- IoU --------\n'
				for l in range(self.num_classes):
					iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
					   self.label2names[l] + ' ' * (14 - len(self.label2names[l])), labelweights[l - 1],
					    total_correct_class[l] / float(total_iou_deno_class[l]))

				print(iou_per_class_str)
				print('Eval mean loss: %f' % (loss_sum / num_batches))
				print('Eval accuracy: %f' % (total_correct / float(total_seen)))

				wandb.log({"OA": total_correct/ float(total_seen),
						  "mIoU": mIoU})

				wandb.log({"Validation Loss", loss_sum / float(num_batches)})

				if mIoU >= best_iou:
					best_oa = mIoU
					print('Updated best IoI & saveing model...')
					savepath = os.path.join(self.args.exp_path, 'best_model.pth')
					print('Saving best model at  %s' % savepath)
					state = {
					    'epoch': e,
					    'class_avg_iou': mIoU,
					    'model_state_dict': self.model.state_dict(),
					}
					torch.save(state, savepath)
					print('Saving model....')
				print('Best mIoU: %f' % best_iou)









	### Attribute Initializers ###
	def _initialize_model(self, model='baaf'):
		if model == 'baaf':
			model = BAAFNet(k=self.config['model']['k'],
							n_points=self.config['n_points'],
							num_classes=self.num_classes,
							dims=list(map(int, config['model']['dims'].split(","))))
		return model.to(self.device)

	def _initialize_optimizer(self, opt='adam'):
		if opt=='adam':
			optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])
			return optimizer

	def _initialize_lr_scheduler(self, step_size=10, gamma=0.5, schedule='step'):
		if schedule=='step':
			scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
			return scheduler

	def _initialize_dataloader(self, split='train'):
		if split == 'train':
			dataloader = DataLoader(self.train_dataset, batch_size=self.config['train']['batch_size'],
									num_workers=self.config['train']['num_workers'])
		elif split == 'test':
			dataloader = DataLoader(self.test_dataset, batch_size=self.config['test']['batch_size'],
									num_workers=self.config['test']['num_workers'])
		return dataloader
	
	def getLoss(self, logits, p_tilde_layers, p_layers, label):
		# Cross-entropy loss for semantic classification
		logits = logits.view(-1, logits.shape[2])
		label = label.view(-1)
		loss = torch.nn.functional.cross_entropy(logits, label)

		# Point Augmentation Loss
		for p_tilde, p, w in zip(p_tilde_layers, p_layers, self.aug_loss_weights):
			p = torch.unsqueeze(p, dim=2).expand(-1, -1, self.model.k, -1)
			p_diff = torch.norm(torch.mean(p_tilde-p, dim=2))
			loss += w * p_diff
		return loss







if __name__ == '__main__':
	args = argument_parser()
	config = parse_yaml(os.path.join(args.exp_path, 'config.yaml'))
	train_dataset = S3DISDataset(split='train', num_point=config['n_points'], transform=T_S3DIS)
	test_dataset = S3DISDataset(split='test', num_point=config['n_points'], transform=T_S3DIS)
	trainer = Trainer(args=args,
					  config=config, 
					  train_dataset=train_dataset,
					  test_dataset=test_dataset)
	trainer.train()