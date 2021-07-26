import argparse
import torch
from datetime import datetime
def argument_parser():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('--exp_path', 
						default='/media/TrainDataset/sungwon95/experiments/baaf/210720')


	args = parser.parse_args()

	return args