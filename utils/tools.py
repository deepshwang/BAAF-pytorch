import yaml



def parse_yaml(fpath):
	with open(fpath) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	return config