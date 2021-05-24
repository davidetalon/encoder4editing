from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, preprocess=None):

		attributes_path = Path(root) / "list_attr_celeba.txt"
		attr = pandas.read_csv(attributes_path, delim_whitespace=True, header=1)
		split_map = {
			"train": 0,
			"valid": 1,
			"test": 2,
			"all": None,
			}
		split_ = split_map[split.lower()]
		attributes_path = Path(root) / "list_eval_partition.txt"
		splits = pandas.read_csv(
			attributes_path, delim_whitespace=True, header=None, index_col=0
        )
        mask = slice(None) if split_ is None else (splits[1] == split_)
        self.attr = torch.as_tensor(attr[mask].values)
		print(attr)
		
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im

if __name__ == '__main__':
	dataset = InferenceDataset("/home/davidetalon/Dev/learning-self/data/raw/celeba", split='train' )