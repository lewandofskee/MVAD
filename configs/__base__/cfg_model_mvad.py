from argparse import Namespace


class cfg_model_mvad(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_t = Namespace()
		self.model_t.name = 'timm_wide_resnet50_2'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='',
							  strict=False, features_only=True, out_indices=[1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'de_wide_resnet50_2'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
		self.model = Namespace()
		self.model.name = 'mvad'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t, model_s=self.model_s)
