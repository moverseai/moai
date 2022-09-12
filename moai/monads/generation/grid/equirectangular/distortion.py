from moai.monads.generation.grid.equirectangular.grid import Equirectangular as Grid

import torch

#NOTE: from WS-PSNR
'''
    double getWeight(int form,int i,int j,int width,int height){
	double a;
    //======  format0: Equirectangular  =======  
	if (form == 0) {
		a = cos((j-(height/2-0.5))*3.1415926/height);
		return a;
	}
'''

class Distortion(Grid):
    def __init__(self,
        mode:               str='latitude',
        width:              int=512,
        persistent:         bool=True,
    ):
        super(Distortion, self).__init__(
            mode='pi', order='latlong', inclusive=False,
            width=width, long_offset_pi=0.0,
            persistent=persistent,
        )
        if mode == 'latitude':
            self.grid = torch.cos(self.grid[:, 0, ...]).unsqueeze(1)
