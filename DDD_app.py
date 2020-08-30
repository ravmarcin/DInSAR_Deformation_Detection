from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import wxmplot.interactive as wi
import scipy.ndimage as sc_nd
from sklearn.model_selection import train_test_split
import pandas
import random
"""-------------------------------------------------------------------------------------------------------------"""

class DDD(object):

    """ Wprowadzenie zmiennych """
    def __init__(self, multi=True, c_valid=True, correlation=True, lambda_=1000,
                 learn_rate=False, lambda_multi=(-3,5), art_impact=True, no_data_conversion=True,
                 multi_mask=True, #parameter_update=False,
                 trend_parameter=True, deformation_parameter=True, semi_var_parameter=True, var_parameter=True,
                 standard_dev=True, jump_remove=True, gaussian_par=False, depths=[0.000,(-0.005),(-0.010),(-0.020)],
                 masks=[5,9,15],image_import=True,crop_=False,non_val_reduction=True,
                 change_mask=False,left=8,upper=8,right=132,lower=179,data_type='csv',image_import_shape=None,
                 deformation_type=-1,
                 parameters=True, standard_technique=True, reshaping_technique=True, ovr_technique=True,
                 n_iter=50, shuffle=True, random_state=None,
                 image_cutter=None,
                 max_in_loop=150000,
                 rm_density_=10,rm_seed_=3,rm_min_population_factor=0.01,rm_mask_size=31,
                 max_rate=0.9,
                 max_impact_px=10,
                 CV_iter=10):
        """ Basic mask and image settings  """
        self.change_mask = change_mask
        self.image_cutter = image_cutter
        self.multi_mask = multi_mask
        """ Logistic regression function setting  """
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        self.lambda_ = lambda_
        self.learn_rate = learn_rate
        self.lambda_multi = lambda_multi
        self.multi = multi
        """ Standardization technique setting """
        self.standard_technique = standard_technique
        """ Multiclass setting """
        self.ovr_technique = ovr_technique
        """ Import settings """
        self.left = left
        self.upper = upper
        self.right = right
        self.lower = lower
        self.data_type = data_type
        self.image_import_shape = image_import_shape
        self.non_val_reduction = non_val_reduction
        self.image_import = image_import
        self.crop_ = crop_
        """ Mask stacking settings """
        self.masks = masks
        self.max_in_loop = max_in_loop
        """ Parametrization settings """
        self.deformation_type = deformation_type
        self.parameters = parameters
        self.depths = depths
        self.trend_parameter = trend_parameter
        self.deformation_parameter = deformation_parameter
        self.var_parameter = var_parameter
        self.semi_var_parameter = semi_var_parameter
        self.standard_dev = standard_dev
        self.gaussian_par = gaussian_par
        #self.parameter_update = parameter_update
        self.parameters_number = np.array([0,0,0,0,0,0])
        self.reshaping_technique = reshaping_technique
        """ Removing jumps setting """
        self.jump_remove = jump_remove
        self.rm_density_ = rm_density_
        self.rm_seed_ = rm_seed_
        self.rm_min_population_factor = rm_min_population_factor
        self.rm_mask_size = rm_mask_size
        """ Decorelation parameter """
        self.max_rate = max_rate
        self.correlation = correlation
        """ Art-impact parameter """
        self.art_impact = art_impact
        self.max_impact_px = max_impact_px
        """ Non value data finder """
        self.no_data_conversion = no_data_conversion
        """ Cross validation parameter """
        self.CV_iter = CV_iter
        self.c_valid = c_valid

    """----------------------------------------------------------------------------------------"""
    """---------------------------------- Funkcje podstawowe ----------------------------------"""
    """----------------------------------------------------------------------------------------"""

    """ Import danych
     
     Import formatów obrazów oraz txt/csv.
     Z dodatkowymi funkcjami obcięcia, lustrzanego odbicia (obraz pobrany w formacie od dołu do góry) 
     oraz redukcji pikseli bez wartości.
     """
    def import_data(self,directory,non_value=0):
        if self.image_import:
            im_origo = ['None']
            im_res = ['None']
            im_system = ['None']
            if self.data_type == 'image':
                image_ = Image.open(directory)
                image_C = image_
                if self.crop_:
                    image_C = image_.crop((self.left,self.upper,self.right,self.lower))
                image_C = np.array(image_C)[::-1]
                image_N = image_C
                if self.non_val_reduction:
                    image_N = np.nan_to_num(image_C, nan=non_value)
            elif self.data_type == 'txt' or self.data_type == 'csv':
                image_ = pandas.read_csv(directory)
                image_ = np.array(image_, dtype=float)[::-1]
                image_C = image_
                if self.crop_:
                    image_C = image_[self.left:self.right,self.upper:self.lower]
                image_N = image_C
                if self.non_val_reduction:
                    image_N = np.nan_to_num(image_C, nan=non_value)
            elif self.data_type == 'GeoTIFF':
                image_ = Image.open(directory)
                im_origo = [image_.tag[33922][3], image_.tag[33922][4]]
                im_res = [image_.tag[33550][0], image_.tag[33550][1]]
                im_system = [image_.tag[34737][0]]
                image_C = image_
                if self.crop_:
                    image_C = image_.crop((self.left,self.upper,self.right,self.lower))
                    im_origo = [im_origo[0] + (self.left * im_res[0]), im_origo[1] - (self.upper * im_res[0])]
                image_C = np.array(image_C)[::-1]
                image_N = image_C
                if self.non_val_reduction:
                    image_N = np.nan_to_num(image_C, nan=non_value)
            self.image_import_shape = [image_N.shape[0],image_N.shape[1]]
            print('Data import: DONE ')
            return image_N, im_origo, im_res, im_system

    """ Maska podstawowa """
    def Mask(self,image_,i,j,size):
        Mask_dif = int(size / 2 + 0.5)
        Mask_ = image_[i-Mask_dif+1:i+Mask_dif,j-Mask_dif+1:j+Mask_dif]
        return Mask_

    """ Maska podłużna """
    def Mask_horizontal(self,image_,i,size):
        Mask_dif = int(size / 2 + 0.5)
        Mask_ = image_[i-Mask_dif+1:i+Mask_dif,:,:]
        return Mask_

    """ Maska poprzeczna """
    def Mask_vertical(self,image_,i,size):
        Mask_dif = int(size / 2 + 0.5)
        Mask_ = image_[:,i-Mask_dif+1:i+Mask_dif]
        return Mask_

    """ Określenie osi """
    def Axis(self,image_,m,mask_size):
        Mask_dif = int(mask_size / 2 + 0.5)
        Axis_image = range((0 + Mask_dif), (image_.shape[m] - Mask_dif))
        return Axis_image

    """ Obcinanie obrazu """
    def Image_cutter_(self,image_to_cut,smaller_image):
        size_dif = [(image_to_cut.shape[0] - smaller_image.shape[0]),
                    (image_to_cut.shape[1] - smaller_image.shape[1])]
        image_cut = image_to_cut[int(size_dif[0] / 2):int(image_to_cut.shape[0] - size_dif[0] / 2),
                    int(size_dif[1] / 2):int(image_to_cut.shape[1] - size_dif[1] / 2)]
        return image_cut

    """ Transformacja wymiarów obrazu na macierz o n-liczbie pikseli i m-liczbie cech"""
    def Reshaping(self,image_,no_data,bound_mask):
        if self.reshaping_technique:
            if len(np.shape(image_)) < 3:
                image_ = np.atleast_3d(image_)
            else:
                pass
            image_list = []
            if no_data:
                for i in range(image_.shape[0]):
                    for j in range(image_.shape[1]):
                        if bound_mask[i,j] > 0:
                            pass
                        else:
                            image_list.append(image_[i,j,:])
            else:
                for i in range(image_.shape[0]):
                    for j in range(image_.shape[1]):
                        image_list.append(image_[i,j,:])
            return np.vstack(image_list)

    """ Transformacja wymiarów macierzy 1xn na obraz mxm"""
    def Reshaping_back(self,Matrix_,input_image_shape_0,input_image_shape_1):
        if self.reshaping_technique:
            image_ = []
            if self.parameters:
                axis0 = input_image_shape_0-self.image_cutter
                axis1 = input_image_shape_1-self.image_cutter
                for i in range(axis0):
                    image_.append(np.array(Matrix_[i * axis1:(i + 1) * axis1 - 1]))
            else:
                axis0 = input_image_shape_0
                axis1 = input_image_shape_1
                for i in range(axis0):
                    image_.append(np.array(Matrix_[i*axis1:(i+1)*axis1-1]))
            return image_

    """ Uszeregowanie masek pikseli w formie listy macierzy o wymiarach MxMxN, gdzie:
     M - wymiar maski, N - liczba pikseli"""
    def Mask_stacking(self,masks,image_,no_data,bound_mask):
        if self.multi_mask:
            masks = masks
        else:
            masks = [np.array(masks)]
        max_mask = np.max(masks)
        axis_0, axis_1 = image_.shape[0], image_.shape[1]
        all_masks_stack = []
        if no_data:
            for mask in masks:
                mask_stack_axis1 = np.atleast_3d(np.zeros((axis_0, mask)))
                bound_mask_axis1 = np.atleast_3d(np.zeros((axis_0, mask)))
                for i in self.Axis(image_=image_, m=1, mask_size=max_mask):
                    mask_stack_axis1 = np.append(mask_stack_axis1,
                                                 np.atleast_3d(self.Mask_vertical(image_=image_, i=i, size=mask)),
                                                 axis=2)
                    bound_mask_axis1 = np.append(bound_mask_axis1,
                                                 np.atleast_3d(self.Mask_vertical(image_=bound_mask, i=i,
                                                                                  size=mask)), axis=2)
                mask_stack_axis1 = mask_stack_axis1[:, :, 1:]
                bound_mask_axis1 = bound_mask_axis1[:, :, 1:]
                mask_stack_axis2 = np.atleast_3d(np.zeros((mask, mask)))
                for i in self.Axis(image_=image_, m=0, mask_size=max_mask):
                    if np.max(self.Mask_horizontal(image_=bound_mask_axis1,i=i, size=mask)) > 0:
                        pass
                    else:
                        mask_stack_axis2 = np.append(mask_stack_axis2,
                                                     np.atleast_3d(self.Mask_horizontal(image_=mask_stack_axis1,
                                                                                        i=i, size=mask)), axis=2)
                all_masks_stack.append(mask_stack_axis2[:, :, 1:])
        else:
            for mask in masks:
                mask_stack_axis1 = np.atleast_3d(np.zeros((axis_0,mask)))
                for i in self.Axis(image_=image_, m=1, mask_size=max_mask):
                    mask_stack_axis1 = np.append(mask_stack_axis1,
                                                 np.atleast_3d(self.Mask_vertical(image_=image_, i=i, size=mask)),
                                                 axis=2)
                mask_stack_axis1 = mask_stack_axis1[:,:,1:]
                mask_stack_axis2 = np.atleast_3d(np.zeros((mask, mask)))
                for i in self.Axis(image_=image_, m=0, mask_size=max_mask):
                    mask_stack_axis2 = np.append(mask_stack_axis2,
                                                 np.atleast_3d(self.Mask_horizontal(image_=mask_stack_axis1,
                                                                                    i=i, size=mask)),axis=2)
                all_masks_stack.append(mask_stack_axis2[:,:,1:])
        return all_masks_stack

    """ Optymalizacja szeregowania masek"""
    def Mask_stacking_opt(self,masks,image_,no_data,bound_mask):
        self.image_cutter = np.max(masks) + 1
        if (image_.shape[1] * image_.shape[0]) > self.max_in_loop * 2:
            opt_size = int(self.max_in_loop / image_.shape[1])
            max_mask = max(masks) + 1
            factor_size_i = int(image_.shape[0] / opt_size)
            Mask_stack_ = []
            for i in range(factor_size_i):
                image_slice = image_[(opt_size * i):(opt_size * (i + 1) + max_mask),:]
                image_slice_mask = self.Mask_stacking(image_=image_slice, masks=masks, no_data=no_data,
                                                      bound_mask=bound_mask)
                Mask_stack_.append(image_slice_mask)
            image_slice = image_[(opt_size * factor_size_i):,:]
            image_slice_mask = self.Mask_stacking(image_=image_slice, masks=masks, no_data=no_data,
                                                  bound_mask=bound_mask)
            Mask_stack_.append(image_slice_mask)
            Mask_stack_all = []
            for i in range(np.size(masks)):
                Mask_stack_all.append(np.atleast_3d(np.zeros((masks[i],masks[i]))))
            for i in Mask_stack_:
                for j in range(np.size(masks)):
                    Mask_stack_all[j] = np.append(Mask_stack_all[j],i[j],axis=2)
            for i in range(len(Mask_stack_all)):
                Mask_stack_all[i] = Mask_stack_all[i][:,:,1:]
        else:
            Mask_stack_all = self.Mask_stacking(image_=image_, masks=masks, no_data=no_data,bound_mask=bound_mask)
        print('Mask stacking operation: DONE')
        return Mask_stack_all

    """ Zobrazowanie danych """
    def Ploting_interactive(self,image_):
        wi.imshow(image_, colormap='viridis')

    """----------------------------------------------------------------------------------------"""
    """------------------------------------ Parametryzacja ------------------------------------"""
    """----------------------------------------------------------------------------------------"""

    """ Parametr wariancji
    Wartość wariancji w zależności oddalanie się od środka maski, 
    a także wartości semiwariacji w masce dla każdej możliwej odległości poziomej i pionowej.
    """
    def Variance_line_from_px(self,masks_stack,mask_size):
        pixels_no = masks_stack.shape[2]
        grid = np.indices((mask_size, mask_size))
        grid_NS = grid[0] - int(mask_size / 2)
        grid_EW = grid[1] - int(mask_size / 2)
        d_to_center = np.sqrt((grid_NS ** 2) + (grid_EW ** 2))
        d_to_center_stack = np.repeat(np.atleast_3d(d_to_center), repeats=pixels_no, axis=2)
        px_matrix = masks_stack[int(mask_size / 2), int(mask_size / 2), :]
        px_matrix_mask_shape = np.ones((mask_size,mask_size,pixels_no)) * px_matrix
        masks_stack_dif = masks_stack - px_matrix_mask_shape
        masks_stack_dif_reshape = np.reshape(masks_stack_dif, (mask_size ** 2, pixels_no), order='F')
        d_to_center_reshape = np.reshape(d_to_center_stack, (mask_size ** 2, pixels_no), order='F')
        a_1 = np.ones(mask_size ** 2).T
        """ Płaszczyzna płaska """
        a_2d_flat = np.array([a_1, d_to_center_reshape[:,0].T])
        a_2d_flat_dot_inv = np.linalg.pinv(a_2d_flat.dot(a_2d_flat.T))
        a_3d_flat_dot_inv = np.repeat(np.atleast_3d(a_2d_flat_dot_inv), repeats=pixels_no, axis=2)
        a_1_3d = np.repeat(np.atleast_2d(a_1).T, repeats=pixels_no, axis=1)
        a_flat = np.array([a_1_3d, d_to_center_reshape])
        la_flat_3d_dot_ = self.Dot_3D(a_flat,masks_stack_dif_reshape.T)
        x_flat_3d = self.Dot_3D(a_3d_flat_dot_inv,la_flat_3d_dot_.T)
        v_flat_3d = self.Dot_3D(np.transpose(a_flat,axes=(1,0,2)),x_flat_3d.T) - masks_stack_dif_reshape
        sig_flat_3d = np.sqrt(self.Dot_2D_plus(v_flat_3d,v_flat_3d.T) / (mask_size ** 2 - 2))
        """ Płaszczyzna krzywa """
        a_2d_curve = np.array([a_1, d_to_center_reshape[:,0].T, d_to_center_reshape[:,0].T ** 2])
        a_2d_curve_dot_inv = np.linalg.pinv(a_2d_curve.dot(a_2d_curve.T))
        a_3d_curve_dot_inv = np.repeat(np.atleast_3d(a_2d_curve_dot_inv), repeats=pixels_no, axis=2)
        a_curve = np.array([a_1_3d, d_to_center_reshape, d_to_center_reshape ** 2])
        la_curve_3d_dot_ = self.Dot_3D(a_curve,masks_stack_dif_reshape.T)
        x_curve_3d = self.Dot_3D(a_3d_curve_dot_inv,la_curve_3d_dot_.T)
        v_curve_3d = self.Dot_3D(np.transpose(a_curve,axes=(1,0,2)),x_curve_3d.T) - masks_stack_dif_reshape
        sig_curve_3d = np.sqrt(self.Dot_2D_plus(v_curve_3d,v_curve_3d.T) / (mask_size ** 2 - 2))
        print('Variance from the pixel: DONE')
        return x_flat_3d[1:,:], sig_flat_3d, x_curve_3d[1:,:], sig_curve_3d

    def Variance_in_mask_with_step(self,masks_stack,mask_size):
        pixels_no = masks_stack.shape[2]
        semi_var_ = np.zeros((1, pixels_no))
        for i in range(mask_size-1):
            temp_mask_ax0_1 = masks_stack[0:mask_size - (i + 1),:,:]
            temp_mask_ax0_2 = masks_stack[i + 1:, :, :]
            n_temp_ = 2 * mask_size * (mask_size - (i + 1))
            temp_mask_ax0 = np.sum(((temp_mask_ax0_1 - temp_mask_ax0_2) ** 2), axis=(0,1))
            temp_mask_ax1_1 = masks_stack[:, 0:mask_size - (i + 1),:]
            temp_mask_ax1_2 = masks_stack[:, i + 1:, :]
            temp_mask_ax1 = np.sum(((temp_mask_ax1_1 - temp_mask_ax1_2) ** 2), axis=(0,1))
            semi_var_step = (temp_mask_ax0 + temp_mask_ax1) / (2 * n_temp_)
            semi_var_ = np.append(semi_var_,np.atleast_2d(semi_var_step),axis=0)
        print('Variance in the mask: DONE')
        return semi_var_[1:,:].T

    """ Detekcja trendu 
    Wpasowanie maski w płaszczyzne płaską oraz krzywą. 
    W rezultacie orztymujemy drugie i trzecie parametry płaszczyz, 
    wartości odstające (odchylenie std) oraz błąd dla środkowego piksela.
    """
    """ Wpasowanie metoda najmnijeszych kwadratów"""
    def Least_square_(self,a,l_):
        x = np.linalg.pinv(a.dot(a.T)).dot(l_.dot(a.T))
        v = x.dot(a) - l_
        sig = np.sqrt(v.dot(v.T) / (np.size(l_) - 2))
        v_pix = np.take(v, v.size // 2)
        return x, sig, v_pix

    """ Dot produkt macierzy 2D+1 oraz 1D+1"""
    def Dot_3D(self,x1,x2):
        x2 = np.atleast_3d(x2)
        x1 = np.atleast_3d(x1)
        vector_i_sum = []
        for i in range(x1.shape[1]):
            vector_i = x1[:,i,:] * x2[:,i,0]
            vector_i_sum.append(vector_i)
        dot_product = np.sum(vector_i_sum,axis=0)
        return np.array(dot_product)

    """ Dot produkt macierzy 1D+1 oraz 1D+1"""
    def Dot_2D_plus(self,x1,x2):
        vector_i_sum = []
        for i in range(x1.shape[0]):
            vector_i = x1[i,:] * x2[:,i]
            vector_i_sum.append(vector_i)
        dot_product = np.sum(vector_i_sum,axis=0)
        return np.array(dot_product)

    def Trend_detection_(self, masks_stack, mask_size):
        pixels_no = masks_stack.shape[2]
        grid = np.indices((mask_size, mask_size))
        grid_NS = grid[0] + 1
        grid_EW = grid[1] + 1
        grid_NS_stack = np.repeat(np.atleast_3d(grid_NS), repeats=pixels_no, axis=2)
        grid_EW_stack = np.repeat(np.atleast_3d(grid_EW), repeats=pixels_no, axis=2)
        masks_stack_reshape = np.reshape(masks_stack,(mask_size ** 2, pixels_no), order='F')
        grid_NS_stack_reshape = np.reshape(grid_NS_stack, (mask_size ** 2, pixels_no), order='F')
        grid_EW_stack_reshape = np.reshape(grid_EW_stack, (mask_size ** 2, pixels_no), order='F')
        a_1 = np.ones(mask_size ** 2).T
        """ Płaszczyzna płaska """
        a_2d_flat = np.array([a_1, grid_NS_stack_reshape[:,0].T, grid_EW_stack_reshape[:,0].T])
        a_2d_flat_dot_inv = np.linalg.pinv(a_2d_flat.dot(a_2d_flat.T))
        a_3d_flat_dot_inv = np.repeat(np.atleast_3d(a_2d_flat_dot_inv), repeats=pixels_no, axis=2)
        a_1_3d = np.repeat(np.atleast_2d(a_1).T, repeats=pixels_no, axis=1)
        a_flat = np.array([a_1_3d, grid_NS_stack_reshape, grid_EW_stack_reshape])
        la_flat_3d_dot_ = self.Dot_3D(a_flat,masks_stack_reshape.T)
        x_flat_3d = self.Dot_3D(a_3d_flat_dot_inv,la_flat_3d_dot_.T)
        v_flat_3d = self.Dot_3D(np.transpose(a_flat,axes=(1,0,2)),x_flat_3d.T) - masks_stack_reshape
        sig_flat_3d = np.sqrt(self.Dot_2D_plus(v_flat_3d,v_flat_3d.T) / (mask_size ** 2 - 2))
        v_flat_pix_3d = v_flat_3d[int(mask_size ** 2 / 2),:]
        """ Płaszczyzna krzywa """
        a_2d_curve = np.array([a_1, grid_NS_stack_reshape[:,0].T, grid_EW_stack_reshape[:,0].T,
                              grid_NS_stack_reshape[:,0].T ** 2, grid_EW_stack_reshape[:,0].T ** 2])
        a_2d_curve_dot_inv = np.linalg.pinv(a_2d_curve.dot(a_2d_curve.T))
        a_3d_curve_dot_inv = np.repeat(np.atleast_3d(a_2d_curve_dot_inv), repeats=pixels_no, axis=2)
        a_curve = np.array([a_1_3d, grid_NS_stack_reshape, grid_EW_stack_reshape, grid_NS_stack_reshape ** 2,
                            grid_EW_stack_reshape ** 2])
        la_curve_3d_dot_ = self.Dot_3D(a_curve,masks_stack_reshape.T)
        x_curve_3d = self.Dot_3D(a_3d_curve_dot_inv,la_curve_3d_dot_.T)
        v_curve_3d = self.Dot_3D(np.transpose(a_curve,axes=(1,0,2)),x_curve_3d.T) - masks_stack_reshape
        sig_curve_3d = np.sqrt(self.Dot_2D_plus(v_curve_3d,v_curve_3d.T) / (mask_size ** 2 - 2))
        v_curve_pix_3d = v_curve_3d[int(mask_size ** 2 / 2),:]
        print('Trend detection: DONE')
        return x_flat_3d[1:,:], sig_flat_3d, v_flat_pix_3d, x_curve_3d[1:,:], sig_curve_3d, v_curve_pix_3d

    """ Detekcja deformacji
    Szybkie wyszukiwanie deformacji na podstawie: 
    - określenie czy szukamy osiadań (-1) lub podniesienia (+1),
    - zdefiniowanej wartości granicznej deformacji (depth),
    """
    def Deformation_simple_(self,masks_stack,mask_size,depth):
        if self.deformation_type == -1:
            masks_stack = masks_stack
        elif self.deformation_type == 1:
            masks_stack = -masks_stack
        else:
            print('Depth parameter was not set')
            masks_stack = 0
        Deformation_param = np.sum(np.where(masks_stack <= depth, 1, 0),axis=(0,1))/(mask_size**2)
        print('Simple way deformation detection: DONE')
        return Deformation_param

    """ Obraz odchyleń standardowych w masce"""
    def Standard_deviation_(self,masks_stack):
        Standard_param = np.std(masks_stack,axis=(0,1))
        print('Standard deviation: DONE')
        return Standard_param

    """ Maska różnicowego rozmycia gaussa """
    def Gaussian_dif(self,masks_stack,masks_size):
        pixels_no = masks_stack[0].shape[2]
        start = np.zeros((1,pixels_no))
        for i in range(len(masks_stack)):
            mask_stack = masks_stack[i]
            mask_size = masks_size[i]
            std_p_stack = self.Standard_deviation_(masks_stack=mask_stack)
            std_p_stack_shape = np.ones((mask_size, mask_size, pixels_no)) * std_p_stack
            gaussian_part1 = 1 / (2 * np.pi * (std_p_stack_shape ** 2))
            grid = np.indices((mask_size, mask_size))
            grid_NS = grid[0] - int(mask_size / 2)
            grid_EW = grid[1] - int(mask_size / 2)
            d_to_center = (grid_NS ** 2) + (grid_EW ** 2)
            d_to_center_stack = np.repeat(np.atleast_3d(d_to_center), repeats=pixels_no, axis=2)
            gaussian_part2 = np.exp(- (d_to_center_stack / (2 * (std_p_stack_shape ** 2))))
            gaussian_ = gaussian_part1 * gaussian_part2
            mask_stack_gaussian = np.sum((mask_stack * gaussian_), axis=(0,1))
            start = np.append(start,np.atleast_2d(mask_stack_gaussian),axis=0)
        gaussian_ = start[1:,:]
        dif_gaussian = gaussian_[0:-1,:] - gaussian_[1:,:]
        print('Diferrence Gaussian filter: DONE')
        return dif_gaussian.T

    """ Funkcja parametryzacji """
    def Parametrization(self,masks_stack,masks_size,depths):
        if self.parameters:
            if self.multi_mask:
                masks_size = masks_size
            else:
                masks_size = [np.array(masks_size)]
            pixels_no = masks_stack[0].shape[2]
            bias = np.ones((pixels_no,1))
            for i in range(len(masks_stack)):
                mask_stack = masks_stack[i]
                mask_size = masks_size[i]
                if self.trend_parameter:
                    self.parameters_number[0] = 1
                    td_p = self.Trend_detection_(masks_stack=mask_stack, mask_size=mask_size)
                    for j in range(len(td_p)):
                        bias = np.append(bias,np.atleast_2d(td_p[j]).T, axis=1)
                if self.var_parameter:
                    self.parameters_number[1] = 1
                    vlfp_p = self.Variance_line_from_px(masks_stack=mask_stack, mask_size=mask_size)
                    for j in range(len(vlfp_p)):
                        bias = np.append(bias,np.atleast_2d(vlfp_p[j]).T, axis=1)
                if self.standard_dev:
                    self.parameters_number[2] = 1
                    std_p = self.Standard_deviation_(masks_stack=mask_stack)
                    bias = np.append(bias, np.atleast_2d(std_p).T,axis=1)
                if self.deformation_parameter:
                    self.parameters_number[3] = 1
                    for j in depths:
                        def_p = self.Deformation_simple_(masks_stack=mask_stack, mask_size=mask_size,depth=j)
                        bias = np.append(bias, np.atleast_2d(def_p).T, axis=1)
            masks_size_ar = np.array(masks_size)
            max_mask_size_pos = np.squeeze(np.argwhere(masks_size_ar == max(masks_size_ar)))
            if self.semi_var_parameter:
                self.parameters_number[4] = 1
                semi = self.Variance_in_mask_with_step(masks_stack=masks_stack[max_mask_size_pos],
                                                       mask_size=max(masks_size))
                bias = np.append(bias, np.atleast_2d(semi), axis=1)
            if self.gaussian_par:
                self.parameters_number[5] = 1
                dif_gauss_p = self.Gaussian_dif(masks_stack=masks_stack,masks_size=masks_size)
                bias = np.append(bias, np.atleast_2d(dif_gauss_p), axis=1)
            one_mask = masks_stack[0]
            image_parameters = np.append(bias,np.atleast_2d(one_mask[int(masks_size[0] / 2),
                                                            int(masks_size[0] / 2), :]).T, axis=1)
            print('Parametrization: DONE ')
            return image_parameters

    """ Update parametru """
    """
    def Parametrization_update(self,parameters,masks_stack,depths,function, masks_size):
        if self.parameter_update:
            self.parameters = True
            parameters_number_old = np.copy(self.parameters_number)
            self.trend_parameter, self.var_parameter, self.standard_dev, self.deformation_parameter, \
            self.semi_var_parameter, self.gaussian_par = False, False, False, False, False, False
            function = True
            update_image_parameters = self.Parametrization(masks_stack=masks_stack,masks_size=masks_size,depths=depths)
            update_image_parameters = update_image_parameters[:,1:-1]
            parameters_update = np.copy(parameters)
            par_one_line = parameters_number_old[0] * 6 + parameters_number_old[1] * 4 + parameters_number_old[2] *\
                           1 + parameters_number_old[4] * np.size(depths) + parameters_number_old[5] *\
                           max(masks_size) + parameters_number_old[6]
            tworzymy liniejke, jeżeli w parametrs_number jest zero to dodajemy jedynki o szerokości danego parametru
            a jak jest 1 to dodajemy 0 
            parameter_no = np.squeeze(np.argwhere(self.parameters_number == 1))
            for i in range(np.size(masks_size)):
                if parameter_no == 0:
                    6 + masks_size[i]
                    parameters_update[:,(i + 1) ]
    """




    """ Standaryzacja parametrów uczących i testowych"""
    def Standard_training_new(self,parameters):
        if self.standard_technique:
            P_std = np.copy(parameters)
            self.mean_of_p, self.std_of_p = [], []
            for i in range(parameters.shape[1]-1):
                if parameters[:,(i+1)].std() == 0:
                    P_std[:, (i + 1)] = parameters[:,(i+1)]
                else:
                    mean_of_par = parameters[:, (i + 1)].mean()
                    std_of_par = parameters[:, (i + 1)].std()
                    P_std[:, (i + 1)] = (parameters[:, (i + 1)] - mean_of_par) / std_of_par
                    self.mean_of_p.append(mean_of_par)
                    self.std_of_p.append(std_of_par)
            self.mean_of_p = np.array(self.mean_of_p)
            self.std_of_p = np.array(self.std_of_p)
            print('Standarization: DONE ')
            return P_std

    def Standard_test_new(self,parameters):
        if self.standard_technique:
            P_std = np.copy(parameters)
            for i in range(parameters.shape[1]-1):
                P_std[:,(i+1)] = (parameters[:,(i+1)] - self.mean_of_p[i]) / self.std_of_p[i]
            print('Standarization: DONE ')
            return P_std

    """----------------------------------------------------------------------------------------"""
    """---------------------- ALGORYTM UCZĄCY - REGRESJA LOGISTYCZNA --------------------------"""
    """----------------------------------------------------------------------------------------"""

    """ Przygotowanie danych X i Y treningowych i testowych do wpasowania """
    def Prepare_Y_training_to_fit(self,image_,main_class,no_data,bound_mask):
        self.reshaping_technique = True
        self.ovr_technique = True
        image_reshape = self.Reshaping(image_,no_data=no_data,bound_mask=bound_mask)
        self.image_ovr, image_classes = self.OvR(image_reshape,main_class=main_class)
        return image_classes

    def Prepare_Y_test_to_fit(self,image_,main_class,no_data,bound_mask):
        self.reshaping_technique = True
        self.ovr_technique = True
        image_reshape = self.Reshaping(image_,no_data=no_data,bound_mask=bound_mask)
        image_ovr, image_classes = self.self.OvR(image_reshape,main_class=main_class)
        return image_ovr,image_classes

    def Prepare_X_training_to_fit(self,image_,masks,depths,no_data,bound_mask):
        self.parameters = True
        self.training_masks_stack = self.Mask_stacking_opt(image_=image_,masks=masks,no_data=no_data,
                                                           bound_mask=bound_mask)
        image_parameters = self.Parametrization(masks_stack=self.training_masks_stack,masks_size=masks,depths=depths)
        P_standard = self.Standard_training_new(parameters=image_parameters)
        return P_standard

    def Prepare_X_test_to_fit(self,image_,masks,depths,no_data,bound_mask):
        self.parameters = True
        self.test_masks_stack = self.Mask_stacking_opt(image_=image_,masks=masks,no_data=no_data,bound_mask=bound_mask)
        image_parameters = self.Parametrization(masks_stack=self.test_masks_stack,masks_size=masks,depths=depths)
        P_standard = self.Standard_test_new(parameters=image_parameters)
        return P_standard

    """ Technika One vs Rest (OvR) """
    def OvR(self,Y,main_class):
        if self.ovr_technique:
            image_classes = np.unique(Y)
            classes_sort = []
            for class_ in image_classes:
                if class_ != main_class:
                    classes_sort.append(class_)
            classes_sort.append(main_class)
            Y_OvR = []
            for class_i in classes_sort:
                Y_OvR.append(np.where(Y == class_i,1,0))
            Y_OvR = np.squeeze(np.reshape(Y_OvR, (len(image_classes), np.array(Y_OvR).shape[1])))
            return Y_OvR, classes_sort

    """ Funkcja wpasowania danych uczących
    
     Funkcja kosztu w postaci zlogarytmizowanej funkcji wiarygodności.
     Minimalizacja funkcji kosztu z pomocą gradientu prostego.
     """
    def fit(self, X, Y):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.001, size=X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            """Pegasos eta estymacja"""
            eta = 1. / (self.lambda_*(i+1))
            net_input = self.net_input(x=X)
            output = self.activation(net_input)
            errors = (Y - output)
            self.w_[1:] += eta * (X[:,1:].T.dot(errors) - self.lambda_ * self.w_[1:])
            self.w_[0] += eta * (errors.sum() - self.lambda_ * self.w_[0])
            """ L2 regularyzacja """
            w_norm = np.linalg.norm(self.w_)**2
            L2 = 0.5 * self.lambda_ * w_norm
            """Filtracja wartości 0 logarytmie naturalnym"""
            Log_output = [np.log(value) if value > 0 else 0 for value in output]
            Log_1_output = [np.log(value) if value > 0 else 0 for value in (1 - output)]
            """ Koszt logistyczny """
            cost_no_reg = -Y.dot(Log_output) - ((1 - Y).dot(Log_1_output))
            cost = cost_no_reg + L2
            self.cost_.append(cost)
        #self.Output(X)
        return self

    """ Funkcja 'fit' z uczeniem parametru lambda.
    Wydłużony czas operacji, do dopracowania (współczynnik powinien się nauczyć już po pierwszej klasie)"""
    def fit_with_learning_rate(self, X, Y):
        self.rgen = np.random.RandomState(self.random_state)
        Lambda_dif = self.lambda_multi[1] - self.lambda_multi[0]
        weights_optimization, cost_optimization_sum, cost_optimization = [],[],[]
        for j in np.logspace(start=self.lambda_multi[0], stop=self.lambda_multi[1], num=Lambda_dif + 1, base=10):
            self.w_ = self.rgen.normal(loc=0.0, scale=0.001, size=X.shape[1])
            self.cost_ = []
            for i in range(self.n_iter):
                """Pegasos eta estymacja"""
                eta = 1. / (j*(i+1))
                net_input = self.net_input(x=X)
                output = self.activation(net_input)
                errors = (Y - output)
                self.w_[1:] += eta * (X[:,1:].T.dot(errors) - j * self.w_[1:])
                self.w_[0] += eta * (errors.sum() - j * self.w_[0])
                """ L2 regularyzacja """
                w_norm = np.linalg.norm(self.w_)**2
                L2 = 0.5 * j * w_norm
                """Filtracja wartości 0 logarytmie naturalnym"""
                Log_output = [np.log(value) if value > 0 else 0 for value in output]
                Log_1_output = [np.log(value) if value > 0 else 0 for value in (1 - output)]
                """ Koszt logistyczny """
                cost_no_reg = -Y.dot(Log_output) - ((1 - Y).dot(Log_1_output))
                cost = cost_no_reg + L2
                self.cost_.append(cost)
            cost_optimization.append(self.cost_)
            weights_optimization.append(self.w_)
            """ Minimalizacja kosztu w zależności od współczynnika uczenia"""
            cost_optimization_sum.append(sum(self.cost_[int(self.n_iter / 2):-1])/int(self.n_iter/2))
        position = np.squeeze(np.argwhere(cost_optimization_sum == min(cost_optimization_sum)))
        cost_optimization = np.vstack(cost_optimization)
        weights_optimization = np.vstack(weights_optimization)
        self.cost_ = cost_optimization[position,:]
        self.w_ = weights_optimization[position,:]
        #self.Output(X)
        print('Algorithm learning: DONE ')
        return self

    """ Wpasowanie danych wieloklasowych
     Wpasowanie poprzez technikę OvR oraz klasa do klasy w celu znalezienia najlepiej pasujących wag.
     Wyznaczenie obrazów prawdopodobieństwa 3d """
    def fit_all_class(self,X,Y_OvR):
        if self.multi:
            self.W_, self.Cost_ = [],[]
            for i in range(Y_OvR.shape[0]):
                W_i, Cost_i = [], []
                Xi = X[Y_OvR[i, :] == 1]
                Yi = np.ones((1, Xi.shape[0]))
                for j in range(Y_OvR.shape[0]):
                    if i != j:
                        Xj = X[Y_OvR[j,:] == 1]
                        Xij = np.append(Xi,Xj,axis=0)
                        Yj = np.zeros((1,Xj.shape[0]))
                        Yij = np.append(Yi,Yj)
                        if self.learn_rate:
                            self.fit_with_learning_rate(Xij,Yij)
                        else:
                            self.fit(Xij,Yij)
                        W_i.append(self.w_)
                        Cost_i.append(self.cost_)
                W_i, Cost_i = np.vstack(W_i), np.vstack(Cost_i)
                """ Wyznaczenie najlepszych predyktorów dla danej klasy"""
                Cost_ij_grad_sum = []
                for i in range(Cost_i.shape[0]):
                    Cost_ij_grad_sum.append(sum(np.where(Cost_i[i,0:-1]-Cost_i[i,1:] > 0,1,0)))
                position = np.squeeze(np.argwhere(Cost_ij_grad_sum == max(Cost_ij_grad_sum)))
                if np.size(position) > 1:
                    Cost_ij_min = []
                    for i in position:
                        Cost_ij_min.append(Cost_i[i,-1])
                    position = np.squeeze(np.argwhere(Cost_i[:,-1] == min(Cost_ij_min)))
                self.W_.append(W_i[position, :])
                self.Cost_.append(Cost_i[position,:])
            self.W_ = np.vstack(self.W_)
            self.Cost_ = np.vstack(self.Cost_)
            #self.probe_images = self.probabilities_imaging(x=X,class_number=Y_OvR.shape[0],
            #                                               input_image_shape_0=self.image_import_shape[0],
            #                                               input_image_shape_1=self.image_import_shape[1])
            print('Algorithm learning: DONE ')
            return self

    """ Wpasowanie danych wieloklasowych
     Wpasowanie poprzez technikę OvR z równymi wielkościami klas.
     Wyznaczenie obrazów prawdopodobieństwa 3d """
    def fit_all_class_equal(self,X,Y_OvR,n_rand):
        if self.multi:
            self.W_, self.Cost_ = [],[]
            for i in range(Y_OvR.shape[0]):
                W_i, Cost_i = [], []
                Xi = X[Y_OvR[i, :] == 1]
                Yi = np.ones((1, Xi.shape[0]))
                for j in range(n_rand):
                    Xj = X[Y_OvR[i,:] == 0]
                    if Xj.shape[0] > Xi.shape[0]:
                        Xj_rand_col = random.sample(range(Xj.shape[0]),k=Xi.shape[0])
                        Xj_rand = Xj[Xj_rand_col, :]
                        Xij = np.append(Xi,Xj_rand,axis=0)
                        Yj = np.zeros((1,Xj_rand.shape[0]))
                        Yij = np.append(Yi,Yj)
                    else:
                        Xi_rand_col = random.sample(range(Xi.shape[0]), k=Xj.shape[0])
                        Xi_rand = Xi[Xi_rand_col, :]
                        Yi = np.ones((1, Xi_rand.shape[0]))
                        Xij = np.append(Xi_rand,Xj,axis=0)
                        Yj = np.zeros((1,Xj_rand.shape[0]))
                        Yij = np.append(Yi,Yj)
                    if self.learn_rate:
                        self.fit_with_learning_rate(Xij,Yij)
                    else:
                        self.fit(Xij,Yij)
                    W_i.append(self.w_)
                    Cost_i.append(self.cost_)
                W_i, Cost_i = np.vstack(W_i), np.vstack(Cost_i)
                """ Wyznaczenie najlepszych predyktorów dla danej klasy"""
                Cost_ij_grad_sum = []
                for i in range(Cost_i.shape[0]):
                    Cost_ij_grad_sum.append(sum(np.where(Cost_i[i,0:-1]-Cost_i[i,1:] > 0,1,0)))
                position = np.squeeze(np.argwhere(Cost_ij_grad_sum == max(Cost_ij_grad_sum)))
                if np.size(position) > 1:
                    Cost_ij_min = []
                    for i in position:
                        Cost_ij_min.append(Cost_i[i,-1])
                    position = np.squeeze(np.argwhere(Cost_i[:,-1] == min(Cost_ij_min)))
                self.W_.append(W_i[position, :])
                self.Cost_.append(Cost_i[position,:])
            self.W_ = np.vstack(self.W_)
            self.Cost_ = np.vstack(self.Cost_)
            self.probe_images = self.probabilities_imaging(x=X,class_number=Y_OvR.shape[0],
                                                           input_image_shape_0=self.image_import_shape[0],
                                                           input_image_shape_1=self.image_import_shape[1])
            print('Algorithm learning: DONE ')
            return self

    """ Dekorelacja parametrów 
    Wykonanie poprzez przeniesie skorelowanych danych w nową przestrzeń 
    Skorelowane dane definije się poprzez wyższy niż --self.max_rate-- wartość bezwględna współczynnika korelacji """
    def decorelation(self,X):
        X_conv = X[:,1:-1]
        r_table = self.Correlation_check(X_conv)
        r_table_noOne = np.copy(r_table)
        r_table_noOne[r_table_noOne > 0.99999 ] = 0
        rate = np.max(np.absolute(r_table_noOne))
        epsilon = 0.00000000001
        pos_seq = []
        while rate >= self.max_rate:
            position = np.squeeze(np.argwhere(np.absolute(r_table_noOne) >= rate - epsilon))
            position = position[0]
            if position[0] < r_table_noOne.shape[1]:
                if position[0] == 0:
                    X_conv_n = X_conv[:,position[0]+1:position[1]]
                else:
                    X_conv_n = X_conv[:,0:position[0]]
                    part1 = X_conv[:,position[0]+1:position[1]]
                    for i in range(part1.shape[1]):
                        X_conv_n = np.append(X_conv_n, np.atleast_2d(part1[:,i]).T, axis=1)
                part2 = X_conv[:,position[1]+1:]
                for i in range(part2.shape[1]):
                    X_conv_n = np.append(X_conv_n, np.atleast_2d(part2[:,i]).T, axis=1)
            else:
                X_conv_n = X_conv[:, 0:position[0]]
                part1 = X_conv[:, position[0] + 1:position[1]]
                for i in range(part1.shape[1]):
                    X_conv_n = np.append(X_conv_n, np.atleast_2d(part1[:, i]).T, axis=1)
            par_conv = np.atleast_2d(X_conv[:,position[0]] * X_conv[:,position[1]])
            pos_seq.append(position)
            X_conv_n = np.append(X_conv_n, par_conv.T, axis=1)
            r_table_noOne = self.Correlation_check(X_conv_n)
            r_table_noOne[r_table_noOne > 0.99999 ] = 0
            rate = np.max(np.absolute(r_table_noOne))
            X_conv = np.copy(X_conv_n)
        X_decorel = np.atleast_2d(X[:,0]).T
        for i in range(X_conv.shape[1]):
            X_decorel = np.append(X_decorel, np.atleast_2d(X_conv[:, i]).T, axis=1)
        X_decorel = np.append(X_decorel, np.atleast_2d(X[:,-1]).T, axis=1)
        print(u'New number of parameters:')
        print(X_decorel.shape[1])
        print(u'Max. R rate:')
        print(rate)
        return X_decorel, pos_seq

    def decorelation_test(self,X,seq):
        X_conv = X[:,1:-1]
        seq = np.array(seq)
        for i in range(seq.shape[0]):
            position = seq[i,:]
            if position[0] < X_conv.shape[1]:
                if position[0] == 0:
                    X_conv_n = X_conv[:,position[0]+1:position[1]]
                else:
                    X_conv_n = X_conv[:,0:position[0]]
                    part1 = X_conv[:,position[0]+1:position[1]]
                    for i in range(part1.shape[1]):
                        X_conv_n = np.append(X_conv_n, np.atleast_2d(part1[:,i]).T, axis=1)
                part2 = X_conv[:,position[1]+1:]
                for i in range(part2.shape[1]):
                    X_conv_n = np.append(X_conv_n, np.atleast_2d(part2[:,i]).T, axis=1)
            else:
                X_conv_n = X_conv[:, 0:position[0]]
                part1 = X_conv[:, position[0] + 1:position[1]]
                for i in range(part1.shape[1]):
                    X_conv_n = np.append(X_conv_n, np.atleast_2d(part1[:, i]).T, axis=1)
            par_conv = np.atleast_2d(X_conv[:,position[0]] * X_conv[:,position[1]])
            X_conv_n = np.append(X_conv_n, par_conv.T, axis=1)
            X_conv = np.copy(X_conv_n)
        X_decorel = np.atleast_2d(X[:,0]).T
        for i in range(X_conv.shape[1]):
            X_decorel = np.append(X_decorel, np.atleast_2d(X_conv[:, i]).T, axis=1)
        X_decorel = np.append(X_decorel, np.atleast_2d(X[:,-1]).T, axis=1)
        return X_decorel

    """ Obrazy prawdopodobieństwa """
    def probabilities_imaging(self,x,class_number,input_image_shape_0,input_image_shape_1):
        prob_image_reshape = []
        for xi in x:
            prob_image_reshape.append(np.array(self.probabilities(x=xi)))
        prob_image_reshape = np.vstack(prob_image_reshape)
        self.reshaping_technique = True
        image_starter = np.atleast_3d(self.Reshaping_back(np.zeros(prob_image_reshape.shape[0]),
                                                          input_image_shape_0=input_image_shape_0,
                                                          input_image_shape_1=input_image_shape_1))
        for i in range(class_number):
            probe_image_i = np.atleast_3d(self.Reshaping_back(prob_image_reshape[:, i],
                                          input_image_shape_0=input_image_shape_0,
                                          input_image_shape_1=input_image_shape_1))
            image_starter = np.append(image_starter, probe_image_i, axis=2)
        probe_images_ = image_starter[:, :, 1:]
        print('Probabilistic images: DONE ')
        return probe_images_

    """ Lista prawdopodobieństwa """
    def predict_list(self,x,classes_):
        prob_list_reshape = []
        for xi in x:
            prob_list_reshape.append(np.array(self.probabilities(x=xi)))
        prob_array_reshape = np.array(prob_list_reshape)
        position = np.argmax(prob_array_reshape, axis=1)
        classes_ = np.array(classes_)
        class_list = np.ones((position.shape[0],classes_.shape[0]))
        class_list_n = class_list * classes_
        predict_list = class_list_n[0,position]
        return predict_list

    """ Błąd predykcji """
    def prediction_error_rate(self,x,classes_,y):
        y = np.squeeze(y)
        predict_list = self.predict_list(x=x,classes_=classes_)
        dif_list = self.Error_check(input_img=y,output_img=predict_list)
        dif_array = np.array(dif_list)
        p_error_rate = np.sum(dif_array) / np.size(dif_array)
        return p_error_rate

    """ Obliczanie wartości teoretycznych (pobudzenia całkowitego) """
    def net_input(self, x):
        return np.dot(x, self.w_[0:])

    """ Obliczanie wartości teoretycznych (pobudzenia całkowitego) dla wyznaczonej klasy """
    def net_input_part(self, x, w_):
        return np.dot(x, w_[0:])

    """ Obliczanie logistycznej, sigmoidalnej funkcji aktywacji """
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    """ Predykacja nowych etykiet """
    def predict(self, X):
        return np.where(self.activation(self.net_input(x=X)) >= 0.5, 1, 0)

    """ Predykacja nowych etykiet dla obrazu prawdopodobieństw"""
    def predict_image(self,probe_images_,image_classes):
        image_predict = None
        for i in range(probe_images_.shape[0]):
            image_predict_axis1 = None
            for j in range(probe_images_.shape[1]):
                position = np.squeeze(np.argwhere(probe_images_[i,j,:] == max(probe_images_[i,j,:])))
                if np.size(position) > 1:
                    position = random.sample(list(position),k=1)
                    position = int(position[0])
                if image_predict_axis1 is not None:
                    image_predict_axis1 = np.append(image_predict_axis1,
                                                    np.atleast_2d(np.array(image_classes[position])),axis=1)
                else:
                    image_predict_axis1 = np.atleast_2d(np.array(image_classes[position]))
            if image_predict is not None:
                image_predict = np.append(image_predict, image_predict_axis1, axis=0)
            else:
                image_predict = image_predict_axis1
        print('Prediction image: DONE ')
        return image_predict

    """ Obliczenie prawdopodobieństw dla każdej z klas """
    def probabilities(self,x):
        if self.multi:
            z_sum, prob = [],[]
            for i in range(self.W_.shape[0]-1):
                z_sum.append(np.exp(np.clip(self.net_input_part(x=x,w_=self.W_[i,:]), -250, 250)))
            z_sum = 1 + np.sum(z_sum)
            prob.append(1 / z_sum)
            for i in range(self.W_.shape[0]-1):
                prob.append(np.exp(np.clip(self.net_input_part(x=x,w_=self.W_[i,:]), -250, 250)) / z_sum)
            return prob

    """ Wynik wpasowania """
    def Output(self, x):
        Result = self.activation(self.net_input(x=x))
        self.Result = self.Reshaping_back(Result,input_image_shape_0=self.image_import_shape[0],
                                          input_image_shape_1=self.image_import_shape[1])
        return self

    """ Funkcja tasowania danych """
    def shuffle_(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    """ Cross-validation 
    30% of testing data vs 70% of learning data """
    def Cross_validation(self,X,Y,classes):
        if self.c_valid:
            Error_matrix = []
            for i in range(self.CV_iter):
                X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,
                                                                    random_state=np.random.randint(100),
                                                                    shuffle=True,stratify=Y)
                Y_train_OVR, classes__ = self.OvR(Y=Y_train,main_class=(-1))
                X_train_std = self.Standard_training_new(parameters=X_train)
                self.fit_all_class(X=X_train_std,Y_OvR=Y_train_OVR)
                X_test_std = self.Standard_test_new(parameters=X_test)
                Error_iter = self.prediction_error_rate(x=X_test_std,classes_=classes,y=Y_test)
                Error_matrix.append(Error_iter)
            Error_matrix = np.array(np.squeeze(Error_matrix))
            Error_cv = (np.sum(Error_matrix)/self.CV_iter)*100
            CV_iter = self.CV_iter
            return Error_cv, Error_matrix, CV_iter

    """ Correlation check
    Sprawdzenie powiązania pomiędzy parametrami"""
    def Correlation_check(self,X):
        if self.correlation:
            R_coef = []
            for i in range(1,X.shape[1]):
                R_coef_i= ['bias']
                for j in range(1,X.shape[1]):
                    R_coef_i.append(np.corrcoef(X[1:,i],X[1:,j])[0,1])
                R_coef.append(np.squeeze(R_coef_i))
            R_coef = np.vstack(R_coef)
            return R_coef

    """ Różnicowy obraz błędów """
    def Error_check(self,input_img,output_img):
        Differential_image = input_img - output_img
        Differential_image[Differential_image != 0] = 1
        return Differential_image

    """ Update danych uczących o dane treningowe"""
    def Update_training(self,predicted_test_image,no_data,bound_mask):
        old_w = self.W_
        image_ovr_test, classes = self.Prepare_Y_test_to_fit(predicted_test_image,no_data=no_data,bound_mask=bound_mask)
        y_ovr = np.append(self.image_ovr,image_ovr_test,axis=1)
        x = np.append(self.training_masks_stack,self.test_masks_stack,axis=2)
        self.fit_all_class(Y_OvR=y_ovr,X=x)
        update_w = self.W_
        return old_w, update_w

    """----------------------------------------------------------------------------------------"""
    """---------------------------------- POST-PROCESSING -------------------------------------"""
    """----------------------------------------------------------------------------------------"""

    """ Wpływ wyznaczonych artefaktów na pobliską okolicę, 
    poprzez promień wpływu oraz funkcję liniową"""
    def Art_impact_calc(self,grid_NS,grid_EW,Area,max_impact_px,i,j):
        D_matrix = np.sqrt(((grid_NS - i) ** 2) + ((grid_EW - j) ** 2))
        Mix_area = Area * D_matrix
        Mix_area[Mix_area > -1] = -9999
        if np.max(Mix_area) == -9999:
            Impact = 0
        else:
            position = np.squeeze(np.argwhere(Mix_area == np.max(Mix_area)))
            if np.size(position)>2:
                position_n = position[0]
            else:
                position_n = position
            if D_matrix[position_n[0], position_n[1]] > max_impact_px:
                Impact = 0
            else:
                Impact = (max_impact_px - D_matrix[position_n[0], position_n[1]]) / max_impact_px
        return np.array(Impact)

    def Art_impact(self,image_,probe_images_):
        if self.art_impact:
            axis_0, axis_1 = image_.shape[0],image_.shape[1]
            grid = np.indices((axis_0,axis_1))
            grid_NS = grid[0]
            grid_EW = grid[1]
            Impact_image = []
            for i in range(axis_0):
                Impact_image_axis1 = []
                if i <= self.max_impact_px:
                    for j in range(axis_1):
                        if j <= self.max_impact_px:
                            Impact_area = image_[0:(i + self.max_impact_px),0:(j + self.max_impact_px)]
                            grid_impact_NS = grid_NS[0:(i + self.max_impact_px),0:(j + self.max_impact_px)]
                            grid_impact_EW = grid_EW[0:(i + self.max_impact_px),0:(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,
                                                          Impact_area,self.max_impact_px,i,j)
                            if j == 0:
                                Impact_image_axis1 = np.atleast_2d(Impact)
                            else:
                                Impact_image_axis1 = np.append(Impact_image_axis1,np.atleast_2d(Impact),axis=1)
                        elif (axis_1 - j) <= self.max_impact_px:
                            Impact_area = image_[0:(i + self.max_impact_px),(j - self.max_impact_px):]
                            grid_impact_NS = grid_NS[0:(i + self.max_impact_px),(j - self.max_impact_px):]
                            grid_impact_EW = grid_EW[0:(i + self.max_impact_px),(j - self.max_impact_px):]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                        else:
                            Impact_area = image_[0:(i + self.max_impact_px),
                                          (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_NS = grid_NS[0:(i + self.max_impact_px),
                                             (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_EW = grid_EW[0:(i + self.max_impact_px),
                                             (j - self.max_impact_px):(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                    if i == 0:
                        Impact_image = Impact_image_axis1
                    else:
                        Impact_image = np.append(Impact_image, Impact_image_axis1, axis=0)
                elif (axis_0 - i) <= self.max_impact_px:
                    for j in range(axis_1):
                        if j <= self.max_impact_px:
                            Impact_area = image_[(i - self.max_impact_px):,0:(j + self.max_impact_px)]
                            grid_impact_NS = grid_NS[(i - self.max_impact_px):,0:(j + self.max_impact_px)]
                            grid_impact_EW = grid_EW[(i - self.max_impact_px):,0:(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area
                                                          ,self.max_impact_px,i,j)
                            if j == 0:
                                Impact_image_axis1 = np.atleast_2d(Impact)
                            else:
                                Impact_image_axis1 = np.append(Impact_image_axis1,np.atleast_2d(Impact),axis=1)
                        elif (axis_1 - j) <= self.max_impact_px:
                            Impact_area = image_[(i - self.max_impact_px):,(j - self.max_impact_px):]
                            grid_impact_NS = grid_NS[(i - self.max_impact_px):,(j - self.max_impact_px):]
                            grid_impact_EW = grid_EW[(i - self.max_impact_px):,(j - self.max_impact_px):]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                        else:
                            Impact_area = image_[(i - self.max_impact_px):,
                                          (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_NS = grid_NS[(i - self.max_impact_px):,
                                             (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_EW = grid_EW[(i - self.max_impact_px):,
                                             (j - self.max_impact_px):(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                    Impact_image = np.append(Impact_image, Impact_image_axis1, axis=0)
                else:
                    for j in range(axis_1):
                        if j <= self.max_impact_px:
                            Impact_area = image_[(i - self.max_impact_px):(i + self.max_impact_px),
                                          0:(j + self.max_impact_px)]
                            grid_impact_NS = grid_NS[(i - self.max_impact_px):(i + self.max_impact_px),
                                             0:(j + self.max_impact_px)]
                            grid_impact_EW = grid_EW[(i - self.max_impact_px):(i + self.max_impact_px),
                                             0:(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            if j == 0:
                                Impact_image_axis1 = np.atleast_2d(Impact)
                            else:
                                Impact_image_axis1 = np.append(Impact_image_axis1,np.atleast_2d(Impact),axis=1)
                        elif (axis_1 - j) <= self.max_impact_px:
                            Impact_area = image_[(i - self.max_impact_px):(i + self.max_impact_px),
                                          (j - self.max_impact_px):]
                            grid_impact_NS = grid_NS[(i - self.max_impact_px):(i + self.max_impact_px),
                                             (j - self.max_impact_px):]
                            grid_impact_EW = grid_EW[(i - self.max_impact_px):(i + self.max_impact_px),
                                             (j - self.max_impact_px):]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                        else:
                            Impact_area = \
                                image_[(i - self.max_impact_px):(i + self.max_impact_px),
                                (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_NS = \
                                grid_NS[(i - self.max_impact_px):(i + self.max_impact_px),
                                (j - self.max_impact_px):(j + self.max_impact_px)]
                            grid_impact_EW = \
                                grid_EW[(i - self.max_impact_px):(i + self.max_impact_px),
                                (j - self.max_impact_px):(j + self.max_impact_px)]
                            Impact = self.Art_impact_calc(grid_impact_NS,grid_impact_EW,Impact_area,
                                                          self.max_impact_px,i,j)
                            Impact_image_axis1 = np.append(Impact_image_axis1, np.atleast_2d(Impact), axis=1)
                    Impact_image = np.append(Impact_image, Impact_image_axis1, axis=0)
            Probe_images_art_impact = np.copy(probe_images_)
            Probe_images_art_impact[:, :, 0] = probe_images_[:, :, 0] + Impact_image
            print('Artifacts impact definition: DONE ')
            return Probe_images_art_impact

    def Art_impact_fast(self,image_,probe_images_):
        max_impact_px_mask = self.max_impact_px * 2
        image_art_only = np.copy(image_)
        image_art_only[image_art_only > -1] = 0
        s_image_a_o = image_art_only[self.max_impact_px:-self.max_impact_px, self.max_impact_px:-self.max_impact_px]
        ij_image = np.ones(s_image_a_o.shape)
        ij_image_one_c_all = np.atleast_3d(ij_image)
        for i in range(max_impact_px_mask):
            for j in range(max_impact_px_mask):
                i_chan = s_image_a_o - image_art_only[i:-(self.max_impact_px * 2 - i),j:-(self.max_impact_px * 2 - j)]
                i_chan[i_chan < 0] = 0
                ij_image_one = ij_image * np.sqrt(((i - self.max_impact_px) ** 2) + ((j - self.max_impact_px) ** 2))
                ij_image_one_c = i_chan * ij_image_one
                ij_image_one_c_all = np.append(ij_image_one_c_all, np.atleast_3d(ij_image_one_c), axis=2)
        ij_image_one_c_all = ij_image_one_c_all[:,:,1:]
        ij_image_one_c_all[ij_image_one_c_all > self.max_impact_px] = 0
        ij_image_one_c_all[ij_image_one_c_all == 0] = 999
        image_dist = np.min(ij_image_one_c_all,axis=2)
        image_dist[image_dist == 999] = 0
        image_art_imp = ((self.max_impact_px - image_dist) / self.max_impact_px)
        image_art_imp[image_art_imp < 0] = 0
        probe_images_cut = np.copy(probe_images_[self.max_impact_px:-self.max_impact_px,
                                   self.max_impact_px:-self.max_impact_px,:])
        probe_images_cut[:, :, 0] = probe_images_cut[:, :, 0] + image_art_imp * probe_images_cut[:, :, 0]
        self.image_cutter = self.image_cutter + self.max_impact_px
        print('Artifacts impact definition: DONE ')
        return probe_images_cut

    """ Usunięcie obszarów poza faktycznym pomiarem
    Usunięcie poprzez wskazanie zerowego odchylenia standardowego oraz rozszenienie wpływu artefaktów
    lub ręczne opisanie punktów skrajnych: lewy górny, prawy górnt, prawy dolny i lewy dolny"""
    def No_data(self,image_,image_classes,probe_images_,no_data_type,corners_utm,hemisphare,training,
                speed):
        if self.no_data_conversion:
            if no_data_type == 'self':
                if image_.shape[0] < self.image_NS.shape[0] or image_.shape[1] < self.image_NS.shape[1]:
                    self.image_NS = self.Image_cutter_(image_to_cut=self.image_NS, smaller_image=image_)
                    self.image_EW = self.Image_cutter_(image_to_cut=self.image_EW, smaller_image=image_)
                left_upper, right_upper, right_lower, left_lower = corners_utm
                left_a = (left_upper[0] - left_lower[0]) / (left_upper[1] - left_lower[1])
                left_b = left_upper[0] - (left_upper[1] * left_a)
                upper_a = (left_upper[0] - right_upper[0]) / (left_upper[1] - right_upper[1])
                upper_b = left_upper[0] - (left_upper[1] * upper_a)
                right_a = (right_upper[0] - right_lower[0]) / (right_upper[1] - right_lower[1])
                right_b = right_upper[0] - (right_upper[1] * right_a)
                lower_a = (left_lower[0] - right_lower[0]) / (left_lower[1] - right_lower[1])
                lower_b = left_lower[0] - (left_lower[1] * lower_a)
                left_bound = self.image_EW * left_a + left_b
                upper_bound = self.image_EW * upper_a + upper_b
                right_bound = self.image_EW * right_a + right_b
                lower_bound = self.image_EW * lower_a + lower_b
                bound_mask = np.zeros(image_.shape)
                if hemisphare == 'norht_right':
                    bound_mask[self.image_NS < left_bound] = 1
                    bound_mask[self.image_NS > upper_bound] = 1
                    bound_mask[self.image_NS > right_bound] = 1
                    bound_mask[self.image_NS < lower_bound] = 1
                elif hemisphare == 'norht_left':
                    bound_mask[self.image_NS > left_bound] = 1
                    bound_mask[self.image_NS > upper_bound] = 1
                    bound_mask[self.image_NS < right_bound] = 1
                    bound_mask[self.image_NS < lower_bound] = 1
                elif hemisphare == 'south_left':
                    bound_mask[self.image_NS > left_bound] = 1
                    bound_mask[self.image_NS < upper_bound] = 1
                    bound_mask[self.image_NS < right_bound] = 1
                    bound_mask[self.image_NS > lower_bound] = 1
                elif hemisphare == 'south_right':
                    bound_mask[self.image_NS < left_bound] = 1
                    bound_mask[self.image_NS < upper_bound] = 1
                    bound_mask[self.image_NS > right_bound] = 1
                    bound_mask[self.image_NS > lower_bound] = 1
                Probe_images_No_data_Fin = np.copy(probe_images_)
                Probe_images_No_data_Fin[:, :, 0] = probe_images_[:, :, 0] + bound_mask
            else:
                self.trend_parameter = False
                self.semi_var_parameter = False
                self.deformation_parameter = False
                self.standard_dev = True
                if training:
                    std_image = self.Standard_deviation_(masks_stack=self.training_masks_stack[0])
                else:
                    std_image = self.Standard_deviation_(masks_stack=self.test_masks_stack[0])
                bound_mask = np.where(std_image <= 1e-8,1,0)
                Probe_images_No_data = np.copy(probe_images_)
                Probe_images_No_data[:,:,0] = probe_images_[:,:,0] + bound_mask
                Predicted_image_std = self.predict_image(probe_images_=Probe_images_No_data, image_classes=image_classes)
                self.art_impact = True
                Probe_images_No_data_Fin = probe_images_[:, :, 0]
                if speed == 'fast':
                    Probe_images_No_data_Fin = self.Art_impact_fast(image_=Predicted_image_std,
                                                                    probe_images_=Probe_images_No_data)
                elif speed == 'slow':
                    Probe_images_No_data_Fin = self.Art_impact(image_=Predicted_image_std,
                                                               probe_images_=Probe_images_No_data)
            print('No date definition: DONE ')
            return Probe_images_No_data_Fin, bound_mask

    """ Usunięcie outliers dla profili (np. histogramu) """
    def Outliers(self,profile_,x_axis,window_,max_deviation,m):
        profile_range = self.Axis(image_=profile_,m=m,mask_size=window_)
        clean_profile = []
        for i in profile_range:
            window_actual = profile_[i - int(window_/2):i + int(window_/2)]
            if np.size(x_axis) == 1:
                range_actual = range(i - int(window_/2),i + int(window_/2))
            else:
                range_actual = x_axis[i - int(window_/2):i + int(window_/2)]
            if max_deviation == 0:
                max_deviation = np.std(window_actual)
            a = np.array([np.ones(window_).T, range_actual.T])
            l_ = np.array(window_actual.T)
            x, sig, v_i = self.Least_square_(a=a,l_=l_)
            if np.size(x_axis) == 1:
                y_t = x[0] + (i * x[1])
            else:
                y_t = x[0] + (x_axis[i] * x[1])
            if np.absolute(v_i) < max_deviation:
                clean_profile.append(profile_[i])
            else:
                clean_profile.append(y_t)
        return np.array(clean_profile)

    """ Wygładzanie obrazu przez profilowanie"""
    def Profile_smoothing(self,image_,profile_mode,window_,max_deviation):
        cleaned_image = []
        if profile_mode == 'V':
            for i in range(image_.shape[0]):
                cleaned_profile = self.Outliers(profile_=image_[i,:], x_axis=0, window_=window_,
                                                max_deviation=max_deviation,m=0)
                cleaned_image.append(np.array(cleaned_profile))
        elif profile_mode == 'H':
            for i in range(image_.shape[1]):
                cleaned_profile = self.Outliers(profile_=image_[:, i], x_axis=0, window_=window_,
                                                max_deviation=max_deviation,m=1)
                cleaned_image.append(np.array(cleaned_profile))
        return cleaned_image

    """ Niwelacja skoków wartości na obszarze zdjęcia
     
     Wprowadzenie:
     - wymóg wcześniejszej predykcji klasy artefaktów w celu niwelacji wprowadzenia błędnych danych do rozwiązania,
     - technika --fast-- określa rozwiązanie dla obszaru okrojonego o wielkość maski, natomiast technika --slow-- dla 
       całości obszaru zdjęcia predykcji,
     - gęstość określana jako obszar brany po uwagę w celu wyszukania maxima lokalnego,
     - ziarno określa stopień zaokrąglenia miejsc po przecinku do wyznaczenie wielkości populacji,
     - stopień minimum populacji okreslany jako wymagany zakres populacji na maksima 
     - wielkość maski określa obszar na zdjęcia, służacy do wyznaczenie średniej sprawdzanej w procesie liczenia skoku
     
     Rozwiązanie:
     Wyznaczenie lokalnych maksima na wygładzonym histogramie wartości zaokroąglonych. 
     Maksima służą do określenia barier pomiędzy nimi, która stanowią zakres grup dla któych wyznaczana jest 
     wartość liczbowa skoku. Skok definiuje się jako średnia z grupy pomiędzy danymi barierami.
     
     Następnie dla każdego piksela określana jest maska, w celu sprawdzenie do której z grup dany piksel należy.
     Niwelacja skoku wyznaczana jest poprzez różnicę wartości piksela do wartości skoku z określonej grupy.
     """
    def Removing_jumps_in_mask(self,px_value,image_mask,predicted_image_mask,barriers,jumps):
        image_reshaped = self.Reshaping(image_=image_mask,no_data=False,bound_mask=0)
        image_predicted_reshaped = self.Reshaping(image_=predicted_image_mask,no_data=False,bound_mask=0)
        image_reshaped_no_art = []
        for i in range(image_reshaped.shape[0]):
            if image_predicted_reshaped[i] != -1:
                image_reshaped_no_art.append(image_reshaped[i])
        pixel_no_jump = px_value
        if np.size(image_reshaped_no_art) > 0:
            mean_from_mask = np.mean(image_reshaped_no_art)
            for i in range(barriers.shape[0]):
                if mean_from_mask <= barriers[i]:
                    pixel_no_jump = px_value - jumps[i]
                    break
        return pixel_no_jump

    def Removing_jumps_slow(self,image_,image_predicted):
        if self.jump_remove:
            if np.size(image_) > np.size(image_predicted):
                image_c_r = self.Image_cutter_(image_to_cut=image_, smaller_image=image_predicted)
                image_ = image_c_r
            barriers, jumps, population, bins_n = self.Removing_jumps_histogram(image_=image_,
                                                                                image_predicted=image_predicted,
                                                                                density_=self.rm_density_,
                                                                                seed_=self.rm_seed_,
                                                                                min_population_factor=
                                                                                self.rm_min_population_factor)
            axis_0, axis_1 = image_.shape[0], image_.shape[1]
            max_px = int(self.rm_mask_size / 2 + 0.5)
            image_no_jump = []
            for i in range(axis_0):
                Nojump_image_axis1 = []
                if i <= max_px:
                    for j in range(axis_1):
                        if j <= max_px:
                            image_area = image_[0:(i + max_px),0:(j + max_px)]
                            mask_area = image_predicted[0:(i + max_px), 0:(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            if j == 0:
                                Nojump_image_axis1 = np.atleast_2d(pixel_no_jump)
                            else:
                                Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),
                                                               axis=1)
                        elif (axis_1 - j) <= max_px:
                            image_area = image_[0:(i + max_px),(j - max_px):]
                            mask_area = image_predicted[0:(i + max_px),(j - max_px):]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                        else:
                            image_area = image_[0:(i + max_px),(j - max_px):(j + max_px)]
                            mask_area = image_predicted[0:(i + max_px), (j - max_px):(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                    if i == 0:
                        image_no_jump = Nojump_image_axis1
                    else:
                        image_no_jump = np.append(image_no_jump, Nojump_image_axis1, axis=0)
                elif (axis_0 - i) <= max_px:
                    for j in range(axis_1):
                        if j <= max_px:
                            image_area = image_[(i - max_px):,0:(j + max_px)]
                            mask_area = image_predicted[(i - max_px):, 0:(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            if j == 0:
                                Nojump_image_axis1 = np.atleast_2d(pixel_no_jump)
                            else:
                                Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                        elif (axis_1 - j) <= max_px:
                            image_area = image_[(i - max_px):,(j - max_px):]
                            mask_area = image_predicted[(i - max_px):, (j - max_px):]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                        else:
                            image_area = image_[(i - max_px):,(j - max_px):(j + max_px)]
                            mask_area = image_predicted[(i - max_px):, (j - max_px):(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                    image_no_jump = np.append(image_no_jump, Nojump_image_axis1, axis=0)
                else:
                    for j in range(axis_1):
                        if j <= max_px:
                            image_area = image_[(i - max_px):(i + max_px),0:(j + max_px)]
                            mask_area = image_predicted[(i - max_px):(i + max_px), 0:(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            if j == 0:
                                Nojump_image_axis1 = np.atleast_2d(pixel_no_jump)
                            else:
                                Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                        elif (axis_1 - j) <= max_px:
                            image_area = image_[(i - max_px):(i + max_px),(j - max_px):]
                            mask_area = image_predicted[(i - max_px):(i + max_px), (j - max_px):]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                        else:
                            image_area = image_[(i - max_px):(i + max_px),(j - max_px):(j + max_px)]
                            mask_area = image_predicted[(i - max_px):(i + max_px), (j - max_px):(j + max_px)]
                            pixel_no_jump = self.Removing_jumps_in_mask(px_value=image_[i,j],image_mask=image_area,
                                                                        predicted_image_mask=mask_area,
                                                                        barriers=barriers,jumps=jumps)
                            Nojump_image_axis1 = np.append(Nojump_image_axis1,np.atleast_2d(pixel_no_jump),axis=1)
                    image_no_jump = np.append(image_no_jump, Nojump_image_axis1, axis=0)
            image_no_jump = np.array(image_no_jump)
            return image_no_jump, jumps, barriers, population, bins_n

    def Removing_jumps_histogram(self, image_, image_predicted, density_, seed_, min_population_factor):
        self.reshaping_technique = True
        if np.size(image_) > np.size(image_predicted):
            image_c_r = self.Image_cutter_(image_to_cut=image_, smaller_image=image_predicted)
            image_ = image_c_r
        image_reshaped = self.Reshaping(image_=image_,no_data=False,bound_mask=0)
        image_predicted_reshaped = self.Reshaping(image_=image_predicted,no_data=False,bound_mask=0)
        image_reshaped_no_art = []
        for i in range(image_reshaped.shape[0]):
            if image_predicted_reshaped[i] != -1:
                image_reshaped_no_art.append(image_reshaped[i])
        image_reshaped_no_art = image_reshaped_no_art
        image_reshaped_no_art_around = np.around(image_reshaped_no_art,seed_)
        bins = np.unique(image_reshaped_no_art_around)
        population, bins_n = np.histogram(image_reshaped_no_art_around,bins=bins)
        #population = self.Outliers(profile_=population,x_axis=bins_n,window_=density_,max_deviation=0,m=0)
        maxima = []
        min_population = min_population_factor * np.sum(population)
        for i in self.Axis(population,0,density_):
            if population[i] == max(population[i-int(density_/2):i+int(density_/2)]):
                if population[i] > min_population:
                    maxima.append(bins_n[i])
        maxima = np.array(maxima)
        barriers = maxima[0:-1] + (maxima[1:] - maxima[0:-1]) / 2
        barriers = np.append(barriers,[(max(bins_n) + 1)],axis=0)
        sums = np.zeros(barriers.shape[0])
        counts = np.zeros(barriers.shape[0])
        """ druga możliwość liczenia barier to wpasowanie krzywej pomiedzy maxima i znalezienie minima"""
        image_reshaped_no_art = np.array(image_reshaped_no_art)
        for i in range(image_reshaped_no_art.shape[0]):
            barriers_ = np.squeeze(np.argwhere(image_reshaped_no_art[i] <= barriers))
            if np.size(barriers_) > 1:
                position = np.squeeze(np.argwhere(image_reshaped_no_art[i] <= barriers))[0]
            else:
                position = barriers_
            sums[position] += image_reshaped_no_art[i]
            counts[position] += 1
        jumps = sums / counts
        return barriers, jumps, population, bins_n

    def Removing_jumps_fast(self,image_,image_predicted):
        if self.jump_remove:
            if self.rm_min_population_factor == 0:
                self.rm_min_population_factor = 0.03
            if np.size(image_) > np.size(image_predicted):
                image_c_r = self.Image_cutter_(image_to_cut=image_, smaller_image=image_predicted)
                image_ = image_c_r
            image_shape_0 = np.array(image_).shape[0]
            image_shape_1 = np.array(image_).shape[1]
            barriers, jumps, population, bins_n = self.Removing_jumps_histogram(image_=image_,
                                                                                image_predicted=image_predicted,
                                                                                density_=self.rm_density_,
                                                                                seed_=self.rm_seed_,
                                                                                min_population_factor=
                                                                                self.rm_min_population_factor)
            self.multi_mask = False
            image_masks_stack = np.array(self.Mask_stacking_opt(image_=image_, masks=self.rm_mask_size, no_data=False,
                                                                bound_mask=0))[0]
            image_predicted_masks_stack = np.array(self.Mask_stacking_opt(image_=image_predicted, masks=self.rm_mask_size,
                                                                          no_data=False, bound_mask=0))[0]
            image_masks_stack_px = image_masks_stack[int(self.rm_mask_size/2),int(self.rm_mask_size/2),:]
            art_mask_count = np.count_nonzero(image_predicted_masks_stack == -1, axis=(0,1))
            image_masks_stack_no_art = np.copy(image_masks_stack)
            image_masks_stack_no_art[image_predicted_masks_stack == -1] = 0
            image_masks_stack_sum = np.sum(image_masks_stack_no_art, axis=(0,1))
            epsilon = 1e-8
            image_masks_stack_mean = image_masks_stack_sum / ((self.rm_mask_size ** 2) - art_mask_count + epsilon)
            clean_jumps_px = np.copy(image_masks_stack_mean)
            image_masks_stack_clean = np.copy(image_masks_stack_px)
            for i in range(barriers.shape[0]):
                clean_jumps_px[clean_jumps_px < barriers[i]] = i + 999
            clean_jumps_px = clean_jumps_px - 999
            for i in range(barriers.shape[0]):
                image_masks_stack_clean[clean_jumps_px == i] += - jumps[i]
            image_no_jump = self.Reshaping_back(image_masks_stack_clean,input_image_shape_0=image_shape_0,
                                                input_image_shape_1=image_shape_1)
            image_no_jump = np.array(image_no_jump)
            print('Jumps removing: DONE')
            return image_no_jump, jumps, barriers, population, bins_n

    def Histogram_check(self, barriers, jumps, population, bins_n):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(bins_n[0:-1], population, alpha=1, width=0.001)
        ax.set_xlabel('Value [m]')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram without artifacts')
        print(barriers[0:-1])
        for xc in barriers[0:-1]:
            plt.axvline(x=xc, label='line at x = {}'.format(xc), c='black')
        for xc in jumps:
            plt.axvline(x=xc, label='line at x = {}'.format(xc), c='green')
        plt.show()

    """----------------------------------------------------------------------------------------"""
    """---------------------------------- GEO-LOKALIZACJA -------------------------------------"""
    """----------------------------------------------------------------------------------------"""

    """ Projekcja współrzędnych z WGS do UTM"""
    def WGS_UTM(self,lat_new,lon_new,utm_zone):
        # Datum constants
        a = 6378137  # equatorial radius
        b = 6356752.3142452  # polar radius
        one_f = 298.257223563  # inverse flattening
        f = 1 / one_f  # flattening
        mR = (a * b) ** 0.5  # mean Radius
        e = np.sqrt(1 - ((b / a) ** 2))  # eccentricity
        e2 = e * e / (1 - e * e)
        n = (a - b) / (a + b)
        # Meridional Arc Constants
        A0 = a * (1 - n + (5 * n * n / 4) * (1 - n) + (81 * (n ** 4) / 64) * (1 - n))
        B0 = (3 * a * n / 2) * (1 - n - (7 * n * n / 8) * (1 - n) + 55 * (n ** 4) / 64)
        C0 = (15 * a * n * n / 16) * (1 - n + (3 * n * n / 4) * (1 - n))
        D0 = (35 * a * (n ** 3) / 48) * (1 - n + 11 * n * n / 16)
        E0 = (315 * a * (n ** 4) / 51) * (1 - n)
        # Other Constants
        Sin1 = np.pi / (180 * 3600)
        LAT_1r = lat_new * np.pi / 180
        LON_1r = lon_new * np.pi / 180
        L_z1 = (utm_zone * 6) - 183
        L_rest1 = (lon_new - L_z1) * 3600 / 10000
        k01 = np.sqrt((1 - (e2 ** 2) * np.sin(LAT_1r) ** 2))  # scale factor1
        # r_curv_1_1 = a * (1 - e * e) / ((1 - (e * np.sin(LAT_1r)) **2) **(3 / 2))
        r_curv_1_2 = a / ((1 - (e * np.sin(LAT_1r)) ** 2) ** (1 / 2))
        # Calculate Meridional Arc Length
        MAS_1 = A0 * LAT_1r - B0 * np.sin(2 * LAT_1r) + C0 * np.sin(4 * LAT_1r) - D0 * np.sin(6 * LAT_1r) + E0 * np.sin(
            8 * LAT_1r)
        # Coefficients for UTM Coordinates
        K1_1 = MAS_1 * k01
        K1_2 = r_curv_1_2 * np.sin(LAT_1r) * np.cos(LAT_1r) * (Sin1 ** 2) * k01 * 100000000 / 2
        K1_3 = ((Sin1 ** 4 * r_curv_1_2 * np.sin(LAT_1r) * np.cos(LAT_1r) ** 3) / 24) * (
                    5 - np.tan(LAT_1r) ** 2 + (9 * e2) * np.cos(LAT_1r) ** 2 + (4 * e2 ** 2) * np.cos(
                LAT_1r) ** 4) * k01 * 10000000000000000
        K1_4 = r_curv_1_2 * np.cos(LAT_1r) * Sin1 * k01 * 10000
        K1_5 = ((Sin1 * np.cos(LAT_1r)) ** 3) * (r_curv_1_2 / 6) * (
                    1 - np.tan(LAT_1r) ** 2 + e2 * np.cos(LAT_1r) ** 2) * k01 * 1000000000000
        # A1_6 = (((L_rest1 * Sin1) **6) * r_curv_1_2 * np.sin(LAT_1r) * (np.cos(LAT_1r) **5) / 720) *
        # (61 - 58 * np.tan(LAT_1r) **2 + np.tan(LAT_1r) **4 + 270 * e2 * np.cos(LAT_1r) **2 - 330 * 2 *
        # np.sin(LAT_1r) **2) * k01 * (1E+24)
        Raw_N_1 = (K1_1 + K1_2 * L_rest1 ** 2 + K1_3 * L_rest1 ** 4)
        if np.size(Raw_N_1) > 1:
            North1 = []
            for row in Raw_N_1:
                if row[1] < 0:
                    North1.append(row + 10000000)
                else:
                    North1.append(row)
        else:
            if Raw_N_1 < 0:
                North1 = Raw_N_1 + 10000000
            else:
                North1 = Raw_N_1
        East1 = 500000 + (K1_4 * L_rest1 + K1_5 * L_rest1 ** 3)
        return North1, East1

    """ Projekcja współrzędnych z UTM do WGS"""
    def UTM_WGS(self,x_utm,y_utm,hemisphere,utm_zone):
        # Constatns
        k0 = 0.9996  # scale factor
        a = 6378137  # equatorial radius
        b = 6356752.3142452  # polar radius
        e = np.sqrt(1 - ((b / a) ** 2))  # eccentricity
        ei = (1 - (1 - e * e) ** (1 / 2)) / (1 + (1 - e * e) ** (1 / 2))
        eisq = e * e / (1 - e * e)
        C1 = 3 * ei / 2 - 27 * (ei ** 3) / 32
        C2 = 21 * (ei ** 2) / 16 - 55 * (ei ** 4) / 32
        C3 = 151 * (ei ** 3) / 96
        C4 = 1097 * (ei ** 4) / 512
        # Corrected Northing
        if hemisphere == 'N':
            y_utm = y_utm
        else:
            y_utm = 10000000 - y_utm
        # East Prime
        Ep = 500000 - x_utm
        # Arc Length
        AL = y_utm / k0
        mu = AL / (a * (1 - (e ** 2) / 4 - 3 * (e ** 4) / 64 - 5 * (e ** 6) / 256))
        phi = mu + C1 * np.sin(2 * mu) + C2 * np.sin(4 * mu) + C3 * np.sin(6 * mu) + C4 * np.sin(
            8 * mu)  # Footprint Latitude
        c1 = eisq * np.cos(phi) ** 2
        t1 = np.tan(phi) ** 2
        n1 = a / (1 - (e * np.sin(phi)) ** 2) ** (1 / 2)
        r1 = a * (1 - e * e) / (1 - (e * np.sin(phi)) ** 2) ** (3 / 2)
        d = Ep / (n1 * k0)
        fact1 = n1 * np.tan(phi) / r1
        fact2 = d * d / 2
        fact3 = (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * eisq) * (d ** 4) / 24
        fact4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * eisq - 3 * c1 * c1) * (d ** 6) / 720
        Lofact1 = d
        Lofact2 = (1 + 2 * t1 + c1) * (d ** 3) / 6
        Lofact3 = (5 - 2 * c1 + 28 * t1 - 3 * (c1 ** 2) + 8 * eisq + 24 * (t1 ** 2)) * (d ** 5) / 120
        Delta_Long = (Lofact1 - Lofact2 + Lofact3) / np.cos(phi)
        Zone_CM = 6 * utm_zone - 183
        Raw_Lat = 180 * (phi - fact1 * (fact2 + fact3 + fact4)) / np.pi
        if hemisphere == 'N':
            Lat = Raw_Lat
        else:
            Lat = -Raw_Lat
        Lon = Zone_CM - Delta_Long * 180 / np.pi
        return Lon, Lat

    """ Macierze współrzędnych """
    def Coor_matrix(self,image_,yfirst,ystep,xfirst,xstep):
        # y <=> Latitude // x <=> Longitude
        axis_0, axis_1 = image_.shape[0], image_.shape[1]
        grid = np.indices((axis_0, axis_1))
        grid_y = grid[0]
        grid_x = grid[1]
        self.image_NS = yfirst + (grid_y * ystep)
        self.image_EW = xfirst + (grid_x * xstep)
        return self

    """ Resampling obrazu - bilinearna interpolacja"""
    def Resampling_(self,image_,old_res,new_res):
        factor = new_res / old_res
        image_res = sc_nd.zoom(input=image_, zoom=factor, order=1)
        self.image_NS_res = sc_nd.zoom(input=self.image_NS, zoom=factor, order=1)
        self.image_EW_res = sc_nd.zoom(input=self.image_EW, zoom=factor, order=1)
        return image_res

