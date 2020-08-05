"""-------------------------------------------------------------------------------------------------------------"""
"""------------------------------------------------ GUI --------------------------------------------------------"""
"""-------------------------------------------------------------------------------------------------------------"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from datetime import datetime
import pickle
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QFileDialog, QTextEdit, QGridLayout, QWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QInputDialog, QPushButton
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QApplication
from PyQt5.QtGui import QPixmap, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from DDD_app import DDD
"""-------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------- Database ----------------------------------------------------"""
"""-------------------------------------------------------------------------------------------------------------"""
class Current_database(object):
    def __init__(self,set_=[],DDD_run=[],input_image=[],fileName=[],stack=[],parametrization=[],system=[],origin=[],
                 resolut=[]):
        self.set_ = set_
        self.DDD_run = DDD_run
        self.input_image = input_image
        self.fileName = fileName
        self.stack = stack
        self.parametrization = parametrization
        self.system = system
        self.origin = origin
        self.resolut = resolut
        self.current_base = [self.DDD_run, self.input_image, self.fileName, self.stack, self.parametrization,
                             self.system, self.origin, self.resolut]

class Current_directory(object):
    def __init__(self,set_=[],DDD_run=[],input_image=[],fileName=[],stack=[],parametrization=[],system=[],origin=[],
                 resolut=[]):
        self.set_ = set_
        self.DDD_run = DDD_run
        self.input_image = input_image
        self.fileName = fileName
        self.stack = stack
        self.parametrization = parametrization
        self.system = system
        self.origin = origin
        self.resolut = resolut
        self.current_base = [self.DDD_run, self.input_image, self.fileName, self.stack, self.parametrization,
                             self.system, self.origin, self.resolut]

class Training_database(object):
    def __init__(self,set_=[],x_set_=[],x_data_init=[],x_data=[],x_data_std=[],y_set_=[],y_data_init=[],y_data=[],
                 y_data_ovr=[],standardization_parameters=[],training_weights=[],classes=[],type_of=[],cost_sums=[],
                 probe_images=[],predicted_image=[],decorel_seq=[],error_rate=[]):
        self.set_ = set_
        self.x_set_ = x_set_
        self.x_data_init = x_data_init
        self.x_data = x_data
        self.x_data_std = x_data_std
        self.y_set_ = y_set_
        self.y_data_init = y_data_init
        self.y_data = y_data
        self.y_data_ovr = y_data_ovr
        self.standardization_parameters = standardization_parameters
        self.classes = classes
        self.type_of = type_of
        self.training_weights = training_weights
        self.cost_sums = cost_sums
        self.probe_images = probe_images
        self.predicted_image = predicted_image
        self.decorel_seq = decorel_seq
        self.error_rate = error_rate

class Training_current_directory(object):
    def __init__(self,set_=[],x_set_=[],x_data_init=[],x_data=[],x_data_std=[],y_set_=[],y_data_init=[],y_data=[],
                 y_data_ovr=[],standardization_parameters=[],training_weights=[],classes=[],type_of=[],cost_sums=[],
                 probe_images=[],predicted_image=[],decorel_seq=[],error_rate=[]):
        self.set_ = set_
        self.x_set_ = x_set_
        self.x_data_init = x_data_init
        self.x_data = x_data
        self.x_data_std = x_data_std
        self.y_set_ = y_set_
        self.y_data_init = y_data_init
        self.y_data = y_data
        self.y_data_ovr = y_data_ovr
        self.standardization_parameters = standardization_parameters
        self.training_weights = training_weights
        self.classes = classes
        self.type_of = type_of
        self.cost_sums = cost_sums
        self.probe_images = probe_images
        self.predicted_image = predicted_image
        self.decorel_seq = decorel_seq
        self.error_rate =error_rate

"""-------------------------------------------------------------------------------------------------------------"""
"""----------------------------------------------- Main menu ---------------------------------------------------"""

class GUI_panel(QMainWindow):

    def __init__(self):
        super().__init__()
        self.current_database = Current_database(set_=[], DDD_run=[], input_image=[], fileName=[], stack=[],
                                                 parametrization=[], system=[], origin=[], resolut=[])
        self.current_directory = Current_directory(set_=[], DDD_run=[], input_image=[], fileName=[], stack=[],
                                                   parametrization=[], system=[], origin=[], resolut=[])
        self.training_database = Training_database(set_=[], x_set_=[], x_data_init=[], x_data=[], x_data_std=[],
                                                   y_set_=[], y_data=[], y_data_ovr=[], y_data_init=[], classes=[],
                                                   standardization_parameters=[], training_weights=[], type_of=[],
                                                   cost_sums=[], probe_images=[], predicted_image=[], decorel_seq=[],
                                                   error_rate=[])
        self.training_current_directory = Training_current_directory(set_=[], x_set_=[], x_data_init=[], x_data=[],
                                                                     x_data_std=[], y_set_=[], y_data=[],
                                                                     y_data_init=[], y_data_ovr=[],
                                                                     standardization_parameters=[], training_weights=[],
                                                                     classes=[], type_of=[], cost_sums=[],
                                                                     probe_images=[], predicted_image=[],
                                                                     decorel_seq=[], error_rate=[])
        self.initMENU()

    def initMENU(self):

        self.setCentralWidget(START_widgets())
        menubar = self.menuBar()

        fileMenu = menubar.addMenu('&File')
        newInit = QAction('Initialization', self)
        newInit.triggered.connect(self.Initialization)
        impMenu = QMenu('Import', self)
        impIMG = QAction('Image', self)
        impIMG.triggered.connect(self.Import_)
        impSET = QAction('Import image settings', self)
        impSET.triggered.connect(self.Import_settings)
        impDataSET = QAction('Import image data set', self)
        impDataSET.triggered.connect(self.Import_image_set)
        impWDataSET = QAction('Import model data set', self)
        impWDataSET.triggered.connect(self.Import_model_set)
        impMenu.addAction(impIMG)
        impMenu.addAction(impSET)
        impMenu.addAction(impDataSET)
        impMenu.addAction(impWDataSET)

        expMenu = QMenu('Export', self)
        expIMG = QAction('Image', self)
        expIMG.triggered.connect(self.Export_image_set)
        expMDL = QAction('Model', self)
        expMDL.triggered.connect(self.Export_model_set)
        expMenu.addAction(expIMG)
        expMenu.addAction(expMDL)

        SelectMenu = QMenu('Select', self)
        SelectWeights = QAction('Training data set', self)
        SelectWeights.triggered.connect(self.Set_training_data)
        SelectImage = QAction('Image', self)
        SelectImage.triggered.connect(self.Database_Image_search)
        SelectMenu.addAction(SelectImage)
        SelectMenu.addAction(SelectWeights)

        Settings = QAction('Settings', self)
        Settings.triggered.connect(self.All_run_settings)

        fileMenu.addAction(newInit)
        fileMenu.addMenu(impMenu)
        fileMenu.addMenu(expMenu)
        fileMenu.addMenu(SelectMenu)
        fileMenu.addAction(Settings)

        ImageMenu = menubar.addMenu('Image')

        newView = QAction('View image', self)
        newView.triggered.connect(self.Image_view)
        newM_stc = QAction('Mask stacking', self)
        newM_stc.triggered.connect(self.Mask_stacking_)
        newM_set_stc = QAction('Mask stacking settings', self)
        newM_set_stc.triggered.connect(self.Mask_stacking_settings_)
        newR_resh = QAction('Reshaping', self)
        newR_resh.triggered.connect(self.Reshaping_)
        newA_imp_menu = QMenu('Artifacts impact', self)
        newA_imp_fast = QAction('Fast method (Default)', self)
        newA_imp_fast.triggered.connect(self.Art_impact_function_fast)
        newA_imp_slow = QAction('Full method', self)
        newA_imp_slow.triggered.connect(self.Art_impact_function_slow)
        newA_imp_set = QAction('Artifacts impact settings', self)
        newA_imp_set.triggered.connect(self.Art_impact_function_settings)
        newA_imp_menu.addAction(newA_imp_fast)
        newA_imp_menu.addAction(newA_imp_slow)
        newA_imp_menu.addAction(newA_imp_set)

        JumpMenu = QMenu('Remove jumps', self)
        JumpFast = QAction('Fast method (Default)', self)
        JumpFast.triggered.connect(self.Removing_jumps_fast_function)
        JumpSlow = QAction('Full method', self)
        JumpSlow.triggered.connect(self.Removing_jumps_slow_function)
        JumpHist = QAction('Histogram with jumps', self)
        JumpHist.triggered.connect(self.Histogram_with_jumps_function)
        JumpSet = QAction('Settings', self)
        JumpSet.triggered.connect(self.Removing_jums_function_settings)
        JumpMenu.addAction(JumpFast)
        JumpMenu.addAction(JumpSlow)
        JumpMenu.addAction(JumpHist)
        JumpMenu.addAction(JumpSet)

        ImageMenu.addAction(newView)
        ImageMenu.addAction(newM_stc)
        ImageMenu.addAction(newM_set_stc)
        ImageMenu.addAction(newR_resh)
        ImageMenu.addMenu(newA_imp_menu)
        ImageMenu.addMenu(JumpMenu)

        DataMenu = menubar.addMenu('Data Analysis')

        data_act = QAction('Set data', self)
        data_act.triggered.connect(self.Set_training_data)

        XlMenu = QMenu('X learning data', self)

        XlPMeanu = QMenu('Parametrization', self)
        XlPDefault = QAction('Default', self)
        XlPDefault.triggered.connect(self.Par_Default_)
        XlPFull = QAction('Full', self)
        XlPFull.triggered.connect(self.Par_Full_)
        XlPCust = QAction('Custom', self)
        XlPCust.triggered.connect(self.Par_Custom_)
        XlPSet = QAction('Parametrization settings', self)
        XlPSet.triggered.connect(self.Par_settings)

        XlPMeanu.addAction(XlPDefault)
        XlPMeanu.addAction(XlPFull)
        XlPMeanu.addAction(XlPCust)
        XlPMeanu.addAction(XlPSet)

        XlStand = QAction('Standarization', self)
        XlStand.triggered.connect(self.Standardization_training)

        XlMenu.addMenu(XlPMeanu)
        XlMenu.addAction(XlStand)

        XlDecorel = QAction('Decorelation', self)
        XlDecorel.triggered.connect(self.Decorelation_training)
        XlMenu.addAction(XlDecorel)

        XlDecorel_set = QAction('Decorelation maximum rate', self)
        XlDecorel_set.triggered.connect(self.Decorelation_training_set)
        XlMenu.addAction(XlDecorel_set)

        XtMenu = QMenu('X testing data', self)

        XtPMeanu = QMenu('Parametrization', self)
        XtPDefault = QAction('Default', self)
        XtPDefault.triggered.connect(self.Par_Default_)
        XtPFull = QAction('Full', self)
        XtPFull.triggered.connect(self.Par_Full_)
        XtPCust = QAction('Custom', self)
        XtPCust.triggered.connect(self.Par_Custom_)
        XtPSet = QAction('Parametrization settings', self)
        XtPSet.triggered.connect(self.Par_settings)

        XtPMeanu.addAction(XtPDefault)
        XtPMeanu.addAction(XtPFull)
        XtPMeanu.addAction(XtPCust)
        XtPMeanu.addAction(XtPSet)

        XtStand = QAction('Standarization', self)
        XtStand.triggered.connect(self.Standardization_testing)

        XtMenu.addMenu(XtPMeanu)
        XtMenu.addAction(XtStand)

        XtDecorel = QAction('Decorelation', self)
        XtDecorel.triggered.connect(self.Decorelation_testing)
        XtMenu.addAction(XtDecorel)

        YMenu = QMenu('Y data', self)
        YOvR = QAction('One vs Rest', self)
        YOvR.triggered.connect(self.OvR_technique_)
        YMenu.addAction(YOvR)

        MergeM = QAction('Merge models', self)
        MergeM.triggered.connect(self.Update_model)

        DataMenu.addAction(data_act)
        DataMenu.addMenu(XlMenu)
        DataMenu.addMenu(XtMenu)
        DataMenu.addMenu(YMenu)
        DataMenu.addAction(MergeM)

        LearnMenu = menubar.addMenu('Machine Learning')
        Lpmulti = QAction('Learning process', self)
        Lpmulti.triggered.connect(self.Fitting_process_multiclass)

        ProbeIMG = QMenu('Probabilistic images', self)
        ProbeIMGcreate = QAction('Create', self)
        ProbeIMGcreate.triggered.connect(self.Probe_images_init)
        ProbeIMGview = QAction('View', self)
        ProbeIMGview.triggered.connect(self.Probe_img_view)
        ProbeIMG.addAction(ProbeIMGcreate)
        ProbeIMG.addAction(ProbeIMGview)

        PredictIMG = QAction('Prediction image', self)
        PredictIMG.triggered.connect(self.Predict_image_init)

        StatMenu = QMenu('Statisctics of learining', self)
        Statcost = QAction('Sum of logistic cost', self)
        Statcost.triggered.connect(self.Cost_function_view)
        Staterr = QAction('Error rate', self)
        Staterr.triggered.connect(self.Error_rate_view)

        StatMenu.addAction(Statcost)
        StatMenu.addAction(Staterr)

        LearnMenu.addAction(Lpmulti)
        LearnMenu.addMenu(ProbeIMG)
        LearnMenu.addAction(PredictIMG)
        LearnMenu.addMenu(StatMenu)

        TestMenu = menubar.addMenu('Testing')

        CV = QAction('Cross validation', self)
        CV.triggered.connect(self.Estimate_CROSS_VALIDATION_rate_)
        CC = QAction('Correlation check', self)
        CC.triggered.connect(self.Correlation_check_function)

        TestMenu.addAction(CV)
        TestMenu.addAction(CC)

        GEOMenu = menubar.addMenu('GEO-reference')
        M_cor = QAction('Localization', self)
        M_cor.triggered.connect(self.Localization_function)
        Trans = QMenu('Transformation', self)
        WtUTrans = QAction('WGS to UTM', self)
        WtUTrans.triggered.connect(self.Transformation_WGS_to_UTM)
        UtWTrans = QAction('UTM to WGS', self)
        UtWTrans.triggered.connect(self.Transformation_UTM_to_WGS)
        Trans.addAction(WtUTrans)
        Trans.addAction(UtWTrans)

        GEOMenu.addAction(M_cor)
        GEOMenu.addMenu(Trans)

        HelpMenu = menubar.addMenu('Help')

        AboutF = QAction('About functions', self)
        AboutSoft = QAction('About software', self)
        AboutF.triggered.connect(self.Functions_)
        AboutSoft.triggered.connect(self.Software_)

        HelpMenu.addAction(AboutF)
        HelpMenu.addAction(AboutSoft)

        self.setGeometry(50, 50, 600, 500)
        self.setWindowTitle('DInSAR Deformation Detection')
        self.show()

    def Initialization(self):
        self.DDD_run = DDD()
        self.current_directory.set_ = np.size(self.current_database.set_) + 1
        self.current_directory.DDD_run = self.DDD_run
        self.current_directory.input_image = np.array([[],[]])
        self.current_directory.fileName = 'None'
        self.current_directory.stack = (np.array([[], []]))
        self.current_directory.parametrization = (np.array([[], []]))
        self.current_directory.origin = [0]
        self.current_directory.system = [0]
        self.current_directory.resolut = [0]
        self.Append_image_database()
        self.setCentralWidget(INIT_widgets(message=("Initialization process is set with number:\n" +
                                                    str(self.current_directory.set_))))

    def Append_image_database(self):
        self.current_database.set_.append(self.current_directory.set_)
        self.current_database.DDD_run.append(self.DDD_run)
        self.current_database.input_image.append(self.current_directory.input_image)
        self.current_database.fileName.append(self.current_directory.fileName)
        self.current_database.stack.append(self.current_directory.stack)
        self.current_database.parametrization.append(self.current_directory.parametrization)
        self.current_database.origin.append(self.current_directory.origin)
        self.current_database.system.append(self.current_directory.system)
        self.current_database.resolut.append(self.current_directory.resolut)

    def Database_Image_search(self):
        self.setCentralWidget(SELECT_widgets(database_in=self.current_database,directory_in=self.current_directory))

    def Update_database_partition(self,i):
        self.current_database.set_[i] = self.current_directory.set_
        self.current_database.input_image[i] = self.current_directory.input_image
        self.current_database.fileName[i] = self.current_directory.fileName
        self.current_database.stack[i] = self.current_directory.stack
        self.current_database.parametrization[i] = self.current_directory.parametrization
        self.current_database.origin[i] = self.current_directory.origin
        self.current_database.system[i] = self.current_directory.system
        self.current_database.resolut[i] = self.current_directory.resolut

    def Update_database(self):
        if len(self.current_database.DDD_run) > 1:
            for i in range(len(self.current_database.DDD_run)):
                if self.current_database.DDD_run[i] == self.current_directory.DDD_run:
                    self.Update_database_partition(i=i)
        else:
            if self.current_database.DDD_run[0] == self.current_directory.DDD_run:
                self.Update_database_partition(i=0)

    def Update_training_database_partition(self,i):
        self.training_database.x_set_[i] = self.training_current_directory.x_set_
        self.training_database.x_data_init[i] = self.training_current_directory.x_data_init
        self.training_database.x_data[i] = self.training_current_directory.x_data
        self.training_database.x_data_std[i] = self.training_current_directory.x_data_std
        self.training_database.y_set_[i] = self.training_current_directory.y_set_
        self.training_database.y_data_init[i] = self.training_current_directory.y_data_init
        self.training_database.y_data[i] = self.training_current_directory.y_data
        self.training_database.y_data_ovr[i] = self.training_current_directory.y_data_ovr
        self.training_database.standardization_parameters[i] = \
            self.training_current_directory.standardization_parameters
        self.training_database.training_weights[i] = self.training_current_directory.training_weights
        self.training_database.classes[i] = self.training_current_directory.classes
        self.training_database.type_of[i] = self.training_current_directory.type_of
        self.training_database.cost_sums[i] = self.training_current_directory.cost_sums
        self.training_database.probe_images[i] = self.training_current_directory.probe_images
        self.training_database.predicted_image[i] = self.training_current_directory.predicted_image
        self.training_database.decorel_seq[i] = self.training_current_directory.decorel_seq
        self.training_database.error_rate[i] = self.training_current_directory.error_rate

    def Update_training_database(self):
        if len(self.training_database.set_) > 1:
            for i in range(len(self.training_database.set_)):
                if self.training_database.set_[i] == self.training_current_directory.set_:
                    self.Update_training_database_partition(i=i)
        else:
            if self.training_database.set_[0] == self.training_current_directory.set_:
                self.Update_training_database_partition(i=0)

    def Append_training_database(self):
        self.training_database.set_.append(self.training_current_directory.set_)
        self.training_database.x_set_.append(self.training_current_directory.x_set_)
        self.training_database.x_data.append(self.training_current_directory.x_data)
        self.training_database.x_data_init.append(self.training_current_directory.x_data_init)
        self.training_database.x_data_std.append(self.training_current_directory.x_data_std)
        self.training_database.y_set_.append(self.training_current_directory.y_set_)
        self.training_database.y_data_init.append(self.training_current_directory.y_data_init)
        self.training_database.y_data.append(self.training_current_directory.y_data)
        self.training_database.y_data_ovr.append(self.training_current_directory.y_data_ovr)
        self.training_database.standardization_parameters.append(
            self.training_current_directory.standardization_parameters)
        self.training_database.training_weights.append(self.training_current_directory.training_weights)
        self.training_database.classes.append(self.training_current_directory.classes)
        self.training_database.type_of.append(self.training_current_directory.type_of)
        self.training_database.cost_sums.append(self.training_current_directory.cost_sums)
        self.training_database.probe_images.append(self.training_current_directory.probe_images)
        self.training_database.predicted_image.append(self.training_current_directory.predicted_image)
        self.training_database.decorel_seq.append(self.training_current_directory.decorel_seq)
        self.training_database.error_rate.append(self.training_current_directory.error_rate)

    def Import_settings(self):
        if self.current_directory.DDD_run == []:
            self.setCentralWidget(INIT_widgets(message="No new initialization!\n"
                                                       "File->Initialization"))
        else:
            self.setCentralWidget(IMPORT_SET_widgets(application=self.current_directory.DDD_run))

    def Import_(self):
        if self.current_directory.DDD_run == []:
            self.setCentralWidget(INIT_widgets(message="No new initialization!\n"
                                                       "File->Initialization"))
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.fileName, ok_select = QFileDialog.getOpenFileName(self, 'Open file', '',
                                                                   'All Files (*);;Python Files (*.py)', options=options)
            if ok_select:
                if self.fileName[-3:] == self.current_directory.DDD_run.data_type:
                    self.input_image, origo, res, system_ = self.current_directory.DDD_run.import_data(
                        directory=self.fileName, non_value=0)
                    self.setCentralWidget(IMPORT_widgets(data=self.input_image,filename=self.fileName))
                    self.current_directory.input_image = self.input_image
                    self.current_directory.fileName = str(self.fileName)
                    self.current_directory.origin = origo
                    self.current_directory.system = system_
                    self.current_directory.resolut = res
                    self.Update_database()
                elif self.DDD_run.data_type == 'image' and self.fileName[-3:] != 'txt' and self.fileName[-3:] != 'csv':
                    self.input_image, origo, res, system_ = self.current_directory.DDD_run.import_data(
                        directory=self.fileName, non_value=0)
                    self.setCentralWidget(IMPORT_widgets(data=self.input_image,filename=self.fileName))
                    self.current_directory.input_image = self.input_image
                    self.current_directory.fileName = str(self.fileName)
                    self.current_directory.origin = origo
                    self.current_directory.system = system_
                    self.current_directory.resolut = res
                    self.Update_database()
                elif self.DDD_run.data_type == 'GeoTIFF' and self.fileName[-3:] != 'txt' and self.fileName[-3:] != 'csv':
                    self.input_image, origo, res, system_ = self.current_directory.DDD_run.import_data(
                        directory=self.fileName, non_value=0)
                    self.setCentralWidget(IMPORT_widgets(data=self.input_image,filename=self.fileName))
                    self.current_directory.input_image = self.input_image
                    self.current_directory.fileName = str(self.fileName)
                    self.current_directory.origin = origo
                    self.current_directory.system = system_
                    self.current_directory.resolut = res
                    self.Update_database()
                else:
                    self.setCentralWidget(INIT_widgets(message="Wrong import data type!\n"
                                                               "Change import data type in:\n "
                                                               "File->Import settings->Select data type"))

    def Image_view(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Mask_stacking_settings_(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.setCentralWidget(MASK_STACKING_SET_widgets(application=self.current_directory.DDD_run))

    def Mask_stacking_(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.current_directory.DDD_run.parameters = True
            self.image_input_masks_stack = self.current_directory.DDD_run.Mask_stacking_opt(
                image_=self.current_directory.input_image, masks=self.current_directory.DDD_run.masks,
                no_data=False, bound_mask=0)
            self.current_directory.stack = self.image_input_masks_stack
            self.Update_database()
            self.setCentralWidget(INIT_widgets(message="Mask stacking process is done!"))

    def Reshaping_(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.image_input_reshaped = self.current_directory.DDD_run.Reshaping(
                image_=self.current_directory.input_image,no_data=False, bound_mask=0)
            self.current_directory.stack = self.image_input_reshaped
            self.Update_database()
            self.setCentralWidget(INIT_widgets(message="Reshaping process is done!"))

    def Par_settings(self):
        if np.size(self.current_directory.stack) == 0:
            self.setCentralWidget(INIT_widgets(message="No image stack in initialization!\n"
                                                       "Image->Mask stacking"))
        else:
            self.setCentralWidget(PARAMETRIZATION_SET_widgets(application=self.current_directory.DDD_run))

    def Par_final_process(self,par_image):
        self.training_current_directory.x_data = par_image
        self.training_current_directory.x_data_init = self.current_directory.DDD_run
        self.training_current_directory.x_set_ = self.current_directory.set_
        self.training_current_directory.set_ = np.size(self.training_database.set_) + 1
        self.training_current_directory.x_data_std = np.array([[],[]])
        self.training_current_directory.y_set_ = [0]
        self.training_current_directory.y_data_init = [0]
        self.training_current_directory.y_data = np.array([[],[]])
        self.training_current_directory.y_data_ovr = np.array([[],[]])
        self.training_current_directory.standardization_parameters = np.array([[],[]])
        self.training_current_directory.training_weights = np.array([[],[]])
        self.training_current_directory.classes = np.array([])
        self.training_current_directory.type_of = 0
        self.training_current_directory.cost_sums = np.array([[], []])
        self.training_current_directory.probe_images = np.array([[], []])
        self.training_current_directory.predicted_image = np.array([[], []])
        self.training_current_directory.decorel_seq = np.array([[], []])
        self.training_current_directory.error_rate = [0]
        self.Append_training_database()
        self.Update_database()
        self.Update_training_database()
        self.setCentralWidget(INIT_widgets(message="Parametrization process is done!"))

    def Par_Default_(self):
        if np.size(self.current_directory.stack) == 0:
            self.setCentralWidget(INIT_widgets(message="No image stack in initialization!\n"
                                                       "Image->Mask stacking"))
        else:
            self.current_directory.DDD_run.parameters = True
            self.current_directory.DDD_run.trend_parameter = True
            self.current_directory.DDD_run.var_parameter = True
            self.current_directory.DDD_run.standard_dev = True
            self.current_directory.DDD_run.deformation_parameter = True
            self.current_directory.DDD_run.semi_var_parameter = True
            self.current_directory.DDD_run.gaussian_par = False
            self.image_parameters = self.current_directory.DDD_run.Parametrization(masks_stack=
                                                                                   self.current_directory.stack,
                                                                                   masks_size=
                                                                                   self.current_directory.DDD_run.masks,
                                                                                   depths=
                                                                                   self.current_directory.DDD_run.depths)
            self.current_directory.parametrization = self.image_parameters
            self.Par_final_process(self.image_parameters)

    def Par_Full_(self):
        if np.size(self.current_directory.stack) == 0:
            self.setCentralWidget(INIT_widgets(message="No image stack in initialization!\n"
                                                       "Image->Mask stacking"))
        else:
            self.current_directory.DDD_run.parameters = True
            self.current_directory.DDD_run.trend_parameter = True
            self.current_directory.DDD_run.var_parameter = True
            self.current_directory.DDD_run.standard_dev = True
            self.current_directory.DDD_run.deformation_parameter = True
            self.current_directory.DDD_run.semi_var_parameter = True
            self.current_directory.DDD_run.gaussian_par = True
            self.image_parameters = self.current_directory.DDD_run.Parametrization(masks_stack=
                                                                                   self.current_directory.stack,
                                                                                   masks_size=
                                                                                   self.current_directory.DDD_run.masks,
                                                                                   depths=
                                                                                   self.current_directory.DDD_run.depths)
            self.current_directory.parametrization = self.image_parameters
            self.Par_final_process(self.image_parameters)

    def Par_Custom_(self):
        if np.size(self.current_directory.stack) == 0:
            self.setCentralWidget(INIT_widgets(message="No image stack in initialization!\n"
                                                       "Image->Mask stacking"))
        else:
            self.current_directory.DDD_run.parameters = True
            self.image_parameters = self.current_directory.DDD_run.Parametrization(masks_stack=
                                                                                   self.current_directory.stack,
                                                                                   masks_size=
                                                                                   self.current_directory.DDD_run.masks,
                                                                                   depths=
                                                                                   self.current_directory.DDD_run.depths)
            self.current_directory.parametrization = self.image_parameters
            self.Par_final_process(self.image_parameters)

    def Set_training_data(self):
        self.setCentralWidget(Set_training_data_widget(database_in=self.training_database,
                                                       directory_in=self.training_current_directory,
                                                       database_in_all=self.current_database))

    def Standardization_training(self):
        if np.size(self.training_current_directory.x_data) == 0:
            self.setCentralWidget(INIT_widgets(message="No X data in active set!\n"
                                               "Data Analysis->X learning data->Parametrization"))
        else:
            if self.training_current_directory.type_of == 0:
                self.setCentralWidget(INIT_widgets(message="No X data in active set!\n"
                                                           "Data Analysis->X learning data->Parametrization"))
            elif self.training_current_directory.type_of == 1:
                self.setCentralWidget(INIT_widgets(message="Current set is not learning data!\n"
                                                           "Data Analysis->Select master learning set"
                                                           "/Select other learning set"))
            elif self.training_current_directory.type_of == 3 or 2:
                self.training_current_directory.x_data_std = \
                    self.training_current_directory.x_data_init.Standard_training_new(
                    parameters=self.training_current_directory.x_data)
                self.training_current_directory.standardization_parameters = [
                    self.training_current_directory.x_data_init.mean_of_p,
                    self.training_current_directory.x_data_init.std_of_p]
                self.Update_training_database()
                self.setCentralWidget(INIT_widgets(message="Standardization process is done!"))

    def Standardization_testing(self):
        if np.size(self.training_current_directory.x_data) == 0:
            self.setCentralWidget(INIT_widgets(message="No X data in active set!\n"
                                               "Data Analysis->X learning data->Parametrization"))
        else:
            if self.training_current_directory.type_of == 1:
                for i in range(len(self.training_database.set_)):
                    if self.training_database.type_of[i] == 3:
                        mean_of_p_temp = self.training_database.x_data_init[i].mean_of_p
                        std_of_p_temp = self.training_database.x_data_init[i].std_of_p
                        self.training_current_directory.x_data_init.mean_of_p = mean_of_p_temp
                        self.training_current_directory.x_data_init.std_of_p = std_of_p_temp
                        self.training_current_directory.x_data_std = \
                        self.training_current_directory.x_data_init.Standard_test_new(
                            parameters=self.training_current_directory.x_data)
                        self.training_current_directory.standardization_parameters = [mean_of_p_temp, std_of_p_temp]
                        self.Update_training_database()
                        self.setCentralWidget(INIT_widgets(message="Standardization process is done!"))
            else:
                self.setCentralWidget(INIT_widgets(message="Current set is not testing data!\n"
                                                           "Data Analysis->Select testing set"))

    def OvR_technique_(self):
        if np.size(self.training_current_directory.y_data) == 0:
            self.setCentralWidget(INIT_widgets(message="No Y data in active set!\n"
                                               "Data Analysis->Set data->Select Y data set"))
        else:
            self.training_current_directory.y_data_ovr, self.training_current_directory.classes = \
                self.training_current_directory.y_data_init.OvR(self.training_current_directory.y_data,main_class=(-1))
            self.Update_training_database()
            self.setCentralWidget(INIT_widgets(message="One vs Rest process is done!"))

    def Fitting_process_multiclass(self):
        if np.size(self.training_current_directory.y_data_ovr) == 0:
            self.setCentralWidget(INIT_widgets(message="No OvR technique data in active set!\n"
                                               "Data Analysis->Y data->One vs Rest"))
        elif np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            check = int(max(self.training_current_directory.x_data_init.masks)/2) + 1
            lr_y_check = self.training_current_directory.y_data_init.right -\
                         self.training_current_directory.y_data_init.left
            lr_x_check = self.training_current_directory.x_data_init.right -\
                          self.training_current_directory.x_data_init.left
            ul_y_check = self.training_current_directory.y_data_init.lower -\
                         self.training_current_directory.y_data_init.upper
            ul_x_check = self.training_current_directory.x_data_init.lower -\
                          self.training_current_directory.x_data_init.upper
            if self.training_current_directory.x_data.shape[0] != self.training_current_directory.y_data.shape[0]:
                self.setCentralWidget(INIT_widgets(message="Wrong size of the X data or Y data sets!\n"
                                                           "The image Y must be smaller than the image X by an integer "
                                                           "value from half of the maximum mask on each side!\n"
                                                           "Integer value of the maximum mask: " + str(check) + "\n"
                                                           "X data size: " + str(lr_x_check)+"x"+str(ul_x_check) +"\n"
                                                           "Y data size: " + str(lr_y_check)+"x"+str(ul_y_check) +"\n"))
            else:
                self.training_current_directory.x_data_init.fit_all_class(X=self.training_current_directory.x_data_std,
                                                                          Y_OvR=self.training_current_directory.y_data_ovr)
                self.training_current_directory.training_weights = self.training_current_directory.x_data_init.W_
                self.training_current_directory.cost_sums = self.training_current_directory.x_data_init.Cost_
                self.training_current_directory.error_rate = self.Estimate_error_rate()
                self.Update_training_database()
                classes = [self.training_current_directory.classes[-1]]
                for i in range(len(self.training_current_directory.classes) - 1):
                    classes.append(self.training_current_directory.classes[i])
                self.setCentralWidget(Cost_function_widgets(cost_sum=self.training_current_directory.cost_sums,
                                                            classes=classes))

    def Probe_images_init(self):
        if np.size(self.training_current_directory.training_weights) == 0:
            self.setCentralWidget(INIT_widgets(message="No weights in current database!\n"
                                               "Machine Learning->Learning process->..."))
        elif np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            self.training_current_directory.probe_images = self.training_current_directory.x_data_init.\
                probabilities_imaging(x=self.training_current_directory.x_data_std,
                class_number=len(self.training_current_directory.classes),
                input_image_shape_0=self.training_current_directory.x_data_init.image_import_shape[0],
                input_image_shape_1=self.training_current_directory.x_data_init.image_import_shape[1])
            self.Update_training_database()
            self.setCentralWidget(INIT_widgets(message="Probabilistic images are created!"))

    def Predict_image_init(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        else:
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            self.training_current_directory.predicted_image = self.training_current_directory.x_data_init.\
                predict_image(probe_images_=self.training_current_directory.probe_images,
                              image_classes=classes)
            self.Update_training_database()
            self.Initialization()
            self.current_directory.input_image = self.training_current_directory.predicted_image
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Cost_function_view(self):
        if np.size(self.training_current_directory.cost_sums) == 0:
            self.setCentralWidget(INIT_widgets(message="No learning process done in current database!\n"
                                               "Machine Learning->Learning process->..."))
        else:
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            self.setCentralWidget(Cost_function_widgets(cost_sum=self.training_current_directory.cost_sums,
                                                        classes=classes))

    def Probe_img_view(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        else:
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            self.setCentralWidget(Probe_images_widgets(probe_img=self.training_current_directory.probe_images,
                                                       classes=classes))

    def Estimate_error_rate(self):
        if np.size(self.training_current_directory.cost_sums) == 0:
            self.setCentralWidget(INIT_widgets(message="No learning process done in current database!\n"
                                               "Machine Learning->Learning process->..."))
        else:
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            Error_rate = self.training_current_directory.x_data_init.prediction_error_rate(
                x=self.training_current_directory.x_data_std,classes_=classes,y=self.training_current_directory.y_data)
            return Error_rate

    def Error_rate_view(self):
        if np.size(self.training_current_directory.predicted_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No predicted image in current database!\n"
                                               "Machine Learning->Predict image"))
        else:
            Differential_image = self.training_current_directory.x_data_init.Error_check(
                input_img=self.current_database.input_image[(self.training_current_directory.y_set_ - 1)][:,0:-1],
                output_img=self.training_current_directory.predicted_image)
            self.setCentralWidget(ERROR_check_widgets(differential_image=Differential_image))

    def Decorelation_training(self):
        if np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            if self.training_current_directory.type_of == 0:
                self.setCentralWidget(INIT_widgets(message="No X data in active set!\n"
                                                           "Data Analysis->X learning data->Parametrization"))
            elif self.training_current_directory.type_of == 1:
                self.setCentralWidget(INIT_widgets(message="Current set is not learning data!\n"
                                                           "Data Analysis->Select master learning set"
                                                           "/Select other learning set"))
            elif self.training_current_directory.type_of == 3 or 2:
                self.par_decorel, self.pos_seq = self.training_current_directory.x_data_init.decorelation(
                    X=self.training_current_directory.x_data_std)
                self.training_current_directory.x_data = self.par_decorel
                self.training_current_directory.x_data_std = np.array([[], []])
                self.training_current_directory.decorel_seq = self.pos_seq
                self.Update_training_database()
                self.setCentralWidget(INIT_widgets(message="Decorelation process is done"))

    def Decorelation_testing(self):
        if np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            if self.training_current_directory.type_of == 1:
                for i in range(len(self.training_database.set_)):
                    if self.training_database.type_of[i] == 3:
                        pos_seq = self.training_database.decorel_seq[i]
                        self.par_decorel = self.training_current_directory.x_data_init.decorelation_test(
                            X=self.training_current_directory.x_data_std,seq=pos_seq)
                        self.training_current_directory.x_data = self.par_decorel
                        self.training_current_directory.x_data_std = np.array([[], []])
                        self.Update_training_database()
                        self.setCentralWidget(INIT_widgets(message="Decorelation process is done"))
                    else:
                        self.setCentralWidget(INIT_widgets(message="Current set is not testing data!\n"
                                                                   "Data Analysis->Select testing set"))

    def Decorelation_training_set(self):
        if np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            self.setCentralWidget(DECORELATION_SET_widgets(application=self.training_current_directory.x_data_init))

    def Export_image_set(self):
        self.setCentralWidget(EXPORT_image_set_widgets(database_in=self.current_database, directory_in=self.current_directory))

    def Export_model_set(self):
        self.setCentralWidget(Export_model_set_widget(database_in=self.training_database,
                                                      directory_in=self.training_current_directory,
                                                      database_in_all=self.current_database))

    def Import_model_set(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, ok_select = QFileDialog.getOpenFileName(self, 'Open file', '',
                                                               'All Files (*);;Python Files (*.py)', options=options)
        if ok_select:
            if self.fileName[-20:] != "Import_model_set.txt":
                self.setCentralWidget(INIT_widgets(message="Wrong file imported!\n"
                                                           "Open_file with name:\n "
                                                           "Import_model_set.txt"))
            else:
                import_file = open(self.fileName, 'r')
                Lines = import_file.readlines()
                X_data_file = open(Lines[0][0:-1], 'rb')
                X_data_data = pickle.load(X_data_file)
                X_data_file.close()
                X_data_DDD_run_file = open(Lines[1][0:-1], 'rb')
                X_data_DDD_run_data = pickle.load(X_data_DDD_run_file)
                X_data_DDD_run_file.close()
                X_data_std_file = open(Lines[2][0:-1], 'rb')
                X_data_std_data = pickle.load(X_data_std_file)
                X_data_std_file.close()
                Y_data_file = open(Lines[3][0:-1], 'rb')
                Y_data_data = pickle.load(Y_data_file)
                Y_data_file.close()
                Y_data_DDD_run_file = open(Lines[4][0:-1], 'rb')
                Y_data_DDD_run_data = pickle.load(Y_data_DDD_run_file)
                Y_data_DDD_run_file.close()
                Y_data_OvR_file = open(Lines[5][0:-1], 'rb')
                Y_data_OvR_data = pickle.load(Y_data_OvR_file)
                Y_data_OvR_file.close()
                STD_Par_file = open(Lines[6][0:-1], 'rb')
                STD_Par_data = pickle.load(STD_Par_file)
                STD_Par_file.close()
                M_w_file = open(Lines[7][0:-1], 'rb')
                M_w_data = pickle.load(M_w_file)
                M_w_file.close()
                M_c_file = open(Lines[8][0:-1], 'rb')
                M_c_data = pickle.load(M_c_file)
                M_c_file.close()
                CF_file = open(Lines[9][0:-1], 'rb')
                CF_data = pickle.load(CF_file)
                CF_file.close()
                Prob_image_file = open(Lines[10][0:-1], 'rb')
                Prob_image_data = pickle.load(Prob_image_file)
                Prob_image_file.close()
                Pred_image_file = open(Lines[11][0:-1], 'rb')
                Pred_image_data = pickle.load(Pred_image_file)
                Pred_image_file.close()
                Dec_seq_file = open(Lines[12][0:-1], 'rb')
                Dec_seq_data = pickle.load(Dec_seq_file)
                Dec_seq_file.close()
                Error_file = open(Lines[13][0:-1], 'rb')
                Error_data = pickle.load(Error_file)
                Error_file.close()
                self.training_current_directory.set_ = np.size(self.training_database.set_) + 1
                self.training_current_directory.x_data = X_data_data
                self.training_current_directory.x_data_init = DDD()
                for name in X_data_DDD_run_data:
                    self.training_current_directory.x_data_init.__dict__[name] = X_data_DDD_run_data[name]
                self.training_current_directory.x_set_ = [0]
                self.training_current_directory.x_data_std = X_data_std_data
                self.training_current_directory.y_set_ = [0]
                self.training_current_directory.y_data = Y_data_data
                self.training_current_directory.y_data_init = DDD()
                for name in Y_data_DDD_run_data:
                    self.training_current_directory.y_data_init.__dict__[name] = Y_data_DDD_run_data[name]
                self.training_current_directory.y_data_ovr = Y_data_OvR_data
                self.training_current_directory.standardization_parameters = STD_Par_data
                self.training_current_directory.training_weights = M_w_data
                self.training_current_directory.classes = M_c_data
                self.training_current_directory.type_of = 0
                self.training_current_directory.cost_sums = CF_data
                self.training_current_directory.probe_images = Prob_image_data
                self.training_current_directory.predicted_image = Pred_image_data
                self.training_current_directory.decorel_seq = Dec_seq_data
                self.training_current_directory.error_rate = Error_data
                self.Append_training_database()
                self.Update_training_database()
                self.setCentralWidget(INIT_widgets(message="Data import is done!\n"))

    def Import_image_set(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, ok_select = QFileDialog.getOpenFileName(self, 'Open file', '',
                                                               'All Files (*);;Python Files (*.py)', options=options)
        if ok_select:
            if self.fileName[-20:] != "Import_image_set.txt":
                self.setCentralWidget(INIT_widgets(message="Wrong file imported!\n"
                                                           "Open_file with name:\n "
                                                           "Import_image_set.txt"))
            else:
                import_file = open(self.fileName, 'r')
                Lines = import_file.readlines()
                DDD_run_file = open(Lines[0][0:-1], 'rb')
                DDD_run_data = pickle.load(DDD_run_file)
                DDD_run_file.close()
                Image_file = open(Lines[1][0:-1], 'rb')
                Image_data = pickle.load(Image_file)
                Image_file.close()
                File_name_file = open(Lines[2][0:-1], 'rb')
                File_name_data = pickle.load(File_name_file)
                File_name_file.close()
                Stack_file = open(Lines[3][0:-1], 'rb')
                Stack_data = pickle.load(Stack_file)
                Stack_file.close()
                Par_file = open(Lines[4][0:-1], 'rb')
                Par_data = pickle.load(Par_file)
                Par_file.close()
                Origo_file = open(Lines[5][0:-1], 'rb')
                Origo_data = pickle.load(Origo_file)
                Origo_file.close()
                System_file = open(Lines[6][0:-1], 'rb')
                System_data = pickle.load(System_file)
                System_file.close()
                Res_file = open(Lines[7][0:-1], 'rb')
                Res_data = pickle.load(Res_file)
                Res_file.close()
                self.current_directory.set_ = np.size(self.current_database.set_) + 1
                self.current_directory.DDD_run = DDD()
                for name in DDD_run_data:
                    self.current_directory.DDD_run.__dict__[name] = DDD_run_data[name]
                self.current_directory.input_image = Image_data
                self.current_directory.fileName = File_name_data
                self.current_directory.stack = Stack_data
                self.current_directory.parametrization = Par_data
                self.current_directory.origin = Origo_data
                self.current_directory.system = System_data
                self.current_directory.resolut = Res_data
                self.Append_image_database()
                self.Update_database()
                self.setCentralWidget(INIT_widgets(message="Data import is done!\n"))

    def Update_model(self):
        self.setCentralWidget(Update_model_set_widget(database_in=self.training_database,
                                                      directory_in=self.training_current_directory,
                                                      database_in_all=self.current_database))

    def Art_impact_function_fast(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            new_probe_images = self.training_current_directory.x_data_init.Art_impact_fast(
                image_=self.current_database.input_image[self.training_current_directory.x_set_],
                probe_images_=self.training_current_directory.probe_images)
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            predicted_image = self.training_current_directory.x_data_init.\
                predict_image(probe_images_=new_probe_images,
                              image_classes=classes)
            old_classes = self.training_current_directory.classes
            self.Par_final_process(par_image=np.array([[],[]]))
            self.training_current_directory.probe_images = new_probe_images
            self.training_current_directory.classes = old_classes
            self.training_current_directory.predicted_image = predicted_image
            self.Update_training_database()
            self.Initialization()
            self.current_directory.input_image = self.training_current_directory.predicted_image
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Art_impact_function_slow(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            new_probe_images = self.training_current_directory.x_data_init.Art_impact(
                image_=self.current_database.input_image[self.training_current_directory.x_set_],
                probe_images_=self.training_current_directory.probe_images)
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            predicted_image = self.training_current_directory.x_data_init.\
                predict_image(probe_images_=new_probe_images,
                              image_classes=classes)
            old_classes = self.training_current_directory.classes
            self.Par_final_process(par_image=np.array([[],[]]))
            self.training_current_directory.probe_images = new_probe_images
            self.training_current_directory.classes = old_classes
            self.training_current_directory.predicted_image = predicted_image
            self.Update_training_database()
            self.Initialization()
            self.current_directory.input_image = self.training_current_directory.predicted_image
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Art_impact_function_settings(self):
        self.setCentralWidget(ART_IMPACT_SET_widgets(application=self.training_current_directory.x_data_init))

    def Art_impact_function_fast(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            new_probe_images = self.training_current_directory.x_data_init.Art_impact_fast(
                image_=self.current_database.input_image[self.training_current_directory.x_set_],
                probe_images_=self.training_current_directory.probe_images)
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            predicted_image = self.training_current_directory.x_data_init.\
                predict_image(probe_images_=new_probe_images,
                              image_classes=classes)
            old_classes = self.training_current_directory.classes
            self.Par_final_process(par_image=np.array([[],[]]))
            self.training_current_directory.probe_images = new_probe_images
            self.training_current_directory.classes = old_classes
            self.training_current_directory.predicted_image = predicted_image
            self.Update_training_database()
            self.Initialization()
            self.current_directory.input_image = self.training_current_directory.predicted_image
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Art_impact_function_slow(self):
        if np.size(self.training_current_directory.probe_images) == 0:
            self.setCentralWidget(INIT_widgets(message="No probabilistic images in current database!\n"
                                               "Machine Learning->Probabilistic images"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            new_probe_images = self.training_current_directory.x_data_init.Art_impact(
                image_=self.current_database.input_image[self.training_current_directory.x_set_],
                probe_images_=self.training_current_directory.probe_images)
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            predicted_image = self.training_current_directory.x_data_init.\
                predict_image(probe_images_=new_probe_images,
                              image_classes=classes)
            old_classes = self.training_current_directory.classes
            self.Par_final_process(par_image=np.array([[],[]]))
            self.training_current_directory.probe_images = new_probe_images
            self.training_current_directory.classes = old_classes
            self.training_current_directory.predicted_image = predicted_image
            self.Update_training_database()
            self.Initialization()
            self.current_directory.input_image = self.training_current_directory.predicted_image
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(IMPORT_widgets(data=self.current_directory.input_image,
                                                 filename=self.current_directory.fileName))

    def Art_impact_function_settings(self):
        if self.training_current_directory.x_data_init == []:
            self.setCentralWidget(INIT_widgets(message="No new initialization in machine learning models!\n"
                                                       "File->Initialization"))
        else:
            self.setCentralWidget(ART_IMPACT_SET_widgets(application=self.training_current_directory.x_data_init))

    def Removing_jumps_fast_function(self):
        if np.size(self.training_current_directory.predicted_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No predicted image in current database!\n"
                                               "Machine Learning->Predict image"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            image_no_jump, jumps, barriers, population, bins_n =\
                self.training_current_directory.x_data_init.Removing_jumps_fast(
                    image_=self.current_database.input_image[(self.training_current_directory.x_set_ - 1)][:,0:-1],
                    image_predicted=self.training_current_directory.predicted_image)
            self.Initialization()
            self.current_directory.input_image = image_no_jump
            self.current_directory.DDD_run.image_import_shape = [image_no_jump.shape[0], image_no_jump.shape[1]]
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(Removing_jumps_widgets(data=self.current_directory.input_image,
                                                         filename=self.current_database.fileName[
                                                             (self.training_current_directory.x_set_ - 1)],jumps=jumps,
                                                         barriers=barriers,population=population,bins_n=bins_n))

    def Removing_jumps_slow_function(self):
        if np.size(self.training_current_directory.predicted_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No predicted image in current database!\n"
                                               "Machine Learning->Predict image"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            image_no_jump, jumps, barriers, population, bins_n =\
                self.training_current_directory.x_data_init.Removing_jumps_slow(
                    image_=self.current_database.input_image[(self.training_current_directory.x_set_ - 1)][:,0:-1],
                    image_predicted=self.training_current_directory.predicted_image)
            self.Initialization()
            self.current_directory.input_image = image_no_jump
            self.current_directory.DDD_run.image_import_shape = [image_no_jump.shape[0], image_no_jump.shape[1]]
            self.current_directory.fileName = 'No directory'
            self.Update_database()
            self.setCentralWidget(Removing_jumps_widgets(data=self.current_directory.input_image,
                                                         filename=self.current_database.fileName[
                                                             (self.training_current_directory.x_set_ - 1)],jumps=jumps,
                                                         barriers=barriers,population=population,bins_n=bins_n))

    def Histogram_with_jumps_function(self):
        if np.size(self.training_current_directory.predicted_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No predicted image in current database!\n"
                                               "Machine Learning->Predict image"))
        elif np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            image_ = self.current_database.input_image[(self.training_current_directory.x_set_ - 1)][:,0:-1]
            image_predicted = self.training_current_directory.predicted_image
            barriers, jumps, population, bins_n = self.training_current_directory.x_data_init.Removing_jumps_histogram(
                    image_=image_, image_predicted=image_predicted,
                    density_=self.training_current_directory.x_data_init.rm_density_,
                    seed_=self.training_current_directory.x_data_init.rm_seed_,
                    min_population_factor=self.training_current_directory.x_data_init.rm_min_population_factor)
            self.setCentralWidget(Removing_jumps_widgets(data=image_,filename=self.current_database.fileName[
                                                             (self.training_current_directory.x_set_ - 1)],jumps=jumps,
                                                         barriers=barriers,population=population,bins_n=bins_n))

    def Removing_jums_function_settings(self):
        if self.training_current_directory.x_data_init == []:
            self.setCentralWidget(INIT_widgets(message="No new initialization in machine learning models!\n"
                                                       "File->Initialization"))
        else:
            self.setCentralWidget(REM_JUMPS_SET_widgets(application=self.training_current_directory.x_data_init))

    def Correlation_check_function(self):
        if np.size(self.training_current_directory.x_data_std) == 0:
            self.setCentralWidget(INIT_widgets(message="No standardized X data in active set!\n"
                                               "Data Analysis->X learning data->Standardization"))
        else:
            R_coef = self.training_current_directory.x_data_init.Correlation_check(
                self.training_current_directory.x_data_std)
            self.setCentralWidget(COREL_widgets(table_=R_coef))

    def Estimate_CROSS_VALIDATION_rate_(self):
        if np.size(self.training_current_directory.cost_sums) == 0:
            self.setCentralWidget(INIT_widgets(message="No learning process done in current database!\n"
                                               "Machine Learning->Learning process->..."))
        else:
            self.Initialization()
            X = self.training_current_directory.x_data
            Y = self.training_current_directory.y_data
            classes = [self.training_current_directory.classes[-1]]
            for i in range(len(self.training_current_directory.classes)-1):
                classes.append(self.training_current_directory.classes[i])
            Error_cv, Error_matrix, CV_iter = self.current_directory.DDD_run.Cross_validation(X=X,Y=Y,classes=classes)
            self.setCentralWidget(CV_widgets(CV_error=Error_cv,Matrix_error=Error_matrix,CV_iter=CV_iter))

    def Localization_function(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.setCentralWidget(
                Localization_widget(database_in=self.current_database,directory_in=self.current_directory))

    def Transformation_UTM_to_WGS(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.setCentralWidget(Transformation_UTM_WGS_widget(
                database_in=self.current_database,directory_in=self.current_directory))

    def Transformation_WGS_to_UTM(self):
        if np.size(self.current_directory.input_image) == 0:
            self.setCentralWidget(INIT_widgets(message="No image in initialization or no new initialization !\n"
                                                       "File->Initialization\n"
                                                       "File->Import->Image"))
        else:
            self.setCentralWidget(Transformation_WGS_UTM_widget(
                database_in=self.current_database,directory_in=self.current_directory))

    def All_run_settings(self):
        if self.current_directory.DDD_run == []:
            self.setCentralWidget(INIT_widgets(message="No new initialization!\n"
                                                       "File->Initialization"))
        else:
            self.setCentralWidget(ALL_SET_widgets(application=self.current_directory.DDD_run))

    def Software_(self):
        self.setCentralWidget(INIT_widgets(
            message="DInSAR Deformation Detection Application 0.0.1\n"
                    "\n"
                    "Published: GNU AGPLv3 License\n"
                    "\n"
                    "Copyright: \xa9 2020, Poland \n"
                    "\n"
                    "Author: Rafal Marciniak\n"
                    "E-mail: rav.marcin.geodezja@gmail.com\n"
                    "\n"
                    "The main purpose of the application is the ability to define the location\n"
                    "of deformation in DInSAR images. It uses machine learning techniques,\n"
                    "in particular the methods of multiclass logistic regression.\n"
                    "In addition, the application allows you to:\n"
                    "- improvement of the regression model and validation\n"
                    "- connecting models and changing model parameters\n"
                    "- eliminating jumps in values on displacements images\n"
                    "- identification of places with incorrect displacement (artifacts)\n"
                    "- estimation of the probability of the predicted classes\n"
                    "- operating in UTM and WGS systems"))

    def Functions_(self):
        self.setCentralWidget(INIT_widgets(
            message="Function descriptions:\n"
                    "\n"
                    "- --Mask stacking--\n"
                    "  Provide a list of matrixes of masks for each pixels from the image\n"
                    "  Requirements:\n"
                    "               - a image with deformation value\n"
                    "  Result:\n"
                    "               - a list matrixes with sizes (mask x n x m)\n"
                    "  Settings:\n"
                    "               - 'Mask size in stacking' - the sizes of masks for stacking, for which the\n"
                    "                 parameters will be estimates\n"
                    "               - 'Limit of pixels in row/column' - the maximum amount of pixels which can be\n"
                    "                 made in a stack. This setting is made to provide posibility of app operation on\n"
                    "                 slow hardwware\n"
                    "\n"
                    "- --Reshaping--\n"
                    "  Reshape class image to a matrix prepared for machine learning and prediction\n"
                    "  Requirements:\n"
                    "               - a class image\n"
                    "  Result:\n"
                    "               - a matrix with (m x n)\n"
                    "\n"
                    "- --Artifacts impact--\n"
                    "  Expand the area of artifacts (wrong value in pixels) with range given in settings\n"
                    "  Requirements:\n"
                    "               - a image with deformation values\n"
                    "               - a probabilistic images\n"
                    "               - a classes\n"
                    "  Result:\n"
                    "               - a new propabilistic images\n"
                    "               - a new predicted image with more artifacts\n"
                    "  Settings:\n"
                    "               - 'Impact of artifacts' - the range where artifacts can have impact on\n"
                    "                 the other pixels\n"
                    "\n"
                    "- --Remove jumps--\n"
                    "  Remove 'jumps' between values on deformation image.\n"
                    "  The method is based on predicted images with known location of artifacts.\n"
                    "  The histogram is created from values, without artifacts pixels and split into clasess\n"
                    "  with limit based on local minimum in histogram, only if the minimum is higher than -minimum\n"
                    "  no of pixels to consider- (see 'Settings').\n"
                    "  Therfore each pixels is assigned to a group on histogram,\n"
                    "  based on the values of pixels in area with range -Mask size (RJ)- (see 'Settings')"
                    "  Requirements:\n"
                    "               - a predicted image\n"
                    "               - a image with deformation values\n"
                    "  Result:\n"
                    "               - a new image with deformation values without jumps\n"
                    "  Settings:\n"
                    "               - 'Mask size (RJ)' - mask size for assiging pixels into a group\n"
                    "               - 'Histogram desity' - the density of bars in histogram in one seed\n"
                    "               - 'Histogram seed' - the precision of rounding\n"
                    "               - 'Minimum no of pixels to consider' - the minimum population to make\n"
                    "                 limit of a group\n"
                    "\n"
                    "- --Parametrization--\n"
                    "  Create parametrs for machine learning process. The type of parameter:"
                    "  - bias - matrix with ones"
                    "  - trend detection - search for linear and curve plane trends in mask with parameters:\n"
                    "                    - estimated coefficients\n"
                    "                    - diference between centered pixel value and theoretical\n"
                    "                    - RMS error of fitting\n"
                    "  - Variance from px - the value of variance respect to distance from the center of mask:\n"
                    "                    - estimated coefficients\n"
                    "                    - RMS error of fitting\n"      
                    "  - Semi-variance - the value of variance semi variance for each distance in the biggest mask:\n"
                    "                    - semi variance\n"
                    "  - Standard deviation - the value of STD in mask\n"
                    "  - Deformation value - the map of pixels, for which value is lower than 'depth' (see Settings)\n"
                    "                    - zero and one matrix\n"
                    "  - Differential Gaussian - show the places with high changes of values (jumps)\n"
                    "                    - values after differential gaussian filter\n"
                    "  - image - real values from given image"
                    "  Requirements:\n"
                    "               - a mask stack\n"
                    "  Result:\n"
                    "               - a matrix paramaters\n"
                    "  Settings:\n"
                    "               - 'Trend detection'\n"
                    "               - 'Variance from px'\n"
                    "               - 'Standard deviation'\n"
                    "               - 'Deformation value'\n"
                    "               - 'Semi-variance'\n"
                    "               - 'Differential Gaussian'\n"
                    "               - 'Depths class to detect' - classes for deformation value parameter\n"
                    "               - 'Subsidence (-1) / Uplift (+1)' - detection of subsidence or uplift deformation\n"
                    "\n"
                    "- --Machine learning process--\n"
                    "  Learning process is based on multi logistic regression function with technique One vs Rest.\n"
                    "  Learning consists in prediction class for each pixel based on the given parameters\n"
                    "  and the weight of those parameters, and therefore weight change at the time of erroneous\n"
                    "  prediction. L2 regularization was used to achieve overfitting."
                    "  Requirements:\n"
                    "               - a parameters matrix\n"
                    "               - a class matrix\n"
                    "               - a classes\n"
                    "  Result:\n"
                    "               - a new predicted image\n"
                    "               - a new propabilistic images\n"
                    "               - a new model with new weights for each paramater\n"
                    "               - a error of prediction\n"
                    "  Settings:\n"
                    "               - 'Number of iterations'\n"
                    "               - 'Seed to random state generator' - first weight are chosen randomly with given\n"
                    "                  seed\n"
                    "               - 'L2 regularization parameter' - the lambda coefficant\n"
                    "\n"
                    "- --Decoraletion--\n"
                    "  Logistic regression can by used if the parameters are not correlate to each other. However in\n"
                    "  in this application the higher number of parameters are better, without considering\n"
                    "  correlation."
                    "  Requirements:\n"
                    "               - a parameters matrix\n"
                    "  Result:\n"
                    "               - a new parameters matrix with maximum correlation factor given from settings\n"
                    "  Settings:\n"
                    "               - 'Maximum rate '\n"
                    "\n"
                    "- --Cross-validation--\n"
                    "  Prediction modeling in logistic regeresion does not allow to fully assess the accuracy of the\n"
                    "  model, due to the imported overtraining of the model.\n"
                    "  Therefore, up to the date of accuracy, cross validation is applied with 70% of the training\n"
                    "  data to 30% of the test data."
                    "  Requirements:\n"
                    "               - a parameters matrix\n"
                    "               - a classes with images of classes\n"
                    "  Result:\n"
                    "               - a validated prediction error\n"
                    "\n"))

"""-------------------------------------------------------------------------------------------------------------"""
"""------------------------------------------------ Widgets ----------------------------------------------------"""
"""-------------------------------------------------------------------------------------------------------------"""

class START_widgets(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        pic = QLabel(self)
        pic.setPixmap(QPixmap("./Icon/Start_page.png"))
        pic.resize(500, 417)
        pic.move(0,0)
        pic.show()

class INIT_widgets(QWidget):

    def __init__(self,message=None):
        super().__init__()
        self.message = message
        self.initUI(self.message)

    def initUI(self,message):
        Status_toolbar = QTextEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 1, 5, 1)
        self.setLayout(grid)
        Status_toolbar.setText(str(message))
        self.show()

class IMPORT_widgets(QWidget):

    def __init__(self,data='None',filename='None'):
        super().__init__()
        self.data = data
        self.filename = filename
        self.initUI(self.data,self.filename)

    def initUI(self,data,filename):
        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 2)

        Data_toolbar = QTextEdit()
        grid.addWidget(Data_toolbar, 4, 2, 5, 2)

        Status_toolbar.setText(str(filename))
        Data_toolbar.setText(('Shape:' + '                              ' + str(data.shape)))
        Data_toolbar.append(('Standard deviation:' + '          ' + str(np.std(data))))
        Data_toolbar.append(('Maximum and minimum:' + '    ' + str(np.max(data)) + '; ' + str(np.min(data))))
        Data_toolbar.append(('Mean value:' + '                     ' + str(np.mean(data))))

        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        im = self.axes.imshow(data[::-1])
        self.figure.colorbar(im)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        grid.addWidget(self.toolbar, 5, 1, 5, 1)
        grid.addWidget(self.canvas, 1, 1, 5, 1)
        if np.min(data) == -1:
            population, bins_n = np.histogram(data)
        else:
            bins_start = np.arange(np.min(data), np.max(data), 0.0001)
            population, bins_n = np.histogram(data,bins=bins_start)
        bins_n = np.array(bins_n)
        if np.size(bins_n) > 0:
            hist_width = np.min(bins_n[0:-1] - bins_n[1:])
            self.figure1 = plt.figure()
            self.axes1 = self.figure1.add_subplot(111)
            self.axes1.bar(bins_n[0:-1], population, alpha=1, width=hist_width)
            self.axes1.set_ylim([0, np.max(population)])
            self.axes1.set_xlabel('Value [m]')
            self.axes1.set_title('Histogram')
            self.canvas1 = FigureCanvas(self.figure1)
            self.toolbar1 = NavigationToolbar(self.canvas1, self)
            grid.addWidget(self.toolbar1, 3, 2)
            grid.addWidget(self.canvas1, 2, 2)
        self.setLayout(grid)
        self.canvas.show()
        self.show()

class IMPORT_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Import_settings()

    def Import_settings(self):
        self.Crop_line = QLineEdit()
        self.Left_line = QLineEdit()
        self.Upper_line = QLineEdit()
        self.Right_line = QLineEdit()
        self.Lower_line = QLineEdit()
        self.Nan_value_line = QLineEdit()
        self.Data_type_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.Crop_line, 1, 2)
        self.grid.addWidget(self.Left_line, 2, 2)
        self.grid.addWidget(self.Upper_line, 3, 2)
        self.grid.addWidget(self.Right_line, 4, 2)
        self.grid.addWidget(self.Lower_line, 5, 2)
        self.grid.addWidget(self.Nan_value_line, 6,2)
        self.grid.addWidget(self.Data_type_line, 7, 2)
        self.Crop_line.setText(str(self.application.crop_))
        self.Left_line.setText(str(self.application.left))
        self.Upper_line.setText(str(self.application.upper))
        self.Right_line.setText(str(self.application.right))
        self.Lower_line.setText(str(self.application.lower))
        self.Nan_value_line.setText(str(self.application.non_val_reduction))
        self.Data_type_line.setText(str(self.application.data_type))
        btn_crop = QPushButton('Crop the image')
        btn_left = QPushButton('Enter a left edge')
        btn_upper = QPushButton('Enter a upper edge')
        btn_right = QPushButton('Enter a right edge')
        btn_lower = QPushButton('Enter a lower edge')
        btn_nan_value = QPushButton('Change nan value to 0')
        btn_data_type = QPushButton('Select data type')
        btn_finish = QPushButton('Finish')
        btn_crop.clicked.connect(self.Get_crop)
        btn_left.clicked.connect(self.Get_int_left)
        btn_upper.clicked.connect(self.Get_int_upper)
        btn_right.clicked.connect(self.Get_int_right)
        btn_lower.clicked.connect(self.Get_int_lower)
        btn_nan_value.clicked.connect(self.Get_nan_value)
        btn_data_type.clicked.connect(self.Select_data_type)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_crop, 1, 1)
        self.grid.addWidget(btn_left, 2, 1)
        self.grid.addWidget(btn_upper, 3, 1)
        self.grid.addWidget(btn_right, 4, 1)
        self.grid.addWidget(btn_lower, 5, 1)
        self.grid.addWidget(btn_nan_value, 6, 1)
        self.grid.addWidget(btn_data_type, 7, 1)
        self.grid.addWidget(btn_finish,8,1)
        self.setLayout(self.grid)
        self.show()

    def Get_crop(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.crop_ = True
            self.Crop_line.setText(str(self.application.crop_))
        elif okPressed and text_sel == 'No':
            self.application.crop_ = False
            self.Crop_line.setText(str(self.application.crop_))

    def Get_int_left(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.left = value
            self.Left_line.setText(str(self.application.left))

    def Get_int_upper(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.upper = value
            self.Upper_line.setText(str(self.application.upper))

    def Get_int_right(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.right = value
            self.Right_line.setText(str(self.application.right))

    def Get_int_lower(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.lower = value
            self.Lower_line.setText(str(self.application.lower))

    def Get_nan_value(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.non_val_reduction = True
            self.Nan_value_line.setText(str(self.application.non_val_reduction))

        elif okPressed and text_sel == 'No':
            self.application.non_val_reduction = False
            self.Nan_value_line.setText(str(self.application.non_val_reduction))

    def Select_data_type(self):
        data_types = ('txt', 'csv', 'image','GeoTIFF')
        data_type_sel, okPressed = QInputDialog.getItem(self, 'Select data type','list of data types', data_types, 0,
                                                        False)
        if okPressed and data_type_sel == 'txt':
            self.application.data_type = data_type_sel
            self.Data_type_line.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'csv':
            self.application.data_type = data_type_sel
            self.Data_type_line.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'image':
            self.application.data_type = data_type_sel
            self.Data_type_line.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'GeoTIFF':
            self.application.data_type = data_type_sel
            self.Data_type_line.setText(str(self.application.data_type))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 8, 2)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class SELECT_widgets(QWidget):

    def __init__(self,database_in=Current_database,directory_in=Current_directory):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.Select_current_image(database_in=self.database_in,directory_in=self.directory_in)

    def Select_current_image(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.DDD_run))
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setHorizontalHeaderLabels(['Activation', 'Initialization directory', 'Shape of the image',
                                                    'File name', 'Stack sizes','Dimension of parameters',
                                                    'Left upper corner', 'Coordinate system', 'Resolution'])
        self.layout = QVBoxLayout()
        for i in range(len(database_in.DDD_run)):
            if database_in.DDD_run[i] == directory_in.DDD_run:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(database_in.DDD_run[i])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.input_image[i].shape)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(database_in.fileName[i])))
            mask_stack_shapes = []
            for j in range(len(database_in.stack[i])):
                mask_stack_shapes.append(database_in.stack[i][j].shape)
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(mask_stack_shapes)))
            parameters_shapes = [database_in.parametrization[i].shape]
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(database_in.origin[i])))
            self.tableWidget.setItem(i, 7, QTableWidgetItem(str(database_in.system[i])))
            self.tableWidget.setItem(i, 8, QTableWidgetItem(str(database_in.resolut[i])))
        btn_select = QPushButton('Select')
        btn_copy = QPushButton('Copy')
        btn_select.clicked.connect(self.Select_current_)
        btn_copy.clicked.connect(self.Copy_current_)
        self.layout.addWidget(btn_select)
        self.layout.addWidget(btn_copy)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.show()

    def Select_current_(self):
        to_choose = (np.arange(1,len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Current image', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.directory_in.set_, self.directory_in.DDD_run, self.directory_in.input_image, \
                self.directory_in.fileName, self.directory_in.stack, self.directory_in.parametrization,\
                self.directory_in.origin, self.directory_in.system, self.directory_in.resolut = \
                    self.database_in.set_[c_img], self.database_in.DDD_run[c_img], self.database_in.input_image[c_img],\
                    self.database_in.fileName[c_img], self.database_in.stack[c_img],\
                    self.database_in.parametrization[c_img], self.database_in.origin[c_img],\
                    self.database_in.system[c_img], self.directory_in.resolut[c_img]

    def Copy_current_(self):
        to_choose = (np.arange(1,len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Current image', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.database_in.set_.append(self.database_in.set_[c_img])
                self.database_in.DDD_run.append(self.database_in.DDD_run[c_img])
                self.database_in.input_image.append(self.database_in.input_image[c_img])
                self.database_in.fileName.append(self.database_in.fileName[c_img])
                self.database_in.stack.append(self.database_in.stack[c_img])
                self.database_in.parametrization.append(self.database_in.parametrization[c_img])
                self.database_in.origin.append(self.database_in.origin[c_img])
                self.database_in.system.append(self.database_in.system[c_img])
                self.database_in.resolut.append(self.database_in.resolut[c_img])


class MASK_STACKING_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Mask_stacking_settings()

    def Mask_stacking_settings(self):
        self.Masks_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.Masks_line, 1, 3)
        self.Masks_line.setText(str(self.application.masks))
        btn_masks = QPushButton('Select masks')
        btn_masks_reset = QPushButton('Reset masks')
        btn_finish = QPushButton('Finish')
        btn_masks.clicked.connect(self.Select_mask)
        btn_masks_reset.clicked.connect(self.Reset_masks)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_masks, 1, 1)
        self.grid.addWidget(btn_masks_reset, 1, 2)
        self.grid.addWidget(btn_finish,8,1)
        self.setLayout(self.grid)
        self.show()

    def Reset_masks(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.Masks_line.setText(str('0'))
            self.application.masks = []

    def Select_mask(self):
        value_types = ('5', '7', '9', '11', '13', '15', '17', '19', '21')
        value_types_int = [5,7,9,11,13,15,17,19,21]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Masks', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_int)):
                if value_types_int[i] == int(value_sel):
                    if len(self.application.masks) == 0:
                        self.application.masks.append(int(value_sel))
                        self.Masks_line.setText(str(self.application.masks))
                    elif len(self.application.masks) == 1:
                        if self.application.masks != int(value_sel):
                            self.application.masks.append(int(value_sel))
                            self.Masks_line.setText(str(self.application.masks))
                    elif len(self.application.masks) > 1:
                        check = 0
                        for j in range(len(self.application.masks)):
                            if self.application.masks[j] == int(value_sel):
                                check += 1
                        if check == 0:
                            self.application.masks.append(int(value_sel))
                            self.Masks_line.setText(str(self.application.masks))
    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 8, 2)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class PARAMETRIZATION_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Parametrization_settings()

    def Parametrization_settings(self):
        self.Trend_detection_line = QLineEdit()
        self.Var_line = QLineEdit()
        self.STD_dev_line = QLineEdit()
        self.Defo_line = QLineEdit()
        self.Semi_Var_line = QLineEdit()
        self.Gauss_line = QLineEdit()
        self.Depths_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.Trend_detection_line, 1, 3)
        self.grid.addWidget(self.Var_line, 2, 3)
        self.grid.addWidget(self.STD_dev_line, 3, 3)
        self.grid.addWidget(self.Defo_line, 4, 3)
        self.grid.addWidget(self.Semi_Var_line, 5, 3)
        self.grid.addWidget(self.Gauss_line, 6, 3)
        self.grid.addWidget(self.Depths_line, 7, 3)
        self.Depths_line.setText(str(self.application.depths))
        self.Trend_detection_line.setText(str(self.application.trend_parameter))
        self.Var_line.setText(str(self.application.var_parameter))
        self.STD_dev_line.setText(str(self.application.standard_dev))
        self.Defo_line.setText(str(self.application.deformation_parameter))
        self.Semi_Var_line.setText(str(self.application.semi_var_parameter))
        self.Gauss_line.setText(str(self.application.gaussian_par))
        btn_TD = QPushButton('Trend detection ')
        btn_V = QPushButton('Variance from px')
        btn_STD = QPushButton('Standard deviation')
        btn_Def = QPushButton('Deformation value')
        btn_SVV = QPushButton('Semi-variance')
        btn_G = QPushButton('Differential Gaussian')
        btn_Dep = QPushButton('Depths class to detect')
        btn_Dep_reset = QPushButton('Reset depths')
        btn_finish = QPushButton('Finish')
        btn_TD.clicked.connect(self.Get_TD)
        btn_V.clicked.connect(self.Get_V)
        btn_STD.clicked.connect(self.Get_STD)
        btn_Def.clicked.connect(self.Get_Def)
        btn_SVV.clicked.connect(self.Get_SW)
        btn_G.clicked.connect(self.Get_G)
        btn_Dep.clicked.connect(self.Get_Dep)
        btn_Dep_reset.clicked.connect(self.Reset_dep)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_TD, 1, 1)
        self.grid.addWidget(btn_V, 2, 1)
        self.grid.addWidget(btn_STD, 3, 1)
        self.grid.addWidget(btn_Def, 4, 1)
        self.grid.addWidget(btn_SVV, 5, 1)
        self.grid.addWidget(btn_G, 6, 1)
        self.grid.addWidget(btn_Dep, 7, 1)
        self.grid.addWidget(btn_Dep_reset, 7, 2)
        self.grid.addWidget(btn_finish, 9, 1)
        self.setLayout(self.grid)
        self.show()

    def Get_TD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.trend_parameter = True
            self.Trend_detection_line.setText(str(self.application.trend_parameter))
        elif okPressed and text_sel == 'No':
            self.application.trend_parameter = False
            self.Trend_detection_line.setText(str(self.application.trend_parameter))

    def Get_V(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.var_parameter = True
            self.Var_line.setText(str(self.application.var_parameter))
        elif okPressed and text_sel == 'No':
            self.application.var_parameter = False
            self.Var_line.setText(str(self.application.var_parameter))

    def Get_STD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.standard_dev = True
            self.STD_dev_line.setText(str(self.application.standard_dev))
        elif okPressed and text_sel == 'No':
            self.application.standard_dev = False
            self.STD_dev_line.setText(str(self.application.standard_dev))

    def Get_Def(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.deformation_parameter = True
            self.Defo_line.setText(str(self.application.deformation_parameter))
        elif okPressed and text_sel == 'No':
            self.application.deformation_parameter = False
            self.Defo_line.setText(str(self.application.deformation_parameter))

    def Get_SW(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.semi_var_parameter = True
            self.Semi_Var_line.setText(str(self.application.semi_var_parameter))
        elif okPressed and text_sel == 'No':
            self.application.semi_var_parameter = False
            self.Semi_Var_line.setText(str(self.application.semi_var_parameter))

    def Get_G(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.gaussian_par = True
            self.Gauss_line.setText(str(self.application.gaussian_par))
        elif okPressed and text_sel == 'No':
            self.application.gaussian_par = False
            self.Gauss_line.setText(str(self.application.gaussian_par))

    def Reset_dep(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.depths = []
            self.Depths_line.setText(str('0'))

    def Get_Dep(self):
        value_types = ('+0.200', '+0.100', '+0.050', '+0.040', '+0.030', '+0.020', '+0.015', '+0.010', '+0.005',
                       '+0.000', '-0.005', '-0.010', '-0.015', '-0.020', '-0.030', '-0.040', '-0.050', '-0.100',
                       '-0.200')
        value_types_float = [0.2,0.1,0.05,0.04,0.03,0.02,0.015,0.01,0.005,0,(-0.005),(-0.01),(-0.015),(-0.02),(-0.03),
                             (-0.04),(-0.05),(-0.1),(-0.2)]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Masks', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    if len(self.application.depths) == 0:
                        self.application.depths.append(float(value_sel))
                        self.Depths_line.setText(str(self.application.depths))
                    elif len(self.application.depths) == 1:
                        if self.application.depths != float(value_sel):
                            self.application.depths.append(float(value_sel))
                            self.Depths_line.setText(str(self.application.depths))
                    elif len(self.application.depths) > 1:
                        check = 0
                        for j in range(len(self.application.depths)):
                            if self.application.depths[j] == float(value_sel):
                                check += 1
                        if check == 0:
                            self.application.depths.append(float(value_sel))
                            self.Depths_line.setText(str(self.application.depths))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 9, 3)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class Set_training_data_widget(QWidget):

    def __init__(self,database_in=Training_database,directory_in=Training_current_directory,
                 database_in_all=Current_database):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.database_in_all = database_in_all
        self.Set_training_dataset(database_in=self.database_in,directory_in=self.directory_in)

    def Set_training_dataset(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.set_))
        self.tableWidget.setColumnCount(14)
        self.tableWidget.setHorizontalHeaderLabels(['Type','Active', 'X data init.', 'X data',
                                                    'X data standardized', 'Y data init.','Y data', 'Y data OvR',
                                                    'Classes','Weights','Cost sums', 'Probability images',
                                                    'Predict image', 'Error rate'])
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        for i in range(len(database_in.set_)):
            if database_in.type_of[i] == 0:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('NN'))
            elif database_in.type_of[i] == 1:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Testing'))
            elif database_in.type_of[i] == 2:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Learning'))
            elif database_in.type_of[i] == 3:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Master Learning'))
            if database_in.set_[i] == directory_in.set_:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.x_set_[i])))
            parameters_shapes = [database_in.x_data[i].shape]
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.x_data_std[i].shape]
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.y_set_[i])))
            parameters_shapes = [database_in.y_data[i].shape]
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.y_data_ovr[i].shape]
            self.tableWidget.setItem(i, 7, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 8, QTableWidgetItem(str(database_in.classes[i])))
            parameters_shapes = [database_in.training_weights[i].shape]
            self.tableWidget.setItem(i, 9, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.cost_sums[i].shape]
            self.tableWidget.setItem(i, 10, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.probe_images[i].shape]
            self.tableWidget.setItem(i, 11, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.predicted_image[i].shape]
            self.tableWidget.setItem(i, 12, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 13, QTableWidgetItem(str(database_in.error_rate[i])))
        btn_select = QPushButton('Select training data set')
        btn_select.clicked.connect(self.Select_current_training)
        btn_select_2 = QPushButton('Select Y data set to active set')
        btn_select_2.clicked.connect(self.Select_y_data)
        btn_select_3 = QPushButton('Select master learning set')
        btn_select_3.clicked.connect(self.Select_Master_training)
        btn_select_4 = QPushButton('Select other learning set')
        btn_select_4.clicked.connect(self.Select_other_training)
        btn_select_5 = QPushButton('Select testing set')
        btn_select_5.clicked.connect(self.Select_testing)
        btn_select_6 = QPushButton('Select weigths and classes to testing data')
        btn_select_6.clicked.connect(self.Select_training_p_to_test)
        self.grid.addWidget(btn_select, 1, 1)
        self.grid.addWidget(btn_select_2, 1, 2)
        self.grid.addWidget(btn_select_3, 1, 3)
        self.grid.addWidget(btn_select_4, 1, 4)
        self.grid.addWidget(btn_select_5, 1, 5)
        self.grid.addWidget(btn_select_6, 1, 6)
        self.grid.addWidget(self.tableWidget, 2, 1, 6, 6)
        self.setLayout(self.grid)
        self.show()

    def Select_current_training(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Current model data set', to_choose, 0, False)
            c_tr = int(no_sel) - 1
            if okPressed:
                self.directory_in.type_of, self.directory_in.set_, self.directory_in.x_data_init,\
                self.directory_in.x_set_, self.directory_in.x_data, self.directory_in.x_data_std,\
                self.directory_in.y_set_, self.directory_in.y_data_init, self.directory_in.y_data,\
                self.directory_in.y_data_ovr, self.directory_in.standardization_parameters,self.directory_in.classes,\
                self.directory_in.training_weights, self.directory_in.cost_sums, self.directory_in.probe_images,\
                self.directory_in.predicted_image, self.directory_in.decorel_seq, self.directory_in.error_rate\
                    = self.database_in.type_of[c_tr], self.database_in.set_[c_tr], self.database_in.x_data_init[c_tr],\
                      self.database_in.x_set_[c_tr], self.database_in.x_data[c_tr], self.database_in.x_data_std[c_tr],\
                      self.database_in.y_set_[c_tr], self.database_in.y_data_init[c_tr], self.database_in.y_data[c_tr],\
                      self.database_in.y_data_ovr[c_tr], self.database_in.standardization_parameters[c_tr],\
                      self.database_in.classes[c_tr], self.database_in.training_weights[c_tr],\
                      self.database_in.cost_sums[c_tr], self.database_in.probe_images[c_tr],\
                      self.database_in.predicted_image[c_tr], self.database_in.decorel_seq, self.database_in.error_rate

    def Select_y_data(self):
        to_choose = (np.arange(1,len(self.database_in_all.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Y data set', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.directory_in.y_data = self.database_in_all.stack[c_img]
                self.directory_in.y_data_init = self.database_in_all.DDD_run[c_img]
                self.directory_in.y_set_ = self.database_in_all.set_[c_img]
                for i in range(len(self.database_in.set_)):
                    if self.database_in.set_[i] == self.directory_in.set_:
                        self.database_in.y_data[i] = self.directory_in.y_data
                        self.database_in.y_set_[i] = self.directory_in.y_set_
                        self.database_in.y_data_init[i] = self.directory_in.y_data_init

    def Select_Master_training(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Master training data set', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                for i in range(len(self.database_in.set_)):
                    if self.database_in.type_of[i] == 3:
                        self.database_in.type_of[i] = 2
                self.database_in.type_of[c_img] = 3
                self.Select_current_training()

    def Select_other_training(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Other training data set', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.database_in.type_of[c_img] = 2
                self.Select_current_training()

    def Select_testing(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Testing data set', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.database_in.type_of[c_img] = 1
                self.Select_current_training()

    def Select_training_p_to_test(self):
        self.Select_current_training()
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Training data set parameters', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                self.directory_in.classes = self.database_in.classes[c_img]
                self.directory_in.training_weights = self.database_in.training_weights[c_img]
                self.directory_in.x_data_init.W_ = self.database_in.training_weights[c_img]
                for i in range(len(self.database_in.set_)):
                    if self.database_in.set_[i] == self.directory_in.set_:
                        self.database_in.classes[i] = self.directory_in.classes
                        self.database_in.training_weights[i] = self.directory_in.training_weights
                        self.database_in.x_data_init[i].W_ = self.directory_in.training_weights

class Cost_function_widgets(QWidget):

    def __init__(self,cost_sum=[],classes=[]):
        super().__init__()
        self.cost_sum = cost_sum
        self.classes = classes
        self.initUI(self.cost_sum,self.classes)

    def initUI(self,cost_sum,classes):
        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 1)
        Status_toolbar.setText(('Number of classes: ' + str(len(classes)) +
                                '; Number of iteration: ' + str(len(cost_sum[0, :]))))
        for i in range(len(classes)):
            self.figure = plt.figure()
            self.axes = self.figure.add_subplot(111)
            self.axes.set_title('Class: ' + str(classes[i]))
            self.axes.set_xlabel('Iteration')
            self.axes.set_ylabel('Sum of logistic cost')
            self.axes.plot(range(1, len(cost_sum[i, :]) + 1), cost_sum[i, :], marker='o')
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            if ((i + 1) % 3) > 0:
                if ((i + 1) % 2) > 0:
                    grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 1)
                    grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 1)
                else:
                    grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 2)
                    grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 2)
            else:
                grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 3)
                grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 3)
        self.setLayout(grid)
        self.canvas.show()
        self.show()

class Probe_images_widgets(QWidget):

    def __init__(self,probe_img=[],classes=[]):
        super().__init__()
        self.probe_img = probe_img
        self.classes = classes
        self.initUI(self.probe_img,self.classes)

    def initUI(self,probe_img,classes):
        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 1)
        Status_toolbar.setText(('Number of classes: ' + str(len(classes)) +
                                '; Main class: ' + str(classes[0])))
        for i in range(len(classes)):
            self.figure = plt.figure()
            self.axes = self.figure.add_subplot(111)
            self.axes.set_title('Class: ' + str(classes[i]))
            im = self.axes.imshow(probe_img[:,:,i])
            self.figure.colorbar(im)
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            if ((i + 1) % 3) > 0:
                if ((i + 1) % 2) > 0:
                    grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 1)
                    grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 1)
                else:
                    grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 2)
                    grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 2)
            else:
                grid.addWidget(self.toolbar, 3 + (2 * int(i / 3)), 3)
                grid.addWidget(self.canvas, 2 + (2 * int(i / 3)), 3)
        self.setLayout(grid)
        self.canvas.show()
        self.show()

class ERROR_check_widgets(QWidget):

    def __init__(self,differential_image='None'):
        super().__init__()
        self.differential_image = differential_image
        self.initUI(self.differential_image)

    def initUI(self,data):
        data_array = np.array(data)
        error_rate = np.sum(data_array) / np.size(data_array)

        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 2)

        Status_toolbar.setText(('Error rate: ' + str(error_rate)))

        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        im = self.axes.imshow(data[::-1])
        self.figure.colorbar(im)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        grid.addWidget(self.toolbar, 5, 1, 5, 1)
        grid.addWidget(self.canvas, 1, 1, 5, 1)

        self.setLayout(grid)
        self.canvas.show()
        self.show()

class DECORELATION_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Parametrization_settings()

    def Parametrization_settings(self):
        self.max_rate_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.max_rate_line, 1, 3)
        self.max_rate_line.setText(str(self.application.max_rate))
        btn_MR = QPushButton('Maximum rate')
        btn_finish = QPushButton('Finish')
        btn_MR.clicked.connect(self.Get_double_max_rate)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_MR, 1, 1)
        self.grid.addWidget(btn_finish, 9, 1)
        self.setLayout(self.grid)
        self.show()

    def Get_double_max_rate(self):
        value, okPressed = QInputDialog.getDouble(self, 'Get max rate','Value:', 0, 0, 1, 2)
        if okPressed:
            self.application.max_rate = value
            self.max_rate_line.setText(str(value))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 9, 3)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class EXPORT_image_set_widgets(QWidget):

    def __init__(self,database_in=Current_database,directory_in=Current_directory):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.Select_export_image(database_in=self.database_in,directory_in=self.directory_in)

    def Select_export_image(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.DDD_run))
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setHorizontalHeaderLabels(['Activation', 'Initialization directory', 'No of pixels','File name',
                                                    'Stack sizes','Dimension of parameters',
                                                    'Left upper corner', 'Coordinate system', 'Resolution'])
        self.grid = QGridLayout()
        for i in range(len(database_in.DDD_run)):
            if database_in.DDD_run[i] == directory_in.DDD_run:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(database_in.DDD_run[i])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(np.size(database_in.input_image[i]))))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(database_in.fileName[i])))
            mask_stack_shapes = []
            for j in range(len(database_in.stack[i])):
                mask_stack_shapes.append(database_in.stack[i][j].shape)
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(mask_stack_shapes)))
            parameters_shapes = [database_in.parametrization[i].shape]
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(database_in.origin[i])))
            self.tableWidget.setItem(i, 7, QTableWidgetItem(str(database_in.system[i])))
            self.tableWidget.setItem(i, 8, QTableWidgetItem(str(database_in.resolut[i])))
        btn_select = QPushButton('Export')
        btn_select.clicked.connect(self.Select_to_export_i)
        self.grid.addWidget(btn_select, 1, 1)
        self.grid.addWidget(self.tableWidget, 2, 1, 6, 6)
        self.setLayout(self.grid)
        self.show()

    def Select_to_export_i(self):
        to_choose = (np.arange(1,len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Export image', to_choose, 0, False)
            c_img = int(no_sel) - 1
            if okPressed:
                if osp.exists('./EXPORT'):
                    pass
                else:
                    os.mkdir('./EXPORT')
                if osp.exists('./EXPORT/IMAGE_DATA_SET'):
                    pass
                else:
                    os.mkdir('./EXPORT/IMAGE_DATA_SET')
                now = datetime.now()
                dt_str = now.strftime("%Y%m%d_%H%M%S")
                File_name = './EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str
                os.mkdir(File_name)
                f1 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Init_directory.pkl','wb')
                pickle.dump(self.database_in.DDD_run[c_img].__dict__, f1)
                f1.close()
                f2 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Image.pkl','wb')
                pickle.dump(self.database_in.input_image[c_img], f2)
                f2.close()
                f3 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Originally_directory.pkl','wb')
                pickle.dump(self.database_in.fileName[c_img], f3)
                f3.close()
                f4 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Mask_Stack.pkl','wb')
                pickle.dump(self.database_in.stack[c_img], f4)
                f4.close()
                f5 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Parameters.pkl', 'wb')
                pickle.dump(self.database_in.parametrization[c_img], f5)
                f5.close()
                f6 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Left_upper_corner.pkl', 'wb')
                pickle.dump(self.database_in.origin[c_img], f6)
                f6.close()
                f7 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Coordinate_system.pkl', 'wb')
                pickle.dump(self.database_in.system[c_img], f7)
                f7.close()
                f8 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Resolution.pkl', 'wb')
                pickle.dump(self.database_in.resolut[c_img], f8)
                f8.close()
                f9 = open('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Import_image_set.txt', 'w')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Init_directory.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Image.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Originally_directory.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Mask_Stack.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Parameters.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Left_upper_corner.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Coordinate_system.pkl\n')
                f9.write('./EXPORT/IMAGE_DATA_SET/' + 'IMAGE_SET_' + dt_str + '/Resolution.pkl\n')
                f9.close()

class Export_model_set_widget(QWidget):

    def __init__(self,database_in=Training_database,directory_in=Training_current_directory,
                 database_in_all=Current_database):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.database_in_all = database_in_all
        self.Select_export_model(database_in=self.database_in,directory_in=self.directory_in)

    def Select_export_model(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.set_))
        self.tableWidget.setColumnCount(14)
        self.tableWidget.setHorizontalHeaderLabels(['Type','Active', 'X data init.', 'X data',
                                                    'X data standardized', 'Y data init.','Y data', 'Y data OvR',
                                                    'Classes','Weights','Cost sums', 'Probability images',
                                                    'Predict image', 'Error rate'])
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        for i in range(len(database_in.set_)):
            if database_in.type_of[i] == 0:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('NN'))
            elif database_in.type_of[i] == 1:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Testing'))
            elif database_in.type_of[i] == 2:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Learning'))
            elif database_in.type_of[i] == 3:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Master Learning'))
            if database_in.set_[i] == directory_in.set_:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.x_set_[i])))
            parameters_shapes = [database_in.x_data[i].shape]
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.x_data_std[i].shape]
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.y_set_[i])))
            parameters_shapes = [database_in.y_data[i].shape]
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.y_data_ovr[i].shape]
            self.tableWidget.setItem(i, 7, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 8, QTableWidgetItem(str(database_in.classes[i])))
            parameters_shapes = [database_in.training_weights[i].shape]
            self.tableWidget.setItem(i, 9, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.cost_sums[i].shape]
            self.tableWidget.setItem(i, 10, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.probe_images[i].shape]
            self.tableWidget.setItem(i, 11, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.predicted_image[i].shape]
            self.tableWidget.setItem(i, 12, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 13, QTableWidgetItem(str(database_in.error_rate[i])))
        btn_export = QPushButton('Export')
        btn_export.clicked.connect(self.Select_to_export_m)
        self.grid.addWidget(btn_export, 1, 1)
        self.grid.addWidget(self.tableWidget, 2, 1, 6, 6)
        self.setLayout(self.grid)
        self.show()

    def Select_to_export_m(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Export model', to_choose, 0, False)
            c_tr = int(no_sel) - 1
            if okPressed:
                if osp.exists('./EXPORT'):
                    pass
                else:
                    os.mkdir('./EXPORT')
                if osp.exists('./EXPORT/MODEL_DATA_SET'):
                    pass
                else:
                    os.mkdir('./EXPORT/MODEL_DATA_SET')
                now = datetime.now()
                dt_str = now.strftime("%Y%m%d_%H%M%S")
                File_name = './EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str
                os.mkdir(File_name)
                f1 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data.pkl', "wb")
                pickle.dump(self.database_in.x_data[c_tr], f1)
                f1.close()
                f2 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data_Init_directory.pkl', "wb")
                pickle.dump(self.database_in.x_data_init[c_tr].__dict__, f2)
                f2.close()
                f3 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data_standarized.pkl', "wb")
                pickle.dump(self.database_in.x_data_std[c_tr], f3)
                f3.close()
                f4 = open('./MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data.pkl', "wb")
                pickle.dump(self.database_in.y_data[c_tr], f4)
                f4.close()
                f5 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data_Init_directory.pkl', "wb")
                if np.max(self.database_in.y_set_[c_tr]) == 0:
                    pickle.dump(self.database_in.y_data_init[c_tr], f5)
                else:
                    pickle.dump(self.database_in.y_data_init[c_tr].__dict__, f5)
                f5.close()
                f6 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data_OneVsRest.pkl', "wb")
                pickle.dump(self.database_in.y_data_ovr[c_tr], f6)
                f6.close()
                f7 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Standardization_parameters.pkl', "wb")
                pickle.dump(self.database_in.standardization_parameters[c_tr], f7)
                f7.close()
                f8 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Model_weights.pkl', "wb")
                pickle.dump(self.database_in.training_weights[c_tr], f8)
                f8.close()
                f9 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Model_classes.pkl', "wb")
                pickle.dump(self.database_in.classes[c_tr], f9)
                f9.close()
                f10 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Cost_function_stat.pkl', "wb")
                pickle.dump(self.database_in.cost_sums[c_tr], f10)
                f10.close()
                f11 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Probabilistic_images.pkl', "wb")
                pickle.dump(self.database_in.probe_images[c_tr], f11)
                f11.close()
                f12 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Predicted_image.pkl', "wb")
                pickle.dump(self.database_in.predicted_image[c_tr], f12)
                f12.close()
                f13 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Decorelation_sequence.pkl', "wb")
                pickle.dump(self.database_in.decorel_seq[c_tr], f13)
                f13.close()
                f15 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Prediction_error_rate.pkl', "wb")
                pickle.dump(self.database_in.error_rate[c_tr], f15)
                f15.close()
                f14 = open('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Import_model_set.txt', 'w')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data_Init_directory.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/X_data_standarized.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data_Init_directory.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Y_data_OneVsRest.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Standardization_parameters.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Model_weights.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Model_classes.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Cost_function_stat.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Probabilistic_images.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Predicted_image.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Decorelation_sequence.pkl\n')
                f14.write('./EXPORT/MODEL_DATA_SET/' + 'MODEL_SET_' + dt_str + '/Prediction_error_rate.pkl\n')
                f14.close()

class Update_model_set_widget(QWidget):

    def __init__(self,database_in=Training_database,directory_in=Training_current_directory,
                 database_in_all=Current_database):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.database_in_all = database_in_all
        self.Show_training_dataset(database_in=self.database_in,directory_in=self.directory_in)

    def Show_training_dataset(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.set_))
        self.tableWidget.setColumnCount(14)
        self.tableWidget.setHorizontalHeaderLabels(['Type','Active', 'X data init.', 'X data',
                                                    'X data standardized', 'Y data init.','Y data', 'Y data OvR',
                                                    'Classes','Weights','Cost sums', 'Probability images',
                                                    'Predict image', 'Error rate'])
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        for i in range(len(database_in.set_)):
            if database_in.type_of[i] == 0:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('NN'))
            elif database_in.type_of[i] == 1:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Testing'))
            elif database_in.type_of[i] == 2:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Learning'))
            elif database_in.type_of[i] == 3:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Master Learning'))
            if database_in.set_[i] == directory_in.set_:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 1, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.x_set_[i])))
            parameters_shapes = [database_in.x_data[i].shape]
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.x_data_std[i].shape]
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.y_set_[i])))
            parameters_shapes = [database_in.y_data[i].shape]
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.y_data_ovr[i].shape]
            self.tableWidget.setItem(i, 7, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 8, QTableWidgetItem(str(database_in.classes[i])))
            parameters_shapes = [database_in.training_weights[i].shape]
            self.tableWidget.setItem(i, 9, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.cost_sums[i].shape]
            self.tableWidget.setItem(i, 10, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.probe_images[i].shape]
            self.tableWidget.setItem(i, 11, QTableWidgetItem(str(parameters_shapes)))
            parameters_shapes = [database_in.predicted_image[i].shape]
            self.tableWidget.setItem(i, 12, QTableWidgetItem(str(parameters_shapes)))
            self.tableWidget.setItem(i, 13, QTableWidgetItem(str(database_in.error_rate[i])))
        btn_select = QPushButton('Select data sets to merge')
        btn_select.clicked.connect(self.Select_data_to_merge)
        self.grid.addWidget(btn_select, 1, 1)
        self.grid.addWidget(self.tableWidget, 2, 1, 6, 6)
        self.setLayout(self.grid)
        self.show()

    def Select_data_to_merge(self):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Number of sets to merge', to_choose, 0, False)
            if okPressed:
                no_seli = int(no_sel)
                self.temp_database = []
                for i in range(no_seli):
                    self.Select_set_to_merge(temp_database=self.temp_database)
                if np.size(self.temp_database) > 0:
                    x_data = self.database_in.x_data[self.temp_database[0]]
                    y_data = self.database_in.y_data[self.temp_database[0]]
                    if len(self.temp_database) == 2:
                        x_data = np.append(x_data, self.database_in.x_data[self.temp_database[1]], axis=0)
                        y_data = np.append(y_data, self.database_in.y_data[self.temp_database[1]], axis=0)
                    elif len(self.temp_database) >= 3:
                        for i in range(len(self.temp_database) - 1):
                            x_data = np.append(x_data, self.database_in.x_data[self.temp_database[i + 1]], axis=0)
                            y_data = np.append(y_data, self.database_in.y_data[self.temp_database[i + 1]], axis=0)
                    self.database_in.set_.append(np.size(self.database_in.set_) + 1)
                    self.database_in.x_set_.append([0])
                    self.database_in.x_data.append(x_data)
                    self.database_in.x_data_init.append(self.database_in.x_data_init[self.temp_database[0]])
                    self.database_in.x_data_std.append(np.array([[],[]]))
                    self.database_in.y_set_.append([0])
                    self.database_in.y_data_init.append(self.database_in.y_data_init[self.temp_database[0]])
                    self.database_in.y_data.append(y_data)
                    self.database_in.y_data_ovr.append(np.array([[],[]]))
                    self.database_in.standardization_parameters.append(np.array([[],[]]))
                    self.database_in.training_weights.append(np.array([[],[]]))
                    self.database_in.classes.append(np.array([]))
                    self.database_in.type_of.append(0)
                    self.database_in.cost_sums.append(np.array([[], []]))
                    self.database_in.probe_images.append(np.array([[], []]))
                    self.database_in.predicted_image.append(np.array([[], []]))
                    self.database_in.decorel_seq.append(np.array([[], []]))
                    self.database_in.error_rate.append([0])
                    self.temp_database = []

    def Select_set_to_merge(self,temp_database):
        to_choose = (np.arange(1,len(self.database_in.set_) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select','Set to merge', to_choose, 0, False)
            if okPressed:
                set_no = int(no_sel) - 1
                temp_database.append(self.database_in.set_[set_no] - 1)

class ART_IMPACT_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Art_impact_settings()

    def Art_impact_settings(self):
        self.Impact_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.Impact_line, 1, 3)
        self.Impact_line.setText(str(self.application.max_impact_px))
        btn_Imp = QPushButton('Impact of artifacts')
        btn_Imp_reset = QPushButton('Set to default')
        btn_finish = QPushButton('Finish')
        btn_Imp.clicked.connect(self.Get_IMP)
        btn_Imp_reset.clicked.connect(self.Reset_IMP)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_Imp, 1, 1)
        self.grid.addWidget(btn_Imp_reset, 1, 2)
        self.grid.addWidget(btn_finish, 2, 1)
        self.setLayout(self.grid)
        self.show()

    def Reset_IMP(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.max_impact_px = 10
            self.Impact_line.setText(str(self.application.max_impact_px))

    def Get_IMP(self):
        value_types = ('3px', '4px', '5px', '7px', '10px', '15px', '20px', '25px', '30px')
        value_types_float = [3,4,5,7,10,15,20,25,30]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Impact radius', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel[:-2]):
                    self.application.max_impact_px = value_types_float[i]
                    self.Impact_line.setText(str(self.application.max_impact_px))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 2, 3)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class Removing_jumps_widgets(QWidget):

    def __init__(self,data='None',filename='None',jumps='None',barriers='None',population='None',bins_n='None'):
        super().__init__()
        self.data = data
        self.filename = filename
        self.jumps = jumps
        self.barriers = barriers
        self.population = population
        self.bins_n = bins_n
        self.initUI(self.data,self.filename,self.jumps,self.barriers,self.population,self.bins_n)

    def initUI(self,data,filename,jumps,barriers,population,bins_n):
        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 2)

        Data_toolbar = QTextEdit()
        grid.addWidget(Data_toolbar, 4, 2, 5, 2)

        Status_toolbar.setText(str(filename))
        Data_toolbar.setText(('Data:' + '\n'))
        Data_toolbar.append((str(data) + '\n'))
        Data_toolbar.append(('Shape:' + '\n'))
        Data_toolbar.append(str(data.shape))

        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        im = self.axes.imshow(data[::-1])
        self.figure.colorbar(im)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        grid.addWidget(self.toolbar, 5, 1, 5, 1)
        grid.addWidget(self.canvas, 1, 1, 5, 1)
        hist_width = np.min(bins_n[0:-1] - bins_n[1:])
        self.figure1 = plt.figure()
        self.axes1 = self.figure1.add_subplot(111)
        self.axes1.bar(bins_n[0:-1], population, alpha=1, width=hist_width)
        self.axes1.set_ylim([0, np.max(population)])
        self.axes1.set_xlabel('Value [m]')
        self.axes1.set_ylabel('Frequency')
        self.axes1.set_title('Histogram before removing')
        for xc in barriers[0:-1]:
            plt.axvline(x=xc, label='line at x = {}'.format(xc), c='black')
        for xc in jumps:
            plt.axvline(x=xc, label='line at x = {}'.format(xc), c='green')
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        grid.addWidget(self.toolbar1, 3, 2)
        grid.addWidget(self.canvas1, 2, 2)

        self.setLayout(grid)
        self.canvas.show()
        self.show()

class REM_JUMPS_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.Rem_jumps_settings()

    def Rem_jumps_settings(self):
        self.Mask_size_line = QLineEdit()
        self.Density_line = QLineEdit()
        self.Seed_line = QLineEdit()
        self.Min_pop_line = QLineEdit()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.Mask_size_line, 1, 3)
        self.grid.addWidget(self.Density_line, 2, 3)
        self.grid.addWidget(self.Seed_line, 3, 3)
        self.grid.addWidget(self.Min_pop_line, 4, 3)
        self.Mask_size_line.setText(str(self.application.rm_mask_size))
        self.Density_line.setText(str(self.application.rm_density_))
        self.Seed_line.setText(str(self.application.rm_seed_))
        self.Min_pop_line.setText(str(self.application.rm_min_population_factor))
        btn_MS = QPushButton('Mask size (RJ)')
        btn_MS_reset = QPushButton('Mask size (RJ) reset')
        btn_HD = QPushButton('Histogram density')
        btn_HD_reset = QPushButton('Histogram density reset')
        btn_SD = QPushButton('Histogram seed')
        btn_SD_reset = QPushButton('Histogram seed reset')
        btn_MP = QPushButton('Minimum no of pixel to consider')
        btn_MP_reset = QPushButton('Minimum reset')
        btn_finish = QPushButton('Finish')
        btn_MS.clicked.connect(self.Get_MS)
        btn_MS_reset.clicked.connect(self.Reset_MS)
        btn_HD.clicked.connect(self.Get_HD)
        btn_HD_reset.clicked.connect(self.Reset_HD)
        btn_SD.clicked.connect(self.Get_SD)
        btn_SD_reset.clicked.connect(self.Reset_SD)
        btn_MP.clicked.connect(self.Get_MP)
        btn_MP_reset.clicked.connect(self.Reset_MP)
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_MS, 1, 1)
        self.grid.addWidget(btn_MS_reset, 1, 2)
        self.grid.addWidget(btn_HD, 2, 1)
        self.grid.addWidget(btn_HD_reset, 2, 2)
        self.grid.addWidget(btn_SD, 3, 1)
        self.grid.addWidget(btn_SD_reset, 3, 2)
        self.grid.addWidget(btn_MP, 4, 1)
        self.grid.addWidget(btn_MP_reset, 4, 2)
        self.grid.addWidget(btn_finish, 5, 1)
        self.setLayout(self.grid)
        self.show()

    def Reset_MS(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 31
            self.Mask_size_line.setText(str(self.application.rm_mask_size))

    def Get_MS(self):
        value_types = []
        value_types_float = []
        for i in range(10,50):
            value_types_float.append(i * 2 - 1)
            value_types.append(str(i * 2 - 1))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Mask size to find the class for pixel', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_mask_size = value_types_float[i]
                    self.Mask_size_line.setText(str(self.application.rm_mask_size))

    def Reset_HD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 10
            self.Density_line.setText(str(self.application.rm_mask_size))

    def Get_HD(self):
        value_types = ('5','6','7','8','9','10','11','12','13','14','15')
        value_types_float = [5,6,7,8,9,10,11,12,13,14,15]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The density of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_density_ = value_types_float[i]
                    self.Density_line.setText(str(self.application.rm_density_))

    def Reset_SD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 3
            self.Seed_line.setText(str(self.application.rm_mask_size))

    def Get_SD(self):
        value_types = ('1','2','3','4','5','6')
        value_types_float = [1,2,3,4,5,6]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The seed of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_seed_ = value_types_float[i]
                    self.Seed_line.setText(str(self.application.rm_seed_))

    def Reset_MP(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 0.01
            self.Min_pop_line.setText(str(self.application.rm_mask_size))


    def Get_MP(self):
        value_types = []
        value_types_float = []
        for i in range(1,50):
            value_types_float.append(i)
            value_types.append(str(i))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The seed of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_min_population_factor = value_types_float[i] * 0.01
                    self.Min_pop_line.setText(str(self.application.rm_min_population_factor))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 5, 3)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()

class COREL_widgets(QWidget):

    def __init__(self,table_=[]):
        super().__init__()
        self.table_ = np.array(table_)
        self.Table_view(table_R=self.table_)

    def Table_view(self,table_R):
        self.grid = QGridLayout()
        self.Max_corel_line = QLineEdit()
        self.grid.addWidget(self.Max_corel_line, 1, 3)
        self.Max_corel_line.setText(str('1.00'))
        btn_MC = QPushButton('R rate > ...')
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(table_R.shape[0])
        self.tableWidget.setColumnCount(table_R.shape[1])
        labels = []
        for i in range(table_R.shape[0]):
            labels.append(str(i))
        self.tableWidget.setHorizontalHeaderLabels(labels)
        self.grid.setSpacing(table_R.shape[0])
        for i in range(table_R.shape[0]):
            for j in range(table_R.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(table_R[i,j])))
        btn_MC.clicked.connect(self.Get_MC)
        self.grid.addWidget(btn_MC, 1, 1)
        self.grid.addWidget(self.tableWidget, 2, 1, 10, 10)
        self.setLayout(self.grid)
        self.show()

    def Get_MC(self):
        value, okPressed = QInputDialog.getDouble(self, 'Get value between 0 and 1','Value:', 0.95, 0, 1, 2)
        if okPressed:
            self.Max_corel_line.setText(str(value))
            for i in range(self.table_.shape[0]):
                for j in range(1,self.table_.shape[1]):
                    if float(self.table_[i,j]) > value:
                        self.tableWidget.item(i, j).setBackground(QColor(255,0,0))

class CV_widgets(QWidget):

    def __init__(self, CV_error=[], Matrix_error=[], CV_iter=[]):
        super().__init__()
        self.CV_error = np.array(CV_error)
        self.Matrix_error = np.array(Matrix_error)
        self.CV_iter = np.array(CV_iter)
        self.CV_error_view(CV_error=self.CV_error, Matrix_error=self.Matrix_error, CV_iter=self.CV_iter)

    def CV_error_view(self, CV_error, Matrix_error, CV_iter):
        Status_toolbar = QLineEdit()
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(Status_toolbar, 1, 1)
        Status_toolbar.setText(('Cross validation result: ' + str(CV_error) +
                                '; Number of iteration: ' + str(CV_iter)))
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Error rate in cross validation: ')
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Prediction error rate')
        self.axes.plot(range(1, len(Matrix_error[:]) + 1), Matrix_error[:], marker='o')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        grid.addWidget(self.toolbar,3, 1)
        grid.addWidget(self.canvas, 2, 1)
        self.setLayout(grid)
        self.canvas.show()
        self.show()
        if osp.exists('./CROSS_VALIDATION'):
            pass
        else:
            os.mkdir('./CROSS_VALIDATION')
        now = datetime.now()
        dt_str = now.strftime("%Y%m%d_%H%M%S")
        File_name = './CROSS_VALIDATION/' + 'CROSS_VALIDATION_' + dt_str
        os.mkdir(File_name)
        f1 = open('./CROSS_VALIDATION/' + 'CROSS_VALIDATION_' + dt_str + '/Error_rate_in_iteration.txt', 'w')
        for i in range(len(Matrix_error)):
            f1.write('Prediction Error Rate in iteration number: ' + str(i) + ' ; ' + str(Matrix_error[i]) + '\n')
        f1.close()
        f2 = open('./CROSS_VALIDATION/' + 'CROSS_VALIDATION_' + dt_str + '/Average_Error_rate.txt', 'w')
        f2.write('Average Prediction Error Rate: ' + str(CV_error) + '\n')
        f2.write('Number of iteration: ' + str(CV_iter))
        f2.close()

class Localization_widget(QWidget):

    def __init__(self,database_in=Current_database,directory_in=Current_directory):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.Select_image_to_update(database_in=self.database_in,directory_in=self.directory_in)

    def Select_image_to_update(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.DDD_run))
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(['Activation', 'Initialization directory', 'Shape of the image',
                                                    'File name',
                                                    'Left upper corner', 'Coordinate system', 'Resolution'])
        self.layout = QVBoxLayout()
        for i in range(len(database_in.DDD_run)):
            if database_in.DDD_run[i] == directory_in.DDD_run:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(database_in.DDD_run[i])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.input_image[i].shape)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(database_in.fileName[i])))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(database_in.origin[i])))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.system[i])))
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(database_in.resolut[i])))
        btn_select_auto = QPushButton('Select to reference --auto--')
        btn_select_auto.clicked.connect(self.Select_reference_auto)
        self.layout.addWidget(btn_select_auto)
        btn_select_man = QPushButton("Select to reference --manually--")
        btn_select_man.clicked.connect(self.Select_reference_man)
        self.layout.addWidget(btn_select_man)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.show()

    def Select_reference_auto(self):
        to_choose = (np.arange(1, len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel1, okPressed1 = QInputDialog.getItem(self, 'Select','Reference image', to_choose, 0, False)
            no_sel2, okPressed2 = QInputDialog.getItem(self, 'Select', 'Image to locate', to_choose, 0, False)
            c_img1 = int(no_sel1) - 1
            c_img2 = int(no_sel2) - 1
            if okPressed1:
                if okPressed2:
                    text_types = ('North', 'South')
                    text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'North or South Hemisphere', text_types,
                                                               0, False)
                    sign = 1
                    if okPressed and text_sel == 'North':
                        sign = 1
                    elif okPressed and text_sel == 'South':
                        sign = -1
                    shape1 = self.database_in.input_image[c_img1].shape
                    res1 = self.database_in.resolut[c_img1]
                    origo1 = self.database_in.origin[c_img1]
                    if self.database_in.system[c_img1] == self.database_in.system[c_img2] or \
                            self.database_in.system[c_img2] == ['None']:
                        self.database_in.resolut[c_img2] = res1
                        self.database_in.origin[c_img2] = np.copy(origo1)
                        self.database_in.system[c_img2] = self.database_in.system[c_img1]
                        if shape1[0] > self.database_in.input_image[c_img2].shape[0] and shape1[1] >\
                                self.database_in.input_image[c_img2].shape[1]:
                            self.database_in.origin[c_img2][0] = origo1[0] + ((shape1[0] - self.database_in.
                                                                               input_image[c_img2].
                                                                               shape[0]) * res1[0] / 2)
                            self.database_in.origin[c_img2][1] = origo1[1] - (sign * (shape1[1] - self.database_in.
                                                                                      input_image[c_img2].
                                                                                      shape[1]) * res1[1] / 2)
                        elif shape1[0] < self.database_in.input_image[c_img2].shape[0] and shape1[1] <\
                                self.database_in.input_image[c_img2].shape[1]:
                            self.database_in.origin[c_img2][0] = origo1[0] - ((shape1[0] - self.database_in.
                                                                               input_image[c_img2].
                                                                               shape[0]) * res1[0] / 2)
                            self.database_in.origin[c_img2][1] = origo1[1] + (sign * (shape1[1] - self.database_in.
                                                                                      input_image[c_img2]
                                                                                      .shape[1]) * res1[1] / 2)

    def Select_reference_man(self):
        to_choose = (np.arange(1, len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Image to locate', to_choose, 0, False)
            c_img = int(no_sel) - 1
            text_types = ('UTM', 'WGS 1984')
            text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Coordinate system', text_types,
                                                       0, False)
            if okPressed and text_sel == 'UTM':
                text_types1 = ('North', 'South')
                text_sel1, okPressed1 = QInputDialog.getItem(self, 'Select', 'North or South Hemisphere', text_types1,
                                                             0, False)
                if okPressed1 and text_sel1 == 'North':
                    to_choose2 = list(range(1, 61))
                    for i in range(len(to_choose2)):
                        to_choose2[i] = str(to_choose2[i])
                    no_sel2, okPressed2 = QInputDialog.getItem(self, 'Select', 'Zone', to_choose2, 0, False)
                    if okPressed2:
                        file_north = open('./Reference_system/UTM_north.txt', 'r')
                        self.database_in.system[c_img] = file_north.readlines(int(no_sel2))
                        file_north.close()
                elif okPressed and text_sel == 'South':
                    to_choose2 = list(range(1, 61))
                    for i in range(len(to_choose2)):
                        to_choose2[i] = str(to_choose2[i])
                    no_sel2, okPressed2 = QInputDialog.getItem(self, 'Select', 'Zone', to_choose2, 0, False)
                    if okPressed2:
                        file_south = open('./Reference_system/UTM_south.txt', 'r')
                        self.database_in.system[c_img] = file_south.readlines(int(no_sel2))
                        file_south.close()

            elif okPressed and text_sel == 'WGS 1984':
                file_wgs = open('./Reference_system/WGS.txt', 'r')
                self.database_in.system[c_img] = file_wgs.readlines()
                file_wgs.close()

            self.database_in.origin[c_img] = [0,0]
            west_value, okPressed_west = QInputDialog.getDouble(self, 'Get float', 'West side of the upper corner:',
                                                                    0, 0, 999999999, 3)
            if okPressed_west:
                self.database_in.origin[c_img][0] = float(west_value)
            north_value, okPressed_north = QInputDialog.getDouble(self,'Get float','North side of the west corner:',
                                                                0, 0, 999999999, 3)
            if okPressed_north:
                self.database_in.origin[c_img][1] = float(north_value)
            self.database_in.resolut[c_img] = [0,0]
            westR_value, okPressedR_west = QInputDialog.getDouble(self, 'Get float',
                                                                'Resolution in east-west direction:',
                                                                0, 0, 999999999, 10)
            if okPressedR_west:
                self.database_in.resolut[c_img][0] = float(westR_value)
            northR_value, okPressedR_north = QInputDialog.getDouble(self,'Get float',
                                                                    'Resolution in north-south direction:',
                                                                0, 0, 999999999, 10)
            if okPressedR_north:
                self.database_in.resolut[c_img][1] = float(northR_value)

class Transformation_UTM_WGS_widget(QWidget):

    def __init__(self,database_in=Current_database,directory_in=Current_directory):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.Transform_image_UTM_WGS(database_in=self.database_in,directory_in=self.directory_in)

    def Transform_image_UTM_WGS(self, database_in, directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.DDD_run))
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(['Activation', 'Initialization directory', 'Shape of the image',
                                                    'File name',
                                                    'Left upper corner', 'Coordinate system', 'Resolution'])
        self.layout = QVBoxLayout()
        for i in range(len(database_in.DDD_run)):
            if database_in.DDD_run[i] == directory_in.DDD_run:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(database_in.DDD_run[i])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.input_image[i].shape)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(database_in.fileName[i])))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(database_in.origin[i])))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.system[i])))
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(database_in.resolut[i])))
        btn_select_trans = QPushButton('Select image to transform --auto--')
        btn_select_trans.clicked.connect(self.Select_iamge_to_trans)
        self.layout.addWidget(btn_select_trans)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.show()

    def Select_iamge_to_trans(self):
        to_choose = (np.arange(1, len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel1, okPressed1 = QInputDialog.getItem(self, 'Select','Reference image', to_choose, 0, False)
            c_img1 = int(no_sel1) - 1
            if okPressed1:
                system_str = str(self.database_in.system[c_img1])
                if system_str[33] == '|':
                    hemisphere = system_str[32]
                    utm_zone = int(system_str[31])
                else:
                    hemisphere = system_str[33]
                    utm_zone = int(system_str[31:33])
                Easting = float(self.database_in.origin[c_img1][0])
                Northing = float(self.database_in.origin[c_img1][1])
                Lon, Lat = self.database_in.DDD_run[c_img1].UTM_WGS(x_utm=Easting,y_utm=Northing,hemisphere=hemisphere,
                                                                    utm_zone=utm_zone)
                self.database_in.origin[c_img1] = [Lon, Lat]

class Transformation_WGS_UTM_widget(QWidget):

    def __init__(self,database_in=Current_database,directory_in=Current_directory):
        super().__init__()
        self.database_in = database_in
        self.directory_in = directory_in
        self.Transform_image_WGS_UTM(database_in=self.database_in,directory_in=self.directory_in)

    def Transform_image_WGS_UTM(self,database_in,directory_in):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(database_in.DDD_run))
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(['Activation', 'Initialization directory', 'Shape of the image',
                                                    'File name',
                                                    'Left upper corner', 'Coordinate system', 'Resolution'])
        self.layout = QVBoxLayout()
        for i in range(len(database_in.DDD_run)):
            if database_in.DDD_run[i] == directory_in.DDD_run:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Active'))
            else:
                self.tableWidget.setItem(i, 0, QTableWidgetItem('Inactive'))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(database_in.DDD_run[i])))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(database_in.input_image[i].shape)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(database_in.fileName[i])))
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(database_in.origin[i])))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(database_in.system[i])))
            self.tableWidget.setItem(i, 6, QTableWidgetItem(str(database_in.resolut[i])))
        btn_select_trans = QPushButton('Select image to transform --auto--')
        btn_select_trans.clicked.connect(self.Select_iamge_to_trans)
        self.layout.addWidget(btn_select_trans)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.show()

    def Select_iamge_to_trans(self):
        to_choose = (np.arange(1, len(self.database_in.DDD_run) + 1)).tolist()
        if len(to_choose) > 0:
            if len(to_choose) > 1:
                for i in range(len(to_choose)):
                    to_choose[i] = str(to_choose[i])
            else:
                to_choose[0] = str(to_choose[0])
            no_sel1, okPressed1 = QInputDialog.getItem(self, 'Select', 'Reference image', to_choose, 0, False)
            c_img1 = int(no_sel1) - 1
            to_choose2 = list(range(1, 61))
            for i in range(len(to_choose2)):
                to_choose2[i] = str(to_choose2[i])
            no_sel2, okPressed2 = QInputDialog.getItem(self, 'Select', 'Zone', to_choose2, 0, False)
            if okPressed1:
                if okPressed2:
                    utm_zone = int(no_sel2)
                Lon = float(self.database_in.origin[c_img1][0])
                Lat = float(self.database_in.origin[c_img1][1])
                Northing, Easting = self.database_in.DDD_run[c_img1].WGS_UTM(lat_new=Lat,lon_new=Lon,utm_zone=utm_zone)
                self.database_in.origin[c_img1] = [Easting, Northing]

class ALL_SET_widgets(QWidget):

    def __init__(self,application=DDD):
        super().__init__()
        self.application = application
        self.All_settings()

    def All_settings(self):
        self.grid = QGridLayout()
        self.grid.setSpacing(1)
        self.grid.addWidget(QLabel("Basic settings"), 1, 1, 1, 2)
        self.image_cutter_line = QLineEdit()
        self.multi_mask_line = QLineEdit()
        self.grid.addWidget(self.multi_mask_line, 2, 2, 1, 1)
        self.grid.addWidget(self.image_cutter_line, 3, 2, 1, 1)
        btn_mask_multi = QPushButton('Multi mask in mask stacking')
        btn_image_cutter = QPushButton('Number of pixels to cut image')
        self.grid.addWidget(btn_mask_multi, 2, 1, 1, 1)
        self.grid.addWidget(btn_image_cutter, 3, 1, 1, 1)
        self.multi_mask_line.setText(str(self.application.multi_mask))
        self.image_cutter_line.setText(str(self.application.image_cutter))
        self.grid.addWidget(QLabel("Import settings"), 4, 1, 1, 2)
        self.crop_line = QLineEdit()
        self.non_value_reduction_line = QLineEdit()
        self.import_data_type = QLineEdit()
        self.left_line = QLineEdit()
        self.upper_line = QLineEdit()
        self.right_line = QLineEdit()
        self.lower_line = QLineEdit()
        self.grid.addWidget(self.crop_line, 5, 2, 1, 1)
        self.grid.addWidget(self.non_value_reduction_line, 6, 2, 1, 1)
        self.grid.addWidget(self.import_data_type, 7, 2, 1, 1)
        self.grid.addWidget(self.left_line, 8, 2, 1, 1)
        self.grid.addWidget(self.upper_line, 9, 2, 1, 1)
        self.grid.addWidget(self.right_line, 10, 2, 1, 1)
        self.grid.addWidget(self.lower_line, 11, 2, 1, 1)
        btn_crop = QPushButton('Crop the image')
        btn_non_val = QPushButton('Change non value data to 0')
        btn_data_type = QPushButton('Data format')
        btn_left = QPushButton('Left edge')
        btn_upper = QPushButton('Upper edge')
        btn_right = QPushButton('Right edge')
        btn_lower = QPushButton('Lower edge')
        self.grid.addWidget(btn_crop, 5, 1, 1, 1)
        self.grid.addWidget(btn_non_val, 6, 1, 1, 1)
        self.grid.addWidget(btn_data_type, 7, 1, 1, 1)
        self.grid.addWidget(btn_left, 8, 1, 1, 1)
        self.grid.addWidget(btn_upper, 9, 1, 1, 1)
        self.grid.addWidget(btn_right, 10, 1, 1, 1)
        self.grid.addWidget(btn_lower, 11, 1, 1, 1)
        self.crop_line.setText(str(self.application.crop_))
        self.non_value_reduction_line.setText(str(self.application.non_val_reduction))
        self.import_data_type.setText(str(self.application.data_type))
        self.left_line.setText(str(self.application.left))
        self.upper_line.setText(str(self.application.upper))
        self.right_line.setText(str(self.application.right))
        self.lower_line.setText(str(self.application.lower))
        self.grid.addWidget(QLabel("Mask stacking settings"), 12, 1, 1, 2)
        self.masks_in_stack_line = QLineEdit()
        self.limit_loop_line = QLineEdit()
        self.grid.addWidget(self.masks_in_stack_line, 13, 2, 1, 1)
        self.grid.addWidget(self.limit_loop_line, 15, 2, 1, 1)
        btn_mask_stack = QPushButton('Masks size in stacking')
        btn_masks_reset = QPushButton('Reset masks')
        btn_limit_loop = QPushButton('Limit of pixels in row/column')
        self.grid.addWidget(btn_mask_stack, 13, 1, 1, 1)
        self.grid.addWidget(btn_masks_reset, 14, 1, 1, 1)
        self.grid.addWidget(btn_limit_loop, 15, 1, 1, 1)
        self.masks_in_stack_line.setText(str(self.application.masks))
        self.limit_loop_line.setText(str(self.application.max_in_loop))
        self.grid.addWidget(QLabel("Artifacts impact settings"), 16, 1, 1, 2)
        self.Impact_line = QLineEdit()
        self.grid.addWidget(self.Impact_line, 17, 2, 1, 1)
        self.Impact_line.setText(str(self.application.max_impact_px))
        btn_Imp = QPushButton('Impact of artifacts')
        btn_Imp_reset = QPushButton('Set to default')
        self.grid.addWidget(btn_Imp, 17, 1, 1, 1)
        self.grid.addWidget(btn_Imp_reset, 18, 1, 1, 1)
        self.grid.addWidget(QLabel("Decorelation settings"), 19, 1, 1, 2)
        self.dec_max_rate_line = QLineEdit()
        self.grid.addWidget(self.dec_max_rate_line, 20, 2, 1, 1)
        self.dec_max_rate_line.setText(str(self.application.max_rate))
        btn_dec_MR = QPushButton('Maximum rate')
        self.grid.addWidget(btn_dec_MR, 20, 1, 1, 1)
        self.grid.addWidget(QLabel("Machine learning settings"), 21, 1, 1, 2)
        self.iter_no_line = QLineEdit()
        self.random_state_line = QLineEdit()
        self.regul_line = QLineEdit()
        self.grid.addWidget(self.iter_no_line, 22, 2, 1, 1)
        self.grid.addWidget(self.random_state_line, 23, 2, 1, 1)
        self.grid.addWidget(self.regul_line, 24, 2, 1, 1)
        self.iter_no_line.setText(str(self.application.n_iter))
        self.random_state_line.setText(str(self.application.random_state))
        self.regul_line.setText(str(self.application.lambda_))
        btn_iter_no = QPushButton('Number of iterations')
        btn_random = QPushButton('Seed to random no generator')
        btn_reg = QPushButton('L2 regularization parameter')
        self.grid.addWidget(btn_iter_no, 22, 1, 1, 1)
        self.grid.addWidget(btn_random, 23, 1, 1, 1)
        self.grid.addWidget(btn_reg, 24, 1, 1, 1)

        self.grid.addWidget(QLabel("Parametrization settings"), 1, 3, 1, 2)
        self.Trend_detection_line = QLineEdit()
        self.Var_line = QLineEdit()
        self.STD_dev_line = QLineEdit()
        self.Defo_line = QLineEdit()
        self.Semi_Var_line = QLineEdit()
        self.Gauss_line = QLineEdit()
        self.Depths_line = QLineEdit()
        self.grid.addWidget(self.Trend_detection_line, 2, 4, 1, 1)
        self.grid.addWidget(self.Var_line, 3, 4, 1, 1)
        self.grid.addWidget(self.STD_dev_line, 4, 4, 1, 1)
        self.grid.addWidget(self.Defo_line, 5, 4, 1, 1)
        self.grid.addWidget(self.Semi_Var_line, 6, 4, 1, 1)
        self.grid.addWidget(self.Gauss_line, 7, 4, 1, 1)
        self.grid.addWidget(self.Depths_line, 8, 4, 1, 1)
        btn_TD = QPushButton('Trend detection ')
        btn_V = QPushButton('Variance from px')
        btn_STD = QPushButton('Standard deviation')
        btn_Def = QPushButton('Deformation value')
        btn_SVV = QPushButton('Semi-variance')
        btn_G = QPushButton('Differential Gaussian')
        btn_Dep = QPushButton('Depths class to detect')
        btn_Dep_reset = QPushButton('Reset depths')
        self.grid.addWidget(btn_TD, 2, 3, 1, 1)
        self.grid.addWidget(btn_V, 3, 3, 1, 1)
        self.grid.addWidget(btn_STD, 4, 3, 1, 1)
        self.grid.addWidget(btn_Def, 5, 3, 1, 1)
        self.grid.addWidget(btn_SVV, 6, 3, 1, 1)
        self.grid.addWidget(btn_G, 7, 3, 1, 1)
        self.grid.addWidget(btn_Dep, 8, 3, 1, 1)
        self.grid.addWidget(btn_Dep_reset, 9, 3, 1, 1)
        self.Depths_line.setText(str(self.application.depths))
        self.Trend_detection_line.setText(str(self.application.trend_parameter))
        self.Var_line.setText(str(self.application.var_parameter))
        self.STD_dev_line.setText(str(self.application.standard_dev))
        self.Defo_line.setText(str(self.application.deformation_parameter))
        self.Semi_Var_line.setText(str(self.application.semi_var_parameter))
        self.Gauss_line.setText(str(self.application.gaussian_par))
        self.Def_type_line = QLineEdit()
        self.grid.addWidget(self.Def_type_line, 10, 4, 1, 1)
        btn_DefT = QPushButton('Subsidence (-1) / Uplift (+1)')
        self.grid.addWidget(btn_DefT, 10, 3, 1, 1)
        self.Def_type_line.setText(str(self.application.deformation_type))
        self.grid.addWidget(QLabel("Jumps removing settings"), 11, 3, 1, 2)
        self.rm_Mask_size_line = QLineEdit()
        self.rm_Density_line = QLineEdit()
        self.rm_Seed_line = QLineEdit()
        self.rm_Min_pop_line = QLineEdit()
        self.grid.addWidget(self.rm_Mask_size_line, 12, 4, 1, 1)
        self.grid.addWidget(self.rm_Density_line, 14, 4, 1, 1)
        self.grid.addWidget(self.rm_Seed_line, 16, 4, 1, 1)
        self.grid.addWidget(self.rm_Min_pop_line, 18, 4, 1, 1)
        self.rm_Mask_size_line.setText(str(self.application.rm_mask_size))
        self.rm_Density_line.setText(str(self.application.rm_density_))
        self.rm_Seed_line.setText(str(self.application.rm_seed_))
        self.rm_Min_pop_line.setText(str(self.application.rm_min_population_factor))
        btn_MS = QPushButton('Mask size (RJ)')
        btn_MS_reset = QPushButton('Mask size (RJ) reset')
        btn_HD = QPushButton('Histogram density')
        btn_HD_reset = QPushButton('Histogram density reset')
        btn_SD = QPushButton('Histogram seed')
        btn_SD_reset = QPushButton('Histogram seed reset')
        btn_MP = QPushButton('Minimum no of pixel to consider')
        btn_MP_reset = QPushButton('Min. no of pixel- reset')
        self.grid.addWidget(btn_MS, 12, 3, 1, 1)
        self.grid.addWidget(btn_MS_reset, 13, 3, 1, 1)
        self.grid.addWidget(btn_HD, 14, 3, 1, 1)
        self.grid.addWidget(btn_HD_reset, 15, 3, 1, 1)
        self.grid.addWidget(btn_SD, 16, 3, 1, 1)
        self.grid.addWidget(btn_SD_reset, 17, 3, 1, 1)
        self.grid.addWidget(btn_MP, 18, 3, 1, 1)
        self.grid.addWidget(btn_MP_reset, 19, 3, 1, 1)

        btn_mask_multi.clicked.connect(self.Get_multi_mask)
        btn_image_cutter.clicked.connect(self.Get_image_cuter)
        btn_crop.clicked.connect(self.Get_crop)
        btn_left.clicked.connect(self.Get_int_left)
        btn_upper.clicked.connect(self.Get_int_upper)
        btn_right.clicked.connect(self.Get_int_right)
        btn_lower.clicked.connect(self.Get_int_lower)
        btn_non_val.clicked.connect(self.Get_nan_value)
        btn_data_type.clicked.connect(self.Get_data_type)
        btn_masks_reset.clicked.connect(self.Get_Reset_masks_stacking)
        btn_mask_stack.clicked.connect(self.Get_Select_mask)
        btn_limit_loop.clicked.connect(self.Get_loop_limit)
        btn_Imp_reset.clicked.connect(self.GetReset_IMP)
        btn_Imp.clicked.connect(self.Get_IMP)
        btn_dec_MR.clicked.connect(self.Get_double_max_rate)
        btn_TD.clicked.connect(self.Get_TD)
        btn_V.clicked.connect(self.Get_V)
        btn_STD.clicked.connect(self.Get_STD)
        btn_Def.clicked.connect(self.Get_Def)
        btn_SVV.clicked.connect(self.Get_SW)
        btn_G.clicked.connect(self.Get_G)
        btn_Dep.clicked.connect(self.Get_Dep)
        btn_Dep_reset.clicked.connect(self.Get_Reset_dep)
        btn_DefT.clicked.connect(self.Get_type_def)
        btn_MS_reset.clicked.connect(self.Get_Reset_MS)
        btn_MS.clicked.connect(self.Get_MS)
        btn_HD_reset.clicked.connect(self.Get_Reset_HD)
        btn_HD.clicked.connect(self.Get_HD)
        btn_SD_reset.clicked.connect(self.Get_Reset_SD)
        btn_SD.clicked.connect(self.Get_SD)
        btn_MP_reset.clicked.connect(self.Get_Reset_MP)
        btn_MP.clicked.connect(self.Get_MP)
        btn_reg.clicked.connect(self.Get_Regular)
        btn_iter_no.clicked.connect(self.Get_n_ite)
        btn_random.clicked.connect(self.Get_Random)

        btn_finish = QPushButton('Finish')
        btn_finish.clicked.connect(self.Finish)
        self.grid.addWidget(btn_finish,25, 1, 1, 1)
        self.setLayout(self.grid)
        self.show()

    def Get_multi_mask(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.multi_mask = True
            self.multi_mask_line.setText(str(self.application.multi_mask))

        elif okPressed and text_sel == 'No':
            self.application.multi_mask = False
            self.multi_mask_line.setText(str(self.application.multi_mask))

    def Get_image_cuter(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.image_cutter = value
            self.image_cutter_line.setText(str(self.application.image_cutter))

    def Get_crop(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.crop_ = True
            self.crop_line.setText(str(self.application.crop_))
        elif okPressed and text_sel == 'No':
            self.application.crop_ = False
            self.crop_line.setText(str(self.application.crop_))

    def Get_int_left(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.left = value
            self.left_line.setText(str(self.application.left))

    def Get_int_upper(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.upper = value
            self.upper_line.setText(str(self.application.upper))

    def Get_int_right(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.right = value
            self.right_line.setText(str(self.application.right))

    def Get_int_lower(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.lower = value
            self.lower_line.setText(str(self.application.lower))

    def Get_nan_value(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.non_val_reduction = True
            self.non_value_reduction_line.setText(str(self.application.non_val_reduction))

        elif okPressed and text_sel == 'No':
            self.application.non_val_reduction = False
            self.non_value_reduction_line.setText(str(self.application.non_val_reduction))

    def Get_data_type(self):
        data_types = ('txt', 'csv', 'image','GeoTIFF')
        data_type_sel, okPressed = QInputDialog.getItem(self, 'Select data type','list of data types', data_types, 0,
                                                        False)
        if okPressed and data_type_sel == 'txt':
            self.application.data_type = data_type_sel
            self.import_data_type.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'csv':
            self.application.data_type = data_type_sel
            self.import_data_type.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'image':
            self.application.data_type = data_type_sel
            self.import_data_type.setText(str(self.application.data_type))
        elif okPressed and data_type_sel == 'GeoTIFF':
            self.application.data_type = data_type_sel
            self.import_data_type.setText(str(self.application.data_type))

    def Get_Reset_masks_stacking(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.masks_in_stack_line.setText(str('0'))
            self.application.masks = []

    def Get_Select_mask(self):
        value_types = []
        value_types_int = []
        for i in range(2,50):
            value_types_int.append(i * 2 - 1)
            value_types.append(str(i * 2 - 1))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Masks', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_int)):
                if value_types_int[i] == int(value_sel):
                    if len(self.application.masks) == 0:
                        self.application.masks.append(int(value_sel))
                        self.masks_in_stack_line.setText(str(self.application.masks))
                    elif len(self.application.masks) == 1:
                        if self.application.masks != int(value_sel):
                            self.application.masks.append(int(value_sel))
                            self.masks_in_stack_line.setText(str(self.application.masks))
                    elif len(self.application.masks) > 1:
                        check = 0
                        for j in range(len(self.application.masks)):
                            if self.application.masks[j] == int(value_sel):
                                check += 1
                        if check == 0:
                            self.application.masks.append(int(value_sel))
                            self.masks_in_stack_line.setText(str(self.application.masks))

    def Get_loop_limit(self):
        value, okPressed = QInputDialog.getInt(self, 'Get integer','Value:', 0, 0, 999999, 1)
        if okPressed:
            self.application.max_in_loop = value
            self.limit_loop_line.setText(str(self.application.max_in_loop))

    def GetReset_IMP(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.max_impact_px = 10
            self.Impact_line.setText(str(self.application.max_impact_px))

    def Get_IMP(self):
        value_types = ('3px', '4px', '5px', '7px', '10px', '15px', '20px', '25px', '30px')
        value_types_float = [3,4,5,7,10,15,20,25,30]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Impact radius', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel[:-2]):
                    self.application.max_impact_px = value_types_float[i]
                    self.Impact_line.setText(str(self.application.max_impact_px))

    def Get_double_max_rate(self):
        value, okPressed = QInputDialog.getDouble(self, 'Get max rate','Value:', 0, 0, 1, 2)
        if okPressed:
            self.application.max_rate = value
            self.dec_max_rate_line.setText(str(value))

    def Get_TD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.trend_parameter = True
            self.Trend_detection_line.setText(str(self.application.trend_parameter))
        elif okPressed and text_sel == 'No':
            self.application.trend_parameter = False
            self.Trend_detection_line.setText(str(self.application.trend_parameter))

    def Get_V(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.var_parameter = True
            self.Var_line.setText(str(self.application.var_parameter))
        elif okPressed and text_sel == 'No':
            self.application.var_parameter = False
            self.Var_line.setText(str(self.application.var_parameter))

    def Get_STD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.standard_dev = True
            self.STD_dev_line.setText(str(self.application.standard_dev))
        elif okPressed and text_sel == 'No':
            self.application.standard_dev = False
            self.STD_dev_line.setText(str(self.application.standard_dev))

    def Get_Def(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.deformation_parameter = True
            self.Defo_line.setText(str(self.application.deformation_parameter))
        elif okPressed and text_sel == 'No':
            self.application.deformation_parameter = False
            self.Defo_line.setText(str(self.application.deformation_parameter))

    def Get_SW(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.semi_var_parameter = True
            self.Semi_Var_line.setText(str(self.application.semi_var_parameter))
        elif okPressed and text_sel == 'No':
            self.application.semi_var_parameter = False
            self.Semi_Var_line.setText(str(self.application.semi_var_parameter))

    def Get_G(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select', 'Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.gaussian_par = True
            self.Gauss_line.setText(str(self.application.gaussian_par))
        elif okPressed and text_sel == 'No':
            self.application.gaussian_par = False
            self.Gauss_line.setText(str(self.application.gaussian_par))

    def Get_Reset_dep(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.depths = []
            self.Depths_line.setText(str('0'))

    def Get_Dep(self):
        value_types = ('+0.200', '+0.100', '+0.050', '+0.040', '+0.030', '+0.020', '+0.015', '+0.010', '+0.005',
                       '+0.000', '-0.005', '-0.010', '-0.015', '-0.020', '-0.030', '-0.040', '-0.050', '-0.100',
                       '-0.200')
        value_types_float = [0.2,0.1,0.05,0.04,0.03,0.02,0.015,0.01,0.005,0,(-0.005),(-0.01),(-0.015),(-0.02),(-0.03),
                             (-0.04),(-0.05),(-0.1),(-0.2)]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Masks', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    if len(self.application.depths) == 0:
                        self.application.depths.append(float(value_sel))
                        self.Depths_line.setText(str(self.application.depths))
                    elif len(self.application.depths) == 1:
                        if self.application.depths != float(value_sel):
                            self.application.depths.append(float(value_sel))
                            self.Depths_line.setText(str(self.application.depths))
                    elif len(self.application.depths) > 1:
                        check = 0
                        for j in range(len(self.application.depths)):
                            if self.application.depths[j] == float(value_sel):
                                check += 1
                        if check == 0:
                            self.application.depths.append(float(value_sel))
                            self.Depths_line.setText(str(self.application.depths))

    def Get_type_def(self):
        text_types = ('Subsidence', 'Uplift')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Subsidence or Uplift', text_types, 0, False)
        if okPressed and text_sel == 'Subsidence':
            self.application.deformation_type = (-1)
            self.Def_type_line.setText(str(self.application.deformation_type))

        elif okPressed and text_sel == 'Uplift':
            self.application.deformation_type = 1
            self.Def_type_line.setText(str(self.application.deformation_type))

    def Get_Reset_MS(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 31
            self.rm_Mask_size_line.setText(str(self.application.rm_mask_size))

    def Get_MS(self):
        value_types = []
        value_types_float = []
        for i in range(10,50):
            value_types_float.append(i * 2 - 1)
            value_types.append(str(i * 2 - 1))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','Mask size to find the class for pixel', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_mask_size = value_types_float[i]
                    self.rm_Mask_size_line.setText(str(self.application.rm_mask_size))

    def Get_Reset_HD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 10
            self.rm_Density_line.setText(str(self.application.rm_mask_size))

    def Get_HD(self):
        value_types = ('5','6','7','8','9','10','11','12','13','14','15')
        value_types_float = [5,6,7,8,9,10,11,12,13,14,15]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The density of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_density_ = value_types_float[i]
                    self.rm_Density_line.setText(str(self.application.rm_density_))

    def Get_Reset_SD(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 3
            self.rm_Seed_line.setText(str(self.application.rm_mask_size))

    def Get_SD(self):
        value_types = ('1','2','3','4','5','6')
        value_types_float = [1,2,3,4,5,6]
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The seed of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_seed_ = value_types_float[i]
                    self.rm_Seed_line.setText(str(self.application.rm_seed_))

    def Get_Reset_MP(self):
        text_types = ('Yes', 'No')
        text_sel, okPressed = QInputDialog.getItem(self, 'Select','Yes or No', text_types, 0, False)
        if okPressed and text_sel == 'Yes':
            self.application.rm_mask_size = 0.01
            self.rm_Min_pop_line .setText(str(self.application.rm_mask_size))

    def Get_MP(self):
        value_types = []
        value_types_float = []
        for i in range(1,50):
            value_types_float.append(i)
            value_types.append(str(i))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The seed of histogram', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.rm_min_population_factor = value_types_float[i] * 0.01
                    self.rm_Min_pop_line.setText(str(self.application.rm_min_population_factor))

    def Get_Regular(self):
        value_types = []
        value_types_float = []
        for i in range(-10,10):
            value_types_float.append(10 ** i)
            value_types.append(str(10 ** i))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The L2 reg. factor', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.lambda_ = value_types_float[i]
                    self.regul_line.setText(str(self.application.lambda_))

    def Get_n_ite(self):
        value_types = []
        value_types_float = []
        for i in range(1,300):
            value_types_float.append(i)
            value_types.append(str(i))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The no of iterations', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.n_iter = value_types_float[i]
                    self.iter_no_line.setText(str(self.application.n_iter))

    def Get_Random(self):
        value_types = []
        value_types_float = []
        for i in range(-5,5):
            value_types_float.append(10 ** i)
            value_types.append(str(10 ** i))
        value_sel, okPressed = QInputDialog.getItem(self, 'Select','The seed for random state', value_types, 0, False)
        if okPressed:
            for i in range(len(value_types_float)):
                if value_types_float[i] == float(value_sel):
                    self.application.random_state = value_types_float[i]
                    self.random_state_line.setText(str(self.application.random_state))

    def Finish(self):
        Status_toolbar = QLineEdit()
        self.grid.addWidget(Status_toolbar, 25, 2, 1, 1)
        self.setLayout(self.grid)
        Status_toolbar.setText("New setting applied!")
        self.show()
		
"""-------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------- Run APP ------------------------------------------------------"""
"""-------------------------------------------------------------------------------------------------------------"""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI_panel()
    sys.exit(app.exec_())