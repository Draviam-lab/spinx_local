# SpinX-Local

```
Author: David Dang, PhD at Draviam Lab (School of Biological and Chemical Sciences, Queen Mary University of London) and Sastry Group (Department of Computer Science, King's College London)
```

SpinX Software is designed to study dynamics of subcellular structures in 3D time-resolved movies (e.g. spindle and cell cortex). The SpinX framework is composed of four modules:

![spinx_overview](https://raw.githubusercontent.com/Draviam-lab/spinx_local/ff9e74576258645adc042245b3cde1de7cd98bd0/assets/spinx_overview.png)

 | Name								| Type				| Description	| Output
 |----------------------------|------------- |------------ |-------------
 | spinx_ai_module_pred_spindle				| Folder       | SpinX AI prediction module to segment spindle | Segmentation mask of spindles
 | spinx_ai_module_pred_cell_cortex           		            | Folder       | SpinX AI prediction module to segment cell cortex | Segmentation mask of cell cortex
 | spinx_3d_modelling_module | Folder       |  SpinX 3D reconstruction and modelling module | Figures and measurement table .csv)

General notes:

SpinX supports as input list of files (.png, .tif) or multi-page .OME-TIFF files. Each module contains a requirement.txt with detailed information on used dependancies. SpinX requires the installation of Mask R-CNN (Matterport: https://github.com/matterport/Mask_RCNN).

```
Folder Structure for SpinX AI prediction module to segment spindle and cell cortex:
├── input									# List of files (Example images)
├── input_ome_tiff				# OME-TIFF (Example multi-page OME-TIFF)
├── mrcnn          				# Mask R-CNN
├── samples      					# Information on trained neural network
├── output								# This folder will be generated automatically
```

``` 
Folder Structure for SpinX 3D reconstruction and modelling module:
├── input									# List of files (Example images)
		├── cell_cortex				# Segmentation mask of cell cortex (predicted by the previous module)
		├── spindle						# Segmentation mask of spindle (predicted by the previous module)
├── input_ome_tiff				# OME-TIFF (Example multi-page OME-TIFF)
		├── cell_cortex				# Segmentation mask of cell cortex (predicted by the previous module)
		├── spindle						# Segmentation mask of spindle (predicted by the previous module)
├── font          				# Font file for plotting
├── output								# This folder will be generated automatically.

```

