#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2024 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import torch

join=os.path.join

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel

def init_folds():
    from monai.networks.nets import SwinUNETR, UNETR, DynUNet

    model_fold0 = SwinUNETR(
        feature_size= 48,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        spatial_dims= 3,
        use_checkpoint= False,
        use_v2= False,
    )

    model_fold1 = UNETR(
        feature_size= 16,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        # spatial_dims: 3
        hidden_size= 768,
        mlp_dim= 3072,
        num_heads= 12,
        proj_type= "conv",
        norm_name= "instance",
        res_block= True,
        dropout_rate= 0.0,
        # use_checkpoint: True
    )

    model_fold2 = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size= [3, [1, 1, 3], 3, 3],
        strides= [1, 2, 2, 1],
        upsample_kernel_size= [2, 2, 1],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision= False,
        deep_supr_num= 1,
        res_block= False,
    )

    model_fold3 = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size= [3, [1, 1, 3], 3, 3],
        strides= [1, 2, 2, 1],
        upsample_kernel_size= [2, 2, 1],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision= False,
        deep_supr_num= 1,
        res_block= False,
    )
    
    model_fold4 = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size= [3, [1, 1, 3], 3, 3],
        strides= [1, 2, 2, 1],
        upsample_kernel_size= [2, 2, 1],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision= False,
        deep_supr_num= 1,
        res_block= False,
    )

    return model_fold0, model_fold1, model_fold2, model_fold3, model_fold4

def ensemble_apply(modelObj, data: dict):
    from dafne_dl.interfaces import WrongDimensionalityError
    import numpy as np
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        CastToTyped,
        ScaleIntensityd,
        Invertd,
        Activationsd,
        CopyItemsd,
        AsDiscreted,
    )
    from monai.inferers import sliding_window_inference
    import torch

    # Parameters
    roi_size = (128,128,128)
    sw_batch_size = 1

    if len(data['image'].shape) != 3:
        raise WrongDimensionalityError("Input image must be 3D")

    if modelObj.device.type == 'cpu':
        device = 'cpu'
    else:
        if modelObj.device.index is not None:
            device = modelObj.device.index
        else:
            device = 0

    image = data['image']
    affine=data['affine']

    image = np.expand_dims(image, axis=0)

    # Before transforms
    print("Raw image max:", image.max())
    print("Raw image min:", image.min())

    torch_data = {
        "image": image.astype(np.float32),
        "image_meta_dict": {
            "original_channel_dim": 0,
            "spatial_shape": image.shape[1:],
            "affine": affine,
        }
    }

    print("Running model...")

    # Define data transforms
    transforms = Compose(
        [
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"), align_corners=True),
            CastToTyped(keys=["image"], dtype= np.float32),
            ScaleIntensityd(keys=["image"]),
            CastToTyped(keys=["image"], dtype= np.float32),
        ]
    )

    post_pred=Compose(
        [
            Invertd(
                keys="pred",
                transform=transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            Activationsd(keys="pred", softmax=False, sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            CopyItemsd(keys="pred", times=1, names="pred_final"),
        ]
    )

    input_data = transforms(torch_data)
    # After transforms
    print("Transformed input data shape:", input_data['image'].shape)
    print("Transformed input max:", input_data['image'].max())
    print("Transformed input min:", input_data['image'].min())

    input_tensor = torch.tensor(input_data['image']).to(device).unsqueeze(0) #torch.tensor(input_data['image']).to(device).unsqueeze(0)

    print("Input tensor shape:", input_tensor.shape) 

    models=[]

    for ii in range(5):

        model = modelObj.model[ii]
        model.eval()
        models.append(model)

    print("Model loaded")

    # Calculate model output (decrease overlap parameter for faster but less accurate results)
    with torch.no_grad():

        seg_all=[]

        for ii, model_load in enumerate(models):

            val_output = sliding_window_inference(
                input_tensor, roi_size, sw_batch_size, model_load, mode="gaussian", overlap=0.8
            )
            print("Model applied")
            # Model output
            print("Model output shape:", val_output.shape)
            print("Model output max:", val_output.max())
            print("Model output min:", val_output.min())
            print("image meta dict:", input_data['image_meta_dict'])

            # Pass both the prediction (val_output) and metadata (image_meta_dict) to post_pred
            post_pred_input = {
                "pred": val_output,
                "pred_meta_dict": {
                    "spatial_shape": input_data['image'].shape[1:],  
                    "affine": affine 
                },
                "image_meta_dict": torch_data["image_meta_dict"] 
            }
            val_outputs_post = post_pred(post_pred_input)
            # Postprocessing
            print("Postprocessed output max:", val_outputs_post["pred"].max())
            print("Postprocessed output min:", val_outputs_post["pred"].min())
            val_outputs_convert = val_outputs_post['pred_final'].cpu().numpy() 
            seg = np.squeeze(val_outputs_convert[0, 1, ...])  
            print("seg max value:", np.max(seg))
            print("seg min value:", np.min(seg))

            seg_all.append(seg)

        # Ensemble predictions
        seg_all_tensor = torch.stack([torch.tensor(seg) for seg in seg_all]).to(device)  
        summed_voxel = torch.sum(seg_all_tensor, dim=0)
        ensemble = (summed_voxel > 2).cpu().numpy()  
        print("Ensemble shape:", ensemble.shape) 

    return {
        'CHP': ensemble.astype(np.uint8),
    }


def ensemble_incremental_learning(modelObj, trainingData: dict, trainingOutputs, bs=1, minTrainImages=5):
    try:
        import dafne_dl.common.preprocess_train as pretrain 
        from dafne_dl.labels.chp import inverse_labels 
    except ModuleNotFoundError:
        import dl.common.preprocess_train as pretrain 
        from dl.labels.chp import inverse_labels 

    import time
    import math
    import pickle as pkl
    from tqdm import tqdm
    import torch
    import torch.distributed as dist
    import torchio.transforms as torchio_transforms
    from dafne.utils import compressed_pickle
    import monai
    import monai.transforms as monai_transforms
    from monai import losses
    from monai.data import DataLoader, CacheDataset
    from monai.metrics import DiceMetric
    from monai.utils import set_determinism
    from monai.inferers import sliding_window_inference
    from monai.data import decollate_batch

    try:
        np
    except:
        import numpy as np

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    torch.cuda.set_device(device)

    set_determinism(seed=0)
    
    # validation
    def validation(epoch_iterator_val, global_step, model_, device, patch):

        patch = patch
        softmax = True
        output_classes = 2
        overlap_ratio = 0.8

        dice_metric = DiceMetric(include_background=False, reduction="mean")


        # post transforms
        if softmax:
            post_pred = monai_transforms.Compose(
                [monai_transforms.EnsureType(), monai_transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
            )
            post_label = monai_transforms.Compose([monai_transforms.EnsureType(), monai_transforms.AsDiscrete(to_onehot=output_classes)])
        else:
            post_pred = monai_transforms.Compose(
                [monai_transforms.EnsureType(), monai_transforms.Activations(sigmoid=True), monai_transforms.AsDiscrete(threshold=0.5)]
            )

        model_.eval()
        dice_vals = list()

        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):

                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                val_outputs = sliding_window_inference(val_inputs, patch, 4, model_, overlap=overlap_ratio)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                #Dice
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)

                epoch_iterator_val.set_description(
                
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice) 
                )
            dice_metric.reset()

        mean_dice_val = np.mean(dice_vals)

        metrics_values=mean_dice_val
        
        return metrics_values  

    MODEL_RESOLUTION = np.array([1.0, 1.0, 1.0])
    MODEL_SIZE = (128,128,128)
    BATCH_SIZE = bs
    BAND = 64
    MIN_TRAINING_IMAGES = minTrainImages


    t = time.time()

    train_transforms = monai_transforms.Compose(
        [
            monai_transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            monai_transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), align_corners=True),
            monai_transforms.CastToTyped(keys=["image"], dtype= np.float32),
            monai_transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            monai_transforms.EnsureTyped(keys=["image", "label"]),
            monai_transforms.RandBiasFieldd(keys=["image"], degree=3, prob=0.15),
            monai_transforms.RandGibbsNoised(keys=["image"], prob=0.15),
            torchio_transforms.RandomMotion(
                include=["image"], 
                label_keys=["label"], 
                degrees=10, 
                translation=10, 
                num_transforms=2, 
                image_interpolation=("linear"), 
                p=0.3
            ),
            torchio_transforms.RandomElasticDeformation(
                include=["image"], 
                label_keys=["label"], 
                num_control_points= 7,
                max_displacement= 7.5,
                locked_borders= 2,
                image_interpolation=("linear"),
                label_interpolation=("nearest"),
                p=0.3
            ),
            torchio_transforms.RandomGhosting(
                include=["image"], 
                label_keys=["label"], 
                num_ghosts= 10,
                axes= [0, 1, 2],
                intensity= [0.5, 1],
                restore= 0.2,
                p= 0.3
            ),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=0),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=1),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=2),
            monai_transforms.RandRotated(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.15, range_x=0.3, range_y=0.3, range_z=0.3),
            monai_transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
            monai_transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=MODEL_SIZE,
                pos= 1.0,
                neg= 0.0    
            ),
            monai_transforms.SpatialPadd(keys=["image", "label"], mode=('reflect', 'constant'), spatial_size=MODEL_SIZE),
            monai_transforms.CastToTyped(keys=["image", "label"], dtype= (np.float32, np.uint8)),
        ]
    )

    val_transforms = monai_transforms.Compose(
        [
            monai_transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            monai_transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), align_corners=True),
            monai_transforms.CastToTyped(keys=["image"], dtype= np.float32),
            monai_transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            monai_transforms.CastToTyped(keys=["image", "label"], dtype= (np.float32, np.uint8)),
        ]
    )

    image_list, mask_list = pretrain.common_input_process_ensemble(inverse_labels, MODEL_RESOLUTION, trainingData, trainingOutputs)

    print('Done. Elapsed', time.time() - t)
    nImages = len(image_list)

    if nImages < MIN_TRAINING_IMAGES:
        print("Not enough images for training")
        return

    print('Weight calculation')
    t = time.time()

    train_files=[]
    validation_files=[]

    jj=math.ceil(3*len(image_list)/4)
    
    for kk in range(len(image_list)):

        image=image_list[kk]
        seg=mask_list[kk]

        if kk>jj:
            validation_files.append({"image": image, "label": seg})
        else:
            train_files.append({"image": image, "label": seg})
    
    print('Done. Elapsed', time.time() - t)

    print(f'Incremental learning for Choroid Plexus with {nImages} images')
    t = time.time()
    
    train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=float(torch.cuda.device_count()) / 4.0,
            num_workers=8,
            progress=False,
        )

    val_ds = CacheDataset(
        data=validation_files,
        transform=val_transforms,
        cache_rate=float(torch.cuda.device_count()) / 4.0,
        num_workers=2,
        progress=False,
    )

    train_loader = DataLoader(train_ds, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, num_workers=2, batch_size=BATCH_SIZE, shuffle=False)

    torch.backends.cudnn.benchmark = True

    if modelObj.device.type == 'cpu':
        device = 'cpu'
    else:
        if modelObj.device.index is not None:
            device = modelObj.device.index
        else:
            device = 0

    loss_function=losses.DiceCELoss(include_background=False,to_onehot_y=2, sigmoid=False)

    max_iterations= 10000
    num_iterations_per_validation= 100
    num_epochs_per_validation = num_iterations_per_validation // len(train_files)
    num_epochs_per_validation = max(num_epochs_per_validation, 1)
    num_epochs = num_epochs_per_validation * (max_iterations // num_iterations_per_validation)

    state_dicts=[]

    for ii in range(5):
        
        print("Model load")
        model_ = modelObj.model[ii]

        optimizer = torch.optim.AdamW(model_.parameters(), lr=0.0001, weight_decay= 1.0e-05) 

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("num_epochs", num_epochs)
            print("num_epochs_per_validation", num_epochs_per_validation)

        # training
        global_step = 0
        dice_val_best = 0.0
        global_step_best = 0

        while global_step < max_iterations: 

            model_.train()
            epoch_loss = 0
            step = 0
            epoch_iterator = tqdm(
                train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):

                step += 1
                x, y = (batch["image"].to(device), batch["label"].to(device))
                logit_map = model_(x)
                loss = loss_function(logit_map, y) 
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
                )
                if (
                    global_step % num_iterations_per_validation == 0 and global_step != 0
                ) or global_step == max_iterations:
                    epoch_iterator_val = tqdm(
                        val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                    )
                    
                    dice_val=validation(epoch_iterator_val, global_step, model_, device, MODEL_SIZE)
                    epoch_loss /= step
                    
                    if (dice_val > dice_val_best) : 
                        
                        dice_val_best = dice_val
                        global_step_best = global_step

                        last_good_weights = compressed_pickle.dumps(model_.state_dict()) 

                        print(
                            f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best:.4f},  Current Avg. Dice: {dice_val:.4f} \n"
                        )

                global_step += 1

        new_state_dict = compressed_pickle.loads(last_good_weights)
        state_dicts.append(new_state_dict)

    print('Done. Elapsed', time.time() - t)  

    for ii in range(0, len(state_dicts)):
        modelObj.model[ii].load_state_dict(state_dicts[ii])


metadata = {
    'dependencies': {
        'monai': 'monai = 1.4.0',
        'torchio': 'torchio = 0.20.1',
    },
    'description': 'This model segments Choroid Plexus from T1-w MRI images based on ASCHOPLEX.',
}

generate_convert(model_id='e2bb676f-6e8e-45b8-b5d7-542ef8f3e542',
                 default_weights_path='weights',
                 model_name_prefix='aschoplex',
                 model_create_function=init_folds,
                 model_apply_function=ensemble_apply,
                 model_learn_function=ensemble_incremental_learning,
                 dimensionality=3,
                 model_type=DynamicEnsembleModel,
                 metadata=metadata
                 )
