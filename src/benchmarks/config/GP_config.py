config = {
    "train": {
            # choose training settings
            'nb_epochs': 50,
            'lr': 0.00005,
            'optimizer': 'adam',

            # choose bias correction
            'bias_correction': False,

            # choose early stopping
            'early_stopping': False,  # False, True
              },
    "dataset": {
                #choose clinical studies
                'study_name': 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA',

                #choose normalization method
                'norm_func': 'mean_std_0',
                'norm_level': 'dataset_level',

                #choose gaussian sigma for soft label
                'soft_sigma': None,

                #choose augmentation method
                'augment': False,
                'augmentations': 'noise+mirror+rotate+shift',

                #regularisation: noise --> None if no noise
                'noise_std': 0.1,

                #LDS --> not used just a placeholder
                'LDS': False,
                },
    "model": {
            #choose GP kernel
            'kernel': 'linear',

            #choose PCA
            'pca_k': 100,

            #choose image modality
            'obs': 'dr+stiffness', #'dr+stiffness', 'T1'
              }
        }