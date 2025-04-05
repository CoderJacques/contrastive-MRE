config = {
    "train": {
            # choose training settings
            'nb_epochs': 50,
            'lr': 0.00005,
            'weight_decay': 0.0007904963769503105,
            'optimizer': 'adam',  # 'sgd', 'adam'

            # choose loss function
            'loss': 'KL',

            # choose bias correction
            'bias_correction': False,

            # choose early stopping
            'early_stopping': False,  # False, True
              },
    "dataset": {
                'batch_size': 32,

                #choose clinical studies
                'study_name': 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA',

                #choose normalization method
                'norm_func': 'mean_std_0',
                'norm_level': 'image_level',

                #choose gaussian sigma for soft label
                'soft_sigma': 2,

                #choose class imbalance method
                'LDS': False,

                #regularisation: noise --> None if no noise
                'noise_std': 0.1,
                },
    "model": {
            #choose CNN architecture
            'net': 'ResNet',

            #choose latent dimension
            'latent_dim': 256,

            #choose classifier (soft label) or regressor (hard label) --> adapt loss function accordingly
            'mode': 'regressor',

            'dropout': 0.1442504564000854,

            #choose image modality
            'obs': 'dr+stiffness', #'dr+stiffness', 'T1'
              }
        }