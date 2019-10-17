import os
from config.default_config import DefaultConfig
from datetime import datetime

class RESNET_SPL10_CONFIG(DefaultConfig):

    def args(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        ########### Quick Setup ############
        model = 'contrastive'  #
        arch = 'resnet18'  # | resnet50
        pretrained = ''

        gpus = '0,1,2,3'
        batch_size = 256
        task_name = 'local'
        lr_schedule = 'lambda'  # lambda|step|plateau
        lr = 6e-4

        len_gpu = str(len(gpus.split(',')))

        log_path = os.path.join(DefaultConfig.LOG_PATH, model, arch, 'spl10',
                                ''.join([task_name, '_', lr_schedule, '_', 'gpus-', len_gpu
                                ]), current_time)

        return {

            'TASK_NAME': task_name,
            'GPU_IDS': gpus,
            'BATCH_SIZE': batch_size,
            'PRETRAINED': pretrained,

            'LOG_PATH': log_path,

            # MODEL
            'MODEL': model,
            'ARCH': arch,
            'SAVE_BEST': True,

            'LR_POLICY': lr_schedule,

            'NITER': 100,
            'NITER_DECAY': 400,
            'NITER_TOTAL': 500,
            'FIVE_CROP': False,
            'LR': lr,
        }
