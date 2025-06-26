import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def none_or_int(value):
    if value == 'None':
        return None
    return value

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for Brain Tumor Dataset and DataLoader")
    
    # Path to the data
    parser.add_argument('--data_path', type=str, default='./dataset/processed/', help='Path to the dataset')
    
    # Modalities in the dataset
    parser.add_argument('--modalities', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'], help='List of modalities')
    
    # Crop size for images
    parser.add_argument('--crop_size', type=int, nargs=3, default=[128, 128, 128], help='Crop size for images')
    
    # Batch size for DataLoader
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    
    # Number of workers for DataLoader
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')

    # Resume training from a checkpoint
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    
    # Directory for saving and loading checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./model/checkpoints', help='Directory to save and load checkpoints')

    # Resume training from a checkpoint
    parser.add_argument('--ldmtraining', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--cond_training', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--vqvae_training', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--lmunet_training', action='store_true', help='Start training from the latest checkpoint')


    parser.add_argument('--VQVAE', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--LDM', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--COND', action='store_true', help='Start training from the latest checkpoint')

    parser.add_argument('--LMUNET', action='store_true', help='Start training from the latest checkpoint')

    # parser = argparse.ArgumentParser('Vector Quantisation  training and evaluation script', add_help=False)
    # parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='VQUnet', type=str,
                        help='Name of model to train')
    # parser.add_argument('--image_size',  nargs="+",type=int, default=[128, 128, 128],
    #                     help='Size of input into the model. Prostate: [192, 192, 64], abdomen: [96, 96, 96], chest xray: [256, 256]')
    parser.add_argument('--dim', default='3D', type=str,
                        help='Dimension of image input')
    parser.add_argument('--drop_conv', type=float, default=0.0, 
                        help='Dropout rate for convolutional blocks (default: 0.)')
    parser.add_argument('--groups', type=int,  default=None, 
                        help='The number of groups for grouped normalisation. If None then batch normalisation')
    parser.add_argument('--in_ch', type=int,  default=1,
                        help='The number of input channels')
    parser.add_argument('--channels', type=int,  default=8,
                        help='The number of channels from first level of encoder')
    parser.add_argument('--enc_blocks' , nargs="+",type=int, default=[1, 1, 1, 1],
                        help='Number of ResBlocks per level of the encoder')
    parser.add_argument('--dec_blocks', nargs="+",type=int, default=[1, 1, 1],
                        help='Number of ResBlocks per level of the decoder')
    parser.add_argument('--act', default='nn.ReLU()', type=str,
                        help='Activation function to use. Enter "swish" if swish activation or enter function in torch function format if required different activation i.e. "nn.ReLU()"')
    parser.add_argument('--with_conv', type=str_to_bool, default=True,
                        help='Applying upsampling with convolution')
    parser.add_argument('--VQ', type=str_to_bool, default=True,
                        help='Apply vector quantisation in the bottleneck of the architecture. If False, then turns into original architecture')
    parser.add_argument('--pos_encoding', type=str_to_bool, default=False,
                        help='Apply Positional encoding before VQ in the bottleneck of the architecture.')
    parser.add_argument('--quantise', default='spatial', type=str, choices=['spatial', 'channel'],
                        help='Quantise either spatially or channel wise, enter either "spatial" or "channel"')
    parser.add_argument('--n_e', default=256, type=int,
                        help='The number of codebook vectors')
    parser.add_argument('--embed_dim', default=128, type=int,
                        help='The number of channels before quantisation. If quantising spatially, this will be equivelent to the codebook dimension')
    
    #Transformer parameter if using transunet
    parser.add_argument('--trans_attn', default='spatial', type=str, choices=['spatial', 'channel'],
                        help='Perform attention either spatially or channel wise, enter either "spatial" or "channel"')
    parser.add_argument('--trans_layers', default=8, type=int,
                        help='The number of transformer layers')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='The number of attention_heads')
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help='The hidden layer dimension size in the MLP layer of the transformer')
    parser.add_argument('--drop_attn', type=float, default=0.2,
                        help='Dropout rate for attention blocks (default: 0.2)')
    # Dataset parameters
    # parser.add_argument('--dataset', default='prostate', type=str, choices=['prostate', 'abdomen', 'chest'],
                        # help='Pick dataset to use, this can be extended if the user wishes to create their own dataloaders.py file')
    # parser.add_argument('--training_data', default='.../VQRobustSegmentation/data/Prostate/train.csv', type=str, required = True,  
    #                     help='training data csv file')
    # parser.add_argument('--validation_data', default='.../VQRobustSegmentation/data/Prostate/validation.csv', type=str, required = True,
    #                     help='validation data csv file')
    # parser.add_argument('--test_data', default='.../VQRobustSegmentation/data/Prostate/test.csv', type=str, required = True,
    #                     help='test data csv file')
    parser.add_argument('--binarise', type=str_to_bool, default=True,
                        help='Choose whether to binarise the segmentation map')
    parser.add_argument('--image_format', default='nifti',choices=['nifti', 'png'], type=str, 
                        help='state if image in nifti or png format')
    parser.add_argument('--labels', "--list", nargs="+",default=['Whole Prostate'],  
                        help='Label of classes. If not binarise, Prostate:["TZ", "PZ"], Chest: ["L-lung", "R-lung"], Abdomen: ["spleen",  "rkid", "lkid",  "gall", "eso", "liver", "sto",  "aorta",  "IVC",  "veins", "pancreas",  "rad", "lad"]')
    
    # Training parameters
    parser.add_argument('--classes', type=int, default=4, metavar='LR',
                        help='number of classes. 14 for abdomen CT, 3 for Prostate MRI, 3 for chest x-ray.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    # parser.add_argument('--val_epoch', type=int, default=5,
    #                     help='The number of epochs after which to evaluate the validation set')
    # parser.add_argument('--sliding_inference', type=str_to_bool, default=False,
    #                     help='Choose whether to perform sliding window inference. Only required for abdomen')
    
    #GPU usage
    # parser.add_argument('--gpus', type=int, default=1, 
    #                     help='number oF GPUs')
    # parser.add_argument('--nodes', type=int, default=1,
    #                     help='number of nodes')
    # parser.add_argument('--num_workers', default=8, type=int)
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--deterministic', type=int,  default=1,
    #                     help='whether to use deterministic training')

    #Output directory
    # parser.add_argument('--output_directory', default='.../VQRobustSegmentation/data/Prostate/output/', type=str, required = True, 
    #                     help='output directory to save images and results')
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
