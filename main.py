import os
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

curPath = os.path.abspath(os.path.dirname('../'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import warnings
import logging
from datetime import datetime
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from data import DataMovement, DataVolatility
from model.modeling_diva import DIVA





logger = logging.getLogger(__name__)


def parse_arguments(parser):
    ### Training Hyperparameters
    parser.add_argument('--seed', type=int, default=19, help="random seed")
    parser.add_argument("--num_train_epochs", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=86, help="default batch size is 32")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--mlm_probability", type=float, default=0.15,help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument('--use_mlm', default=1, type=int, choices=[0, 1], help="whether to use mlm")
    parser.add_argument('--use_ort', default=1, type=int, choices=[0, 1], help="whether to use use ortho loss")
    parser.add_argument('--use_mmd', default=1, type=int, choices=[0, 1], help="whether use MMD to the shared representation")
    parser.add_argument('--data_dir', type=str, default="./datasets")
    parser.add_argument('--bert_folder', type=str, default="./pretrained_model/", help="The folder name that contains the BERT model")
    parser.add_argument('--gpu', default=1, type=int, help="number of gpus to use")
    parser.add_argument("--fp16",action="store_true",help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--task', type=str, default="mov",choices=["mov","vol"],help="disentangled variables")
    parser.add_argument('--interest', type=str, default="prisk",choices=["prisk","senti"],help="T scenario")



    ### Model Hyperparameters ###
    parser.add_argument('--latent_dim', type=int, default=200, help="hidden dimension")
    parser.add_argument('--g_weight', type=float, default=1.0)
    parser.add_argument('--q_weight', type=float, default=1.0)
    parser.add_argument('--mlm_weight', type=float, default=0.01)
    parser.add_argument('--ort_weight', type=float, default=0.1)
    parser.add_argument('--kl_weight', type=float, default=1)
    parser.add_argument('--mmd_weight', type=float, default=0.1)



    ### Restart Contol
    parser.add_argument("--ckpt_name",default=None,type=str,help="The output directory",)
    parser.add_argument("--eval_only", action="store_true",)
    parser.add_argument("--load_ckpt",default=None,type=str,)
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--save_dir', default='./ckpt', type=str)


    ### Training Hyperparameters
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")



    return parser


# causal inference for earnings conference call movement
def main(parser: ArgumentParser) -> None:
    args = parser.parse_args()
    seed_everything(args.seed)

    args.num_workers = 8
    args.ckpt_name = args.task + "."+ args.interest+".seed"+str(args.seed)
    args.ckpt_dir = args.save_dir + "/" + args.ckpt_name



    os.makedirs(args.ckpt_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    logger.info("Training/evaluation parameters %s", args)

    lr_logger = LearningRateMonitor()

    data_module = None
    save_monitor = None
    save_mod = None



    if args.task == "mov":
        save_monitor = 'v_q_acc'
        save_mod = 'max'
        print('***DIVA Causal Estimation for {} on Stock Movement***'.format(args.interest))
        data_module = DataMovement(args)
    elif args.task == "vol":
        save_monitor = 'v_q_mse'
        save_mod = 'min'
        print('***DIVA Causal Estimation for {} on Stock Volatility***'.format(args.interest))
        data_module = DataVolatility(args)
    else:
        raise ValueError("The no task found")


    model = DIVA(args)

    args.train_len = len(data_module.processed_datasets['train'])


    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs


    if not args.ckpt_name:
        d = datetime.now()
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}lr{}_{}'.format(args.batch_size * args.accumulate_grad_batches,
                                               args.lr, time_str)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=1,
        monitor=save_monitor,
        mode=save_mod,
        save_weights_only=True,
        filename='{epoch}',

    )



    trainer = Trainer(
        # logger=tb_logger,
        enable_progress_bar=True,
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs,
        accelerator='gpu',
        devices=[args.gpu],
        enable_checkpointing=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=0,
        # val_check_interval=1,  # use float to check every n epochs
        precision=16 if args.fp16 else 32,
        callbacks=[lr_logger, checkpoint_callback],

    )


    if args.eval_only:

        print('Evaluating')
        data_module.setup('test')

        files = [f for f in os.listdir(args.ckpt_dir) if not f.startswith('.')][0]

        model.load_state_dict(torch.load(args.ckpt_dir+'/'+files, map_location=model.device)['state_dict'])

        trainer.test(model, datamodule=data_module)
    else:

        # Training Causal Model
        data_module.setup('fit')
        trainer.fit(model, data_module)


        # Plug in to estimate ATE and PEHE
        files = [f for f in os.listdir(args.ckpt_dir) if not f.startswith('.')][0]
        model.load_state_dict(torch.load(args.ckpt_dir+'/'+files, map_location=model.device)['state_dict'])

        data_module.setup('test')
        trainer.test(model, datamodule=data_module) #also loads training dataloader






if __name__ == '__main__':
    parser = ArgumentParser()
    parser = parse_arguments(parser)
    main(parser)