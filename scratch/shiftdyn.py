import os
import pytorch_lightning as pl
import torch
from utils.loader import PQLoader, MaskedDataset, MaskedPQDataset
from utils.modules import PQModule, EarlyStopper

from utils.inference import infer_labels
from data import sample_data
from data.core import split_dataset
from models import pretrained
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help='Name of the run, data will be stored in results/args.run_name')
parser.add_argument('--seeds', type=int, default=100, help='Number of seeds to run')
parser.add_argument('--samples', default=[10, 20, 50], nargs='+', help='Number of samples to use for each dataset')
parser.add_argument('--splits', default=['p', 'q'], nargs='+',
                    help='Run on in or out of distribution data (p, q, or p q)')
parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use')
parser.add_argument('--resume', default=False, action='store_true', help='If not given and run_name exists, will error')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training detectron')
parser.add_argument('--ensemble_size', type=int, default=5, help='Number of models in the ensemble')
parser.add_argument('--max_epochs_per_model', type=int, default=5,
                    help='Maximum number of training epochs per model in the ensemble')
parser.add_argument('--patience', type=int, default=2,
                    help='Patience for early stopping based on no improvement on rejection rate for k models')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--refresh_model', action='store_true', default=False)
args = parser.parse_args()

if os.path.exists(run_dir := os.path.join('results', args.run_name)) and not args.resume:
    raise ValueError(f'Run name <{args.run_name}> already exists')
elif os.path.exists(run_dir) and args.resume:
    print(f'Resuming run <{args.run_name}>')
else:
    os.makedirs(run_dir)
    print(f'Directory created for run: {run_dir}')

load_model = lambda: pretrained.resnet18_trained_on_cifar10()
p_train, p_val, p_test_all = sample_data.cifar10(split='all')
q_all = sample_data.cifar10_1()

test_sets = {'p': p_test_all, 'q': q_all}
base_model = load_model()

# hyperparams ---------------------------------------------
max_epochs_per_model = args.max_epochs_per_model
optimizer = lambda params: torch.optim.Adam(params, lr=args.lr)
ensemble_size = args.ensemble_size
batch_size = args.batch_size
patience = args.patience
refresh_model = args.refresh_model
# ---------------------------------------------------------
gpus = [args.gpu]
num_workers = args.num_workers
# ---------------------------------------------------------


runs_id = 0
runs_total = len(args.samples) * args.seeds * 2
for N in map(int, args.samples):
    for seed in range(args.seeds):
        pl.seed_everything(seed)
        q, _ = split_dataset(test_sets['q'], N, seed)
        p, _ = split_dataset(test_sets['p'], N, seed)

        # setup save paths
        runs_id += 1
        test_path = os.path.join(run_dir, f'test_{seed}_{N}.pt')

        # look for cached results
        if os.path.exists(test_path):
            print(f'Found existing results for seed {seed}, N {N}')
            continue
        else:
            print(f'Running seed {seed}, N {N} (run: {runs_id}/{runs_total})')

        # set up the run parameters
        pl.seed_everything(seed)
        test_results = []
        log = {'N': N, 'seed': seed, 'ensemble_idx': 0}
        count = N

        # set up the dataset
        pq_loader = PQLoader(p_train=p_train,
                             p=p,
                             q=q,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             )

        # evaluate the base model on q
        alpha = 1 / (len(pq_loader.train_dataloader()) * count + 1)
        detector = PQModule(base_model, alpha=alpha)
        pl.Trainer(gpus=gpus, logger=False, max_epochs=1).test(detector, pq_loader.test_dataloader(), verbose=False)
        test_results.append(detector.test_struct.to_dict() | {'count': count} | log)

        # configure early stopping
        stopper = EarlyStopper(patience=patience, mode='min')
        stopper.update(count)

        # train the ensemble
        for i in range(1, ensemble_size + 1):
            log.update({'ensemble_idx': i})

            # set up the training module
            trainer = pl.Trainer(
                gpus=gpus,
                max_epochs=max_epochs_per_model,
                logger=False,
                num_sanity_val_steps=0,
                limit_val_batches=0,
                enable_model_summary=False
            )
            # set alpha to the suggested value in the paper
            # Note 1: we use lambda in the paper, but it is a reserved keyword, so we call it alpha here
            # Note 2: we use a custom batch sampler which slightly changes the way you compute lambda

            if refresh_model:
                detector = PQModule(model=load_model(),
                                    alpha=alpha)

            print(f'α = {1000 * alpha:.3f} × 10⁻³')

            # train the detectron model
            start_time = time.time()
            trainer.fit(detector, pq_loader)
            elapsed_time = time.time() - start_time
            print(f'Elapsed time: {elapsed_time:.2f} s')

            # evaluate the detectron model on the filtered q dataset
            trainer.test(detector, pq_loader.test_dataloader(), verbose=False)
            test_results.append(detector.test_struct.to_dict() | log)

        # save the results
        torch.save(test_results, os.path.join(run_dir, f'test_{seed}_{N}.pt'))
