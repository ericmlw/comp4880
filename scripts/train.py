import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data_loader import build_taggnn_graph, data_split_and_prepare
from utils import create_models, initialize_loss

parser = argparse.ArgumentParser(description='Train GCMC Model')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay rate')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--hidden_dim', type=int, default=5, help='Hidden dimension')
parser.add_argument('--out_dim', type=int, default=5, help='Output dimension')
parser.add_argument('--drop_out', type=float, default=0.0, help='Dropout ratio')
parser.add_argument('--save_steps', type=int, default=100, help='Every #steps to save the model')
parser.add_argument('--log_dir', help='Folder to save log')
parser.add_argument('--saved_model_folder', help='Folder to save model')
parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--use_laplacian_loss', type=int, default=0, help='Use laplacian loss')
parser.add_argument('--laplacian_loss_weight', type=float, default=0.1, help='Laplacian loss weight')

args = parser.parse_args()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def validate(score, mask):
    pred = torch.sigmoid(score)
    pred = pred.cpu().detach().numpy()
    
    test_mask = mask > 0
    test_mask = test_mask.cpu().detach().numpy()
    pred_binary = (pred > 0.5).astype(np.float32)
    
    accuracy = (pred_binary == test_mask).mean()
    
    return accuracy

def main(args):
    # Get arguments
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    drop_out = args.drop_out
    save_steps = args.save_steps
    log_dir = args.log_dir
    saved_model_folder = args.saved_model_folder
    dataset = args.dataset
    use_laplacian_loss = args.use_laplacian_loss
    laplacian_loss_weight = args.laplacian_loss_weight

    # Log directory
    post_fix = '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = log_dir + post_fix
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(saved_model_folder):
        os.makedirs(saved_model_folder)
    weights_name = saved_model_folder + post_fix + '_weights'

    # Prepare data
    args, g = build_taggnn_graph(args)
    label_mat, tc_label_mat, mask_mat, g, split_items = data_split_and_prepare(args, g)

    feature_q, feature_i, feature_t = g.ndata['wids'][:args.n_query], g.ndata['wids'][args.n_query:args.n_query + args.n_item], g.ndata['wids'][args.n_query + args.n_item:]
    M_qi = g.adj(scipy_fmt="csr")[args.n_query:args.n_query + args.n_item, :args.n_query]
    M_it = g.adj(scipy_fmt="csr")[args.n_query + args.n_item:, args.n_query:args.n_query + args.n_item]

    user_item_matrix_train_qi = label_mat[split_items[0] - args.n_query]
    user_item_matrix_train_it = label_mat[split_items[0] - args.n_query]
    user_item_matrix_test_qi = label_mat[split_items[2] - args.n_query]
    user_item_matrix_test_it = label_mat[split_items[2] - args.n_query]

    laplacian_q = torch.eye(args.n_query)  # Adjust if laplacian matrices are precomputed
    laplacian_i = torch.eye(args.n_item)
    laplacian_t = torch.eye(args.n_tag)

    # Initialize model
    model = create_models(feature_q, feature_i, feature_t, args.vocab_size, hidden_dim, M_qi, M_it, out_dim, drop_out)
    if torch.cuda.is_available():
        model = model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Ensure matrices are correctly processed and passed as tensors
    mask_qi = (user_item_matrix_train_qi > 0).astype(np.float32)
    mask_it = (user_item_matrix_train_it > 0).astype(np.float32)
    
    # Initialize the loss function
    criterion = initialize_loss(mask_qi, mask_it, laplacian_loss_weight)
    
    # Training loop
    iter_bar = tqdm(range(num_epochs), desc='Iter (loss=X.XXX)')
    for epoch in iter_bar:
        model.train()
        optimizer.zero_grad()
    
        score_qi, score_it = model()
        loss_value = criterion.loss(score_qi, score_it)
        
        if use_laplacian_loss:
            laplacian_loss = criterion.laplacian_loss(score_qi, score_it, laplacian_q, laplacian_i, laplacian_t)
            loss_value += laplacian_loss
    
        loss_value.backward()
        optimizer.step()
    
        with torch.no_grad():
            model.eval()
            train_acc_qi = validate(score_qi, user_item_matrix_train_qi > 0)
            train_acc_it = validate(score_it, user_item_matrix_train_it > 0)
            val_acc_qi = validate(score_qi, user_item_matrix_test_qi > 0)
            val_acc_it = validate(score_it, user_item_matrix_test_it > 0)
            iter_bar.set_description('Iter (loss=%5.3f, train_acc_qi=%5.3f, train_acc_it=%5.3f, val_acc_qi=%5.5f, val_acc_it=%5.5f)' % (loss_value.item(), train_acc_qi, train_acc_it, val_acc_qi, val_acc_it))
            writer.add_scalars('scalar', {'loss': loss_value.item(), 'train_acc_qi': train_acc_qi, 'train_acc_it': train_acc_it, 'val_acc_qi': val_acc_qi, 'val_acc_it': val_acc_it}, epoch)

        if epoch % save_steps == 0:
            torch.save(model.state_dict(), weights_name)

    torch.save(model.state_dict(), weights_name)
    print('Training complete. Final model saved.')

if __name__ == '__main__':
    main(args)

