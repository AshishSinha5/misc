import time
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
from model import LinearEmbeddingModel
from main import get_dataset, train, evaluate, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'
num_class = 3
epochs = 10
valid_ratio = 0.1
author_code = {
    'EAP': 'Edgar Allan Poe',
    'HPL': 'HP Lovecraft',
    'MWS': 'Mary Shelley'
}

label_code = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}


def objective(trial):
    train_dataset, valid_dataset, vocab = get_dataset(train_file_path, test_file_path, valid_ratio)

    batch_size = trial.suggest_int('batch_size', 4, 64)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    init_range = trial.suggest_float('init_range', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 0, 3)
    out_feats = []
    dropouts = []
    for i in range(num_layers):
        out_feats.append(trial.suggest_int('units_l{}'.format(i), 4, 128))
        dropouts.append(trial.suggest_float('drop_l{}'.format(i), 0.1, 0.75))
    embed_dim = trial.suggest_int('embed_dim', 16, 64)
    vocab_size = len(vocab)

    model = LinearEmbeddingModel(vocab_size, num_class, num_layers, out_feats, dropouts, embed_dim, init_range).to(
        device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_function, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False,
                              collate_fn=valid_dataset.collate_function)

    total_acc = None
    acc_val = 0
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_loader, optimizer, criterion, epoch)
        acc_val, loss_val = evaluate(model, valid_loader, criterion)
        if total_acc is not None and total_acc > acc_val:
            scheduler.step()
        else:
            total_acc = acc_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} | valid loss {:8.5f} '.format(epoch,
                                                                   time.time() - epoch_start_time,
                                                                   acc_val, loss_val))
        print('-' * 59)

    return acc_val



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open('best_params.pickle', 'wb') as f:
        pickle.dump(trial.params, f, protocol=pickle.HIGHEST_PROTOCOL)



