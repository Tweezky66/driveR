import torch
import matplotlib.pyplot as plt
from depth.dataset import DepthDataset, list_episodes
from  depth.model import DepthNet
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

ROOT = "/data"
MAX_DEPTH = 80.0
BATCH_SIZE = 16
LR = 1e-3 
ENCODER_LR = 1e-5 
FROZEN_EPOCHS = 300
FINETUNE_EPOCHS = 0 
CKPT_PATH = "best.pt"
PRINT_EVERY = 10


OVERFIT_N = 8


def masked_log_l1(pred_log, depth, mask):
    valid = mask & (depth > 1e-3)
    if not valid.any():
        return pred_log.sum() * 0.0
    target = torch.log(depth.clamp(min=1e-3))
    return (pred_log - target).abs()[valid].mean()


def depth_errors(pred_log, depth, mask):
    valid = mask & (depth > 1e-3)
    pred = torch.exp(pred_log.clamp(max=7.0))[valid]
    gt = depth[valid]
    err = (pred - gt).abs()
    return (err / gt).mean().item(), err.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    episodes = list_episodes(ROOT)
    if OVERFIT_N:
        full = DepthDataset(ROOT, episodes, max_depth=MAX_DEPTH)
        train_ds = Subset(full, range(min(OVERFIT_N, len(full))))
        val_ds = train_ds
    else:
        train_val_eps, test_eps = train_test_split(episodes, test_size=0.15, random_state=42)
        train_eps, val_eps = train_test_split(train_val_eps, test_size=0.15 / 0.85, random_state=42)
        print("train:", train_eps, "| val:", val_eps, "| test (not used here):", test_eps)
        train_ds = DepthDataset(ROOT, train_eps, max_depth=MAX_DEPTH)
        val_ds = DepthDataset(ROOT, val_eps, max_depth=MAX_DEPTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # keep num_workers 2 for  python env < 3.13
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DepthNet(freeze_encoder=True).to(device)

    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam(decoder_params, lr=LR)

    best_val = float("inf")
    train_hist, val_hist = [], []

    for epoch in range(FROZEN_EPOCHS + FINETUNE_EPOCHS):
        phase = "frozen" if epoch < FROZEN_EPOCHS else "finetune"

        if epoch == FROZEN_EPOCHS:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": ENCODER_LR},
                {"params": decoder_params, "lr": LR},
            ])
            print("phase 2: encoder unfrozen")

        model.train()
        if phase == "frozen":
            model.encoder.eval()

        train_loss = 0.0
        for image, depth, mask in tqdm(train_loader, desc=f"epoch {epoch + 1} [train]", leave=False):
            image, depth, mask = image.to(device), depth.to(device), mask.to(device)

            optimizer.zero_grad()
            pred_log = model(image)
            loss = masked_log_l1(pred_log, depth, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = val_absrel = val_mae = 0.0
        with torch.no_grad():
            for image, depth, mask in val_loader:
                image, depth, mask = image.to(device), depth.to(device), mask.to(device)

                pred_log = model(image)
                val_loss += masked_log_l1(pred_log, depth, mask).item()
                absrel, mae = depth_errors(pred_log, depth, mask)
                val_absrel += absrel
                val_mae += mae

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_absrel /= len(val_loader)
        val_mae /= len(val_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"epoch {epoch + 1:3d} ({phase:8s}) | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| val AbsRel {val_absrel:.3f} | val MAE {val_mae:.2f} m")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_PATH)  

    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("masked log-L1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()