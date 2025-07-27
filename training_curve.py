from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def read_tensorboard_log(path, key_name):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    scalars = event_acc.Scalars(key_name)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return steps, values

def read_all_keys(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    print(event_acc.Tags()['scalars'])
    
    
# Paths
log_path_A = "/home/stud/xiji/SC-Depth_Anything/checkpoints/wo_dym_mask——comb_unty/04-23-20:23/events.out.tfevents.1745432606.node18"
log_path_B = "/home/stud/xiji/SC-Depth_Anything/checkpoints/wo_dym_mask——single_unty/04-03-23:39/events.out.tfevents.1743716389.node18"

# Read losses
steps_train_A, train_loss_A = read_tensorboard_log(log_path_A, "total_loss")
steps_val_A, val_loss_A = read_tensorboard_log(log_path_A, "Total_loss")

steps_train_B, train_loss_B = read_tensorboard_log(log_path_B, "total_loss")
steps_val_B, val_loss_B = read_tensorboard_log(log_path_B, "Total_loss")

# Config
steps_per_epoch = 1608

# Map validation epochs to steps
steps_val_A_mapped = [(epoch_idx + 1) * steps_per_epoch for epoch_idx in range(len(val_loss_A))]
steps_val_B_mapped = [(epoch_idx + 1) * steps_per_epoch for epoch_idx in range(len(val_loss_B))]


plt.figure(figsize=(10,7))

# Training losses
plt.plot(steps_train_A, train_loss_A, label="Combined Projected Uncertainty - Train Loss", color="blue", linestyle="-")
plt.plot(steps_train_B, train_loss_B, label="Single Uncertainty - Train Loss", color="green", linestyle="-")

# Validation losses
plt.plot(steps_val_A_mapped, val_loss_A, label="Combined Projected Uncertainty - Val Loss", color="blue", linestyle="--")
plt.plot(steps_val_B_mapped, val_loss_B, label="Single Uncertainty - Val Loss", color="green", linestyle="--")

plt.xlabel("Steps", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.title("Comparison of Training and Validation Loss\n(Combined Projected Uncertainty vs Single Uncertainty)", fontsize=22)
plt.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_comparison.pdf", format='pdf')
plt.show()
