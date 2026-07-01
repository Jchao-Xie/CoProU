import matplotlib.pyplot as plt
import os


def plot_window_alignment_debug(
    current_window_abs_poses,
    aligned_window_abs_poses,
    last_window_abs_poses,
    sliding_step,
    save_path,
    title="window_alignment_debug",
):
    """
    current_window_abs_poses: [W, 4, 4] current window before alignment
    aligned_window_abs_poses: [W, 4, 4] current window after Sim(3)
    last_window_abs_poses:    [W, 4, 4] previous window in global frame
    """

    current_xyz = current_window_abs_poses[:, :3, 3]
    aligned_xyz = aligned_window_abs_poses[:, :3, 3]
    last_xyz = last_window_abs_poses[:, :3, 3]

    W = current_xyz.shape[0]
    O = W - sliding_step

    fig = plt.figure(figsize=(14, 6))

    # -------------------------
    # 2D top-down view: x-z
    # -------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(current_xyz[:, 0], current_xyz[:, 2], marker='o', label='current (before)')
    ax1.plot(aligned_xyz[:, 0], aligned_xyz[:, 2], marker='o', label='current (aligned)')
    ax1.plot(last_xyz[:, 0], last_xyz[:, 2], marker='o', label='last window')

    if O > 0:
        ax1.scatter(current_xyz[:O, 0], current_xyz[:O, 2], marker='s', s=60, label='current overlap')
        ax1.scatter(aligned_xyz[:O, 0], aligned_xyz[:O, 2], marker='s', s=60, label='aligned overlap')
        ax1.scatter(last_xyz[sliding_step:, 0], last_xyz[sliding_step:, 2], marker='s', s=60, label='last overlap')

    if sliding_step > 0:
        ax1.scatter(aligned_xyz[-sliding_step:, 0], aligned_xyz[-sliding_step:, 2],
                    marker='*', s=120, label='new tail')

    for i in range(W):
        ax1.text(current_xyz[i, 0], current_xyz[i, 2], f'c{i}', fontsize=8)
        ax1.text(aligned_xyz[i, 0], aligned_xyz[i, 2], f'a{i}', fontsize=8)
        ax1.text(last_xyz[i, 0], last_xyz[i, 2], f'l{i}', fontsize=8)

    ax1.set_title("Top-down view (x-z)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # -------------------------
    # 3D view
    # -------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(current_xyz[:, 0], current_xyz[:, 1], current_xyz[:, 2], marker='o', label='current (before)')
    ax2.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], aligned_xyz[:, 2], marker='o', label='current (aligned)')
    ax2.plot(last_xyz[:, 0], last_xyz[:, 1], last_xyz[:, 2], marker='o', label='last window')

    if O > 0:
        ax2.scatter(current_xyz[:O, 0], current_xyz[:O, 1], current_xyz[:O, 2], marker='s', s=60)
        ax2.scatter(aligned_xyz[:O, 0], aligned_xyz[:O, 1], aligned_xyz[:O, 2], marker='s', s=60)
        ax2.scatter(last_xyz[sliding_step:, 0], last_xyz[sliding_step:, 1], last_xyz[sliding_step:, 2], marker='s', s=60)

    if sliding_step > 0:
        ax2.scatter(aligned_xyz[-sliding_step:, 0], aligned_xyz[-sliding_step:, 1], aligned_xyz[-sliding_step:, 2],
                    marker='*', s=120, label='new tail')

    ax2.set_title("3D view")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)