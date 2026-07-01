import numpy as np


def rel_to_abs_poses(rel_poses: np.ndarray) -> np.ndarray:
    """
    Convert a chain of relative poses [N-1, 4, 4] into absolute poses [N, 4, 4]
    with the first frame as identity.
    """
    rel_poses = np.asarray(rel_poses, dtype=np.float64)
    n_rel = rel_poses.shape[0]

    abs_poses = np.tile(np.eye(4, dtype=np.float64)[None], (n_rel + 1, 1, 1))
    for i in range(n_rel):
        abs_poses[i + 1] = abs_poses[i] @ rel_poses[i]
    return abs_poses


def umeyama_sim3(src_xyz: np.ndarray, dst_xyz: np.ndarray, eps: float = 1e-8):
    """
    Estimate Sim(3): dst ~= s * R * src + t
    src_xyz, dst_xyz: [N, 3]
    Returns: s, R, t
    """
    src_xyz = np.asarray(src_xyz, dtype=np.float64)
    dst_xyz = np.asarray(dst_xyz, dtype=np.float64)

    assert src_xyz.shape == dst_xyz.shape
    assert src_xyz.ndim == 2 and src_xyz.shape[1] == 3

    n = src_xyz.shape[0]
    if n == 0:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    mu_src = src_xyz.mean(axis=0)
    mu_dst = dst_xyz.mean(axis=0)

    src_centered = src_xyz - mu_src
    dst_centered = dst_xyz - mu_dst

    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    if var_src < eps:
        return 1.0, np.eye(3, dtype=np.float64), (mu_dst - mu_src)

    cov = (dst_centered.T @ src_centered) / n  # note: dst * src^T

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    s = np.sum(D * np.diag(S)) / var_src
    t = mu_dst - s * (R @ mu_src)

    return float(s), R, t


def se3_from_anchor(src_pose: np.ndarray, dst_pose: np.ndarray):
    """
    Fallback when overlap is too small / degenerate.
    Estimate only a rigid transform from one anchor pose.
    """
    R_src = src_pose[:3, :3]
    t_src = src_pose[:3, 3]
    R_dst = dst_pose[:3, :3]
    t_dst = dst_pose[:3, 3]

    R = R_dst @ np.linalg.inv(R_src)
    t = t_dst - R @ t_src
    s = 1.0
    return s, R, t


def estimate_overlap_sim3(
    current_overlap_abs: np.ndarray,
    previous_overlap_abs: np.ndarray,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    eps: float = 1e-8,
):
    """
    Estimate Sim(3) aligning current overlap to previous overlap.

    current_overlap_abs: [O, 4, 4]
    previous_overlap_abs: [O, 4, 4]

    Returns s, R, t such that:
        previous ~= Sim3(current)
    """
    assert current_overlap_abs.shape == previous_overlap_abs.shape
    assert current_overlap_abs.ndim == 3 and current_overlap_abs.shape[1:] == (4, 4)

    o = current_overlap_abs.shape[0]
    if o == 0:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    src_xyz = current_overlap_abs[:, :3, 3]
    dst_xyz = previous_overlap_abs[:, :3, 3]

    if o < 2:
        return se3_from_anchor(current_overlap_abs[0], previous_overlap_abs[0])

    src_var = np.mean(np.sum((src_xyz - src_xyz.mean(axis=0)) ** 2, axis=1))
    if src_var < eps:
        return se3_from_anchor(current_overlap_abs[0], previous_overlap_abs[0])

    s, R, t = umeyama_sim3(src_xyz, dst_xyz, eps=eps)

    if (not np.isfinite(s)) or (s < min_scale) or (s > max_scale):
        print(f"[Warning] bad Sim(3) scale {s:.4f}, fallback to anchor SE(3)")
        return se3_from_anchor(current_overlap_abs[0], previous_overlap_abs[0])

    return s, R, t


def apply_sim3_to_abs_poses(abs_poses: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply Sim(3) to a set of absolute poses.
    Rotation is left-multiplied by R.
    Translation is transformed by s * R * x + t.
    """
    abs_poses = np.asarray(abs_poses, dtype=np.float64)
    out = abs_poses.copy()

    out[:, :3, :3] = np.einsum("ij,njk->nik", R, abs_poses[:, :3, :3])
    out[:, :3, 3] = (s * (R @ abs_poses[:, :3, 3].T)).T + t[None, :]
    out[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return out