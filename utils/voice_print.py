import numpy as np


def cos_similarity(v1: np.ndarray, v2: np.ndarray, mode_v1=None, mode_v2=None):
    assert len(v1.shape) == 1 and v1.shape[0] > 1 and v1.shape == v2.shape, f'v1: {v1.shape}, v2: {v2.shape}'

    mode_v1 = np.linalg.norm(v1) if mode_v1 is None else mode_v1
    mode_v2 = np.linalg.norm(v2) if mode_v2 is None else mode_v2
    ans = np.dot(v1, v2) / (mode_v1 * mode_v2)

    return np.clip(ans, 0.0, 1.0)


def cos_dis(v1: np.ndarray, v2: np.ndarray, mode_v1=None, mode_v2=None):
    return 1 - cos_similarity(v1, v2, mode_v1, mode_v2)


def cal_feature_center(vp_mat: np.ndarray, by="mean"):
    nums, dim = vp_mat.shape
    norms = np.linalg.norm(vp_mat, axis=1)
    target_norm = np.mean(norms)
    scale = target_norm / norms
    scale_mat = np.expand_dims(scale, axis=1).repeat(dim, axis=1)
    normalized_vp_mat = vp_mat * scale_mat
    result = np.mean(normalized_vp_mat, axis=0)
    return result * target_norm / np.linalg.norm(result)


def calculate_similar_mat(vp_set1: np.ndarray, vp_set2: np.ndarray):
    assert len(vp_set1.shape) == 2 and len(vp_set2.shape) == 2 and vp_set1.shape[1] == vp_set2.shape[1]
    num1, num2 = vp_set1.shape[0], vp_set2.shape[0]
    similar_mat = np.zeros((num1, num2), dtype=np.float32)
    for i in range(num1):
        for j in range(num2):
            similar_mat[i][j] = cos_similarity(v1=vp_set1[i], v2=vp_set2[j])

    return similar_mat


def feature2string(feature):
    return ','.join(str(i) for i in feature)


def string2feature(string_feature):
    str_list = string_feature.split(',')
    feature_list = [float(x) for x in str_list]
    return np.array(feature_list, dtype=np.float32)