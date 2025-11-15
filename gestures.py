import torch

def classify_gesture(keypoints):
    kp = keypoints.xy[0]
    if kp.shape[0] < 17:
        return "Incomplete Pose"

    nose = kp[0]
    left_shoulder, right_shoulder = kp[5], kp[6]
    left_elbow, right_elbow = kp[7], kp[8]
    left_wrist, right_wrist = kp[9], kp[10]
    left_hip, right_hip = kp[11], kp[12]

    def above(p1, p2): return p1[1] < p2[1]
    def aligned(p1, p2): return abs(p1[1] - p2[1]) < 20
    def near(p1, p2): return torch.norm(p1 - p2) < 40

    if above(left_wrist, nose) and above(right_wrist, nose):
        return "Raise Hands"
    if above(left_wrist, nose) and not above(right_wrist, nose):
        return "Left Hand Raised"
    if above(right_wrist, nose) and not above(left_wrist, nose):
        return "Right Hand Raised"
    if aligned(left_wrist, left_shoulder) and aligned(right_wrist, right_shoulder):
        return "T-Pose"
    if near(left_wrist, left_hip) and near(right_wrist, right_hip):
        return "Hands on Hips"
    if left_wrist[0] < left_elbow[0] and aligned(left_wrist, left_shoulder):
        return "Point Left"
    if right_wrist[0] > right_elbow[0] and aligned(right_wrist, right_shoulder):
        return "Point Right"
    if near(left_wrist, right_shoulder) and near(right_wrist, left_shoulder):
        return "Crossed Arms"
    return "Neutral"
