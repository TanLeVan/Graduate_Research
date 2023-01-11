import numpy as np
def pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pixel_point):
    pixelPointVector = np.array([pixel_point[0], pixel_point[1], 1)
    pixelPointVector= pixelPointVector.T
    leftHandSide = np.matmul(np.linalg.inv(cameraMatrix), np.linalg.inv(rotation_mat))
    rightHandSide = np.matmul(np.linalg.inv(rotation_mat),translation_vec.T)
    s = rightHandSide[2]/leftHandSide[2]

    3dPointVector = s*np.matmul(leftHandSide,pixelPointVector) - rightHandSide

    return 3dPointVector

def calculate_3d_gaze(cameraMatrix,rotation_mat, translation_vec, eye_centers_pixel, pupil_pixel):
    eye_centers_3D = np.empty([2,3])
    eye_centers_3D[0] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, eye_centers_pixel[0])
    eye_centers_3D[1] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, eye_centers_pixel[1])
    pupil_3D = np.empty([2,3])
    pupil_3D[0] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pupil_pixel[0])
    pupil_3D[1] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pupil_pixel[1])

    eyeball_centers_3D = eye_centers_3D - np.array([[0, 0, 12.5],[0 , 0, 12.5]])
    eyeball_centers_pixel = cv2.projectPoints(eyeball_centers_3D,rotation_vec, translation_vec, cameraMatrix, distCoeffs=None)

    gaze_3D = pupil_3D - eyeball_centers_3D

    rotation_vec = cv2.Rodrigues(rotation_mat)
    eyeball_centers_2D = cv2.projectPoints(eyeball_centers_3D, rotation_vec, translation_vec, self.cam_matrix, distCoeffs=None)
    print(eyeball_centers_2D)
    return gaze_3D
