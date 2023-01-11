#!/usr/bin/python3
# -*- coding:utf-8 -*-

from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread
import sys

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)

def pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pixel_point):
    """levantann321@gmail.com
    This function input a 2d pixel coordinate point and calculate the corresponding 3d object coordinate of that
    point. Necessary information are internal camera matrix, rotation matrix and translation vector from
    3d camera coordinate to 3d object coordinate"""
    pixelPointVector = np.array([pixel_point[0], pixel_point[1], 1]).astype(int)
    #np.transpose(pixelPointVector)
    #print(pixelPointVector)
    #print(rotation_mat, translation_vec)
    leftHandSide = np.matmul(np.matmul(np.linalg.inv(rotation_mat), np.linalg.inv(cameraMatrix)),pixelPointVector)
    #print("left hand side: ", leftHandSide)
    rightHandSide = np.matmul(np.linalg.inv(rotation_mat),translation_vec)
    #print(rightHandSide)
    s = rightHandSide[2]/leftHandSide[2]
    worldPointVector = s*leftHandSide - rightHandSide
    #print(worldPointVector)
    return worldPointVector

def calculate_3d_gaze_self(frame,cameraMatrix, rotation_vec, translation_vec, eye_centers_pixel, pupil_pixel):
    rotation_mat,_ = cv2.Rodrigues(rotation_vec)
    #print("eye center pixel: ", eye_centers_pixel)
    eye_centers_3D = np.empty([2,3])
    eye_centers_3D[0] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, eye_centers_pixel[0]) #fix this
    eye_centers_3D[1] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, eye_centers_pixel[1]) #fix this
    #print(eye_centers_3D)
    pupil_3D = np.empty([2,3])
    pupil_3D[0] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pupil_pixel[0])
    pupil_3D[1] = pixel_to_3d_coordinate(cameraMatrix, rotation_mat, translation_vec, pupil_pixel[1])

    eyeball_centers_3D = eye_centers_3D - np.array([[0, 0, 12.5],[0 , 0, 12.5]])
    #eyeball_centers_pixel = eyeball_centers_pixel.transpose((1,0,2)).reshape(2,2).astype(np.int32)
    #print(eyeball_centers_pixel)

    gaze_3D = pupil_3D - eyeball_centers_3D
    #print(gaze_3D)
    return gaze_3D, eyeball_centers_3D, pupil_3D

def draw_sticker(src, eyeball_centers_3D, pupil_pixel, pupil_3D, rotation_vec, translation_vec, cameraMatrix, landmarks, blink_thd=0.3,
                    arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]

    right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]

    eyeball_centers_pixel = np.empty([2,2])
    eyeball_centers_pixel[0], _ = cv2.projectPoints(eyeball_centers_3D[0], rotation_vec, translation_vec, cameraMatrix, distCoeffs = None)
    eyeball_centers_pixel[1], _ = cv2.projectPoints(eyeball_centers_3D[1], rotation_vec, translation_vec, cameraMatrix, distCoeffs = None)

    extend_point_3D = pupil_3D + (pupil_3D - eyeball_centers_3D)/12.5*30
    extend_point_pixel=np.empty([2,2])
    extend_point_pixel[0],_ = cv2.projectPoints(extend_point_3D[0], rotation_vec, translation_vec, cameraMatrix, distCoeffs = None)
    extend_point_pixel[1],_ = cv2.projectPoints(extend_point_3D[1], rotation_vec, translation_vec, cameraMatrix, distCoeffs = None)

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1,
                    color=(125, 255, 125), thickness=-1)

    if left_eye_hight / left_eye_width > blink_thd:
        cv2.circle(src, eyeball_centers_pixel[0].astype(int), radius=1, color=(225, 255, 0), thickness=1)
        cv2.circle(src, pupil_pixel[0].astype(int), radius = 1, color= (0,255,255), thickness=1)
        cv2.line(src, eyeball_centers_pixel[0].astype(int), extend_point_pixel[0].astype(int), color=(0, 255, 255), thickness = 1)

    if right_eye_hight / right_eye_width > blink_thd:
        cv2.circle(src, eyeball_centers_pixel[1].astype(int), radius=1, color=(225, 255, 0), thickness=1)
        cv2.circle(src, pupil_pixel[1].astype(int), radius = 1, color= (0,255,255), thickness=1)
        cv2.line(src, eyeball_centers_pixel[1].astype(int), extend_point_pixel[1].astype(int), color=(0, 255, 255), thickness = 1)

    return src


def main(video, gpu_ctx=-1):
    cap = cv2.VideoCapture(video)

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    hp = HeadPoseEstimator("weights/object_points.npy", cap.get(3), cap.get(4))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = fd.detect(frame)

        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            _, euler_angle, rotation_mat, rotation_vec, translation_vec = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
            eye_centers = np.average(eye_markers, axis=1)

            #eye_centers_3D = pixel_to_3d_coordinate(hp.cam_matrix,rotation_mat, translation_vec, eye_centers[0])
            # eye_centers = landmarks[[34, 88]]

            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

            iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = gs.draw_pupil(iris_left, frame, thickness=1) #location of the left pupil (in pixel)

            iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = gs.draw_pupil(iris_right, frame, thickness=1) #Location of the right pupil

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            #print(eye_centers[0], type(eye_centers))
            gaze_3D, eyeball_centers_3D, pupil_3D = calculate_3d_gaze_self(frame, hp.cam_matrix, rotation_vec, translation_vec, eye_centers, pupils)

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            draw_sticker(frame, eyeball_centers_3D, pupils, pupil_3D, rotation_vec, translation_vec, hp.cam_matrix,landmarks)
            gs.draw_eye_markers(eye_markers, frame, thickness=1)
            #print(landmarks[[39]].reshape(2))

            #Visualization of landmark 39 and 35
            #cv2.circle(frame, tuple(landmarks[[89]].reshape(2).astype(int)), 1, (255,0,255), 1, cv2.LINE_AA)
            #cv2.circle(frame, tuple(landmarks[[35]].reshape(2).astype(int)), 1, (255,0,255), 1, cv2.LINE_AA)

        #cv2.imshow('res', cv2.resize(frame, (960, 540)))
        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main(sys.argv[1])
