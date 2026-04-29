import cv2
import numpy as np
import math
from collections import deque

def transform_initial_image(image_path):
    print('TASK 1')
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return
    transformed = cv2.flip(img, -1)
    cv2.imwrite('variant_transformed.jpg', transformed)
    cv2.imshow("Transformed Image (flip -1)", transformed)
    cv2.waitKey(0)
    cv2.destroyWindow("Transformed Image (flip -1)")


def track_marker(marker_img_path, fly_img_path, fly_scale=0.4):
    print('TASK 2, 3 & EXTRA')
    marker_img = cv2.imread(marker_img_path)
    if marker_img is None:
        print(f"Ошибка: не удалось загрузить метку '{marker_img_path}'")
        return
    marker_gray = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_marker, des_marker = orb.detectAndCompute(marker_gray, None)

    #сравнение дескриптеров наборов, счёт кол-ва разных битов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    fly_img = cv2.imread(fly_img_path, cv2.IMREAD_UNCHANGED)
    if fly_img is None:
        print(f"Ошибка: не удалось загрузить '{fly_img_path}'")
        return
    fly_h, fly_w = fly_img.shape[:2]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера не найдена")
        return

    # Переменные для сглаживания
    last_good_center = None
    last_valid_M = None
    lost_frames = 0
    MAX_LOST_FRAMES = 30
    center_history = deque(maxlen=5)

    print("Покажите метку. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]
        center_frame = (w_frame // 2, h_frame // 2)

        # Первичная обработка кадра
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(gray_frame)
        equ = cv2.medianBlur(equ, 3)

        kp_frame, des_frame = orb.detectAndCompute(equ, None)
        good_matches_count = 0
        found_this_frame = False
        M = None

        if des_frame is not None and len(des_frame) > 0:
            matches = bf.knnMatch(des_marker, des_frame, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.80 * n.distance: #Тест Лоу
                    good.append(m)
            good_matches_count = len(good)

            #query - метка, train - кадр
            if good_matches_count > 8:
                src_pts = np.float32([kp_marker[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    found_this_frame = True
                    last_valid_M = M
                    lost_frames = 0

        # Удержание гомографии
        if not found_this_frame and last_valid_M is not None:
            lost_frames += 1
            if lost_frames <= MAX_LOST_FRAMES:
                M = last_valid_M
                found_this_frame = True
            else:
                last_valid_M = None
                last_good_center = None
                center_history.clear()

        # Отрисовка
        if found_this_frame and M is not None:
            h_m, w_m = marker_gray.shape
            corners_marker = np.float32([[0,0],[0,h_m],[w_m,h_m],[w_m,0]]).reshape(-1,1,2)
            corners_frame = cv2.perspectiveTransform(corners_marker, M)

            # Центр метки в пикселях кадра
            center_marker_local = np.float32([[w_m/2, h_m/2]]).reshape(-1,1,2)
            center_marker_frame = cv2.perspectiveTransform(center_marker_local, M)[0][0]
            cX, cY = int(center_marker_frame[0]), int(center_marker_frame[1])
            center_history.append((cX, cY))
            medX = int(np.median([p[0] for p in center_history]))
            medY = int(np.median([p[1] for p in center_history]))
            if found_this_frame and good_matches_count > 8:
                last_good_center = (medX, medY)

            # Расстояние до центра кадра
            distance = math.hypot(medX - center_frame[0], medY - center_frame[1])
            cv2.line(frame, (medX, medY), center_frame, (255,0,0), 2)
            cv2.circle(frame, (medX, medY), 7, (0,255,0), -1)
            cv2.putText(frame, f"Dist to center: {int(distance)} px", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Наложение мухи
            # вычисляем ширину метки в пикселях (среднее из верхней и нижней стороны)
            corners = corners_frame.reshape(4,2)
            top_width = np.linalg.norm(corners[3] - corners[0])   # верх
            bottom_width = np.linalg.norm(corners[2] - corners[1]) # низ
            marker_pixel_width = (top_width + bottom_width) / 2.0

            # Размер мухи пропорционален ширине метки
            fly_display_w = max(30, int(marker_pixel_width * fly_scale))  # не меньше 30px
            fly_display_h = int(fly_display_w * (fly_h / fly_w))

            # Изменяем размер изображения мухи
            new_fly = cv2.resize(fly_img, (fly_display_w, fly_display_h), interpolation=cv2.INTER_AREA)

            # Координаты прямоугольника мухи
            x1 = medX - fly_display_w // 2
            y1 = medY - fly_display_h // 2
            x2 = x1 + fly_display_w
            y2 = y1 + fly_display_h

            # Наложение
            overlay = np.zeros_like(frame)
            fly_x1 = max(0, x1)
            fly_y1 = max(0, y1)
            fly_x2 = min(w_frame, x2)
            fly_y2 = min(h_frame, y2)

            src_x1 = fly_x1 - x1
            src_y1 = fly_y1 - y1
            src_x2 = src_x1 + (fly_x2 - fly_x1)
            src_y2 = src_y1 + (fly_y2 - fly_y1)

            roi = overlay[fly_y1:fly_y2, fly_x1:fly_x2]
            fly_region = new_fly[src_y1:src_y2, src_x1:src_x2]

            if new_fly.shape[2] == 4:
                alpha = fly_region[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                overlay[fly_y1:fly_y2, fly_x1:fly_x2] = (fly_region[:, :, :3] * alpha + roi * (1 - alpha)).astype(
                    np.uint8)
            else:
                fly_region = overlay[fly_y1:fly_y2, fly_x1:fly_x2]

            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        # Отладка
        cv2.drawMarker(frame, center_frame, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"Good matches: {good_matches_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        status = "OK" if found_this_frame else "LOST"
        cv2.putText(frame, f"Status: {status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if found_this_frame else (0,0,255), 1)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


transform_initial_image('variant-7.jpg')
track_marker('ref-point.jpg','fly64.png',0.4)