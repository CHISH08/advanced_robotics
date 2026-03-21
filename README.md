# Домашние задания по Advanced Robotics

| № | Тема | Папка | README | Ноутбук | Краткое описание | Данные |
|---|---|---|---|---|---|---|
| 1.1 | Детекция стен (2D LIDAR + Hough) | `hw1_hough_ransac` | [README](hw1_hough_ransac/README.md) | [wall_detection_hough.ipynb](hw1_hough_ransac/wall_detection_hough.ipynb) | Генерация синтетических 2D lidar-сцен, построение Hough accumulator, поиск пиков и визуализация линий стен. | Генерируются внутри ноутбука |
| 1.2 | Детекция земли (3D + RANSAC) | `hw1_hough_ransac` | [README](hw1_hough_ransac/README.md) | [ground_detection_ransac.ipynb](hw1_hough_ransac/ground_detection_ransac.ipynb) | Реконструкция 3D облака по стереопаре KITTI и выделение плоскости земли собственной реализацией RANSAC. | `hw1_hough_ransac/data/3d` |
| 2 | Оценка траектории (линейный фильтр Калмана) | `hw2_linear_kalman` | [README](hw2_linear_kalman/README.md) | [smartphone_linear_kalman.ipynb](hw2_linear_kalman/smartphone_linear_kalman.ipynb) | Объединение GPS и акселерометра смартфона в 1D линейном фильтре Калмана для оценки положения и скорости. | `hw2_linear_kalman/2026-03-0916.48.43.csv` |
| 3 | Оценка ориентации (Euler vs Quaternion EKF) | `hw3_ekf_orientation` | [README](hw3_ekf_orientation/README.md) | [euler_vs_quaternion_ekf.ipynb](hw3_ekf_orientation/euler_vs_quaternion_ekf.ipynb) | Сравнение двух EKF-представлений ориентации смартфона: углы Эйлера и кватернионы. | `hw3_ekf_orientation/data.csv` |
| 4 | Gaussian Splatting | `hw4_gaussian_splatting` | [README](hw4_gaussian_splatting/README.md) | [gaussian_splatting.ipynb](hw4_gaussian_splatting/gaussian_splatting.ipynb) | Подготовка кадров, SfM/COLMAP, обучение 3D Gaussian Splatting и анализ качества рендера. | `hw4_gaussian_splatting/video.mp4` и артефакты в папке задания |
