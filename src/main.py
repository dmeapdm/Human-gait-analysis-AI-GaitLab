#Análisis completo con gráficas individuales rodilla/cadera/tobillo.

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Función para encontrar las cámaras externas
def find_external_cameras():
    external_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            external_cameras.append(i)
        cap.release()
    return external_cameras

# Función para calcular ángulos entre tres puntos
def calculate_angle(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Función para graficar los resultados
def plot_angles(times, hip_angles, knee_angles, ankle_angles, session_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(times, hip_angles, label="Cadera", color="blue")
    plt.plot(times, knee_angles, label="Rodilla", color="red")
    plt.plot(times, ankle_angles, label="Tobillo", color="green")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (grados)")
    plt.title("Análisis de la marcha: Cadera, Rodilla y Tobillo")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(session_folder, "grafica_compartida.png"))
    plt.show()

    # Gráficas individuales
    plt.figure(figsize=(8, 4))
    plt.plot(times, hip_angles, color="blue")
    plt.title("Ángulo de Cadera")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (grados)")
    plt.grid()
    plt.savefig(os.path.join(session_folder, "grafica_cadera.png"))
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(times, knee_angles, color="red")
    plt.title("Ángulo de Rodilla")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (grados)")
    plt.grid()
    plt.savefig(os.path.join(session_folder, "grafica_rodilla.png"))
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(times, ankle_angles, color="green")
    plt.title("Ángulo de Tobillo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (grados)")
    plt.grid()
    plt.savefig(os.path.join(session_folder, "grafica_tobillo.png"))
    plt.show()

# Función principal de procesamiento
def process_video(camera_index, session_folder):
    cap = cv2.VideoCapture(camera_index)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    drawing_utils = mp.solutions.drawing_utils

    hip_angles = []
    knee_angles = []
    ankle_angles = []
    times = []

    start_time = datetime.datetime.now()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Coordenadas de interés
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            # Ángulos
            hip_angle = calculate_angle(knee, hip, [hip[0], hip[1] - 1])  # Vertical como referencia
            knee_angle = calculate_angle(ankle, knee, hip)
            ankle_angle = calculate_angle(foot, ankle, knee)

            current_time = (datetime.datetime.now() - start_time).total_seconds()
            hip_angles.append(hip_angle)
            knee_angles.append(knee_angle)
            ankle_angles.append(ankle_angle)
            times.append(current_time)

            # Fases de la marcha (simplificadas)
            phase = "Desconocida"
            if hip_angle < 50 and knee_angle > 170:
                phase = "Contacto inicial"
            elif hip_angle > 60 and knee_angle > 150:
                phase = "Apoyo medio"
            elif hip_angle > 50 and knee_angle < 90:
                phase = "Balanceo"
            
            cv2.putText(frame, f"Fase: {phase}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(f"Camera {camera_index}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar y graficar resultados
    plot_angles(times, hip_angles, knee_angles, ankle_angles, session_folder)

# Main
if __name__ == "__main__":
    external_cameras = find_external_cameras()

    if external_cameras:
        session_folder = datetime.datetime.now().strftime("Analisis_Marcha_%Y-%m-%d_%H-%M-%S")
        os.makedirs(session_folder, exist_ok=True)
        for camera_index in external_cameras:
            print(f"Procesando cámara {camera_index}")
            process_video(camera_index, session_folder)
    else:
        print("No se detectaron cámaras externas.")
