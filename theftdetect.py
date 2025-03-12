import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
from datetime import datetime
import time
import keyboard
import multiprocessing
from collections import deque
from queue import Queue

def write_video(queue, video_writer, stop_event, fps):
    frame_interval = 1 / fps
    last_frame_time = time.time()
    while not stop_event.is_set() or not queue.empty():
        try:
            frame = queue.get(timeout=1)
            video_writer.write(frame)
            elapsed_time = time.time() - last_frame_time
            remaining_time = frame_interval - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)
            last_frame_time = time.time()
        except:
            pass
    video_writer.release()
def is_within_region(box, region):
    x1, y1, x2, y2 = box.xyxy[0]
    return (x1 >= region['x_min'] and y1 >= region['y_min'] and x2 <= region['x_max'] and y2 <= region['y_max'])

def run_yolo_screen1(model_path, conf_value, fps_value, save_folder, output_box, stop_event):
    model = YOLO(model_path)
    sct = mss()
    monitor = sct.monitors[1]
    os.makedirs(save_folder, exist_ok=True)
    pre_frames = deque(maxlen=500)
    is_recording = False
    recording_buffer_time = 30
    last_detected_time = 0
    video_queue = Queue(maxsize=300)
    video_writer = None
    region_of_interest = {
        'x_min': 505,
        'y_min': 105,
        'x_max': 1215,
        'y_max': 1080 
    }
    while not stop_event.is_set():
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        pre_frames.append(frame)
        results = model.predict(source=frame, imgsz=480, conf=conf_value, show=False)
        annotated_frame = results[0].plot()
        if len(results[0].boxes) == 0:
            output_box.insert(tk.END, "no detected on screen 1\n")
            if is_recording and time.time() - last_detected_time > recording_buffer_time:
                output_box.insert(tk.END, "Stopping recording as no object detected in buffer time\n")
                is_recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
        else:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = box.conf[0]
                if is_within_region(box, region_of_interest):
                    if class_name == "SUSPECT" and confidence >= 0.3:
                        last_detected_time = time.time()
                        if not is_recording:
                            output_box.insert(tk.END, f"SUSPECT detected on screen 1 with confidence {confidence:.2f}\n")
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            video_path = os.path.join(save_folder, f'suspect_screen1_video_{timestamp}.avi')
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            video_writer = cv2.VideoWriter(video_path, fourcc, fps_value, (frame.shape[1], frame.shape[0]))
                            is_recording = True
                            output_box.insert(tk.END, f"กำลังบันทึกวิดีโอ suspect on screen 1 ไปยัง: {video_path}\n")
                            record_thread = threading.Thread(target=write_video, args=(video_queue, video_writer, stop_event, fps_value))
                            record_thread.start()
                        video_queue.put(annotated_frame)
                else:
                    output_box.insert(tk.END, "Object detected outside the region on screen 1\n")
        output_box.see(tk.END)
        if keyboard.is_pressed('q'):
            output_box.insert(tk.END, "หยุดการทำงานบนจอ 1 เนื่องจากกด 'q'\n")
            output_box.see(tk.END)
            break
    cv2.destroyAllWindows()

def run_yolo_screen2(model_path, conf_value, fps_value, save_folder, output_box, stop_event):
    model = YOLO(model_path)
    sct = mss()
    monitor = sct.monitors[2]
    os.makedirs(save_folder, exist_ok=True)
    pre_frames = deque(maxlen=int(fps_value * 20))
    post_frames = deque(maxlen=int(fps_value * 20))
    is_recording = False
    video_queue = Queue(maxsize=500)
    video_writer = None
    recording_buffer_time = 30
    last_detected_time = 0
    
    while not stop_event.is_set():
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        pre_frames.append(frame)
        results = model.predict(source=frame, imgsz=480, conf=conf_value, show=False)
        annotated_frame = results[0].plot()
        if len(results[0].boxes) == 0:
            output_box.insert(tk.END, "no detected on screen 2\n")
            if is_recording and time.time() - last_detected_time > recording_buffer_time:
                output_box.insert(tk.END, "Stopping recording after buffer time expired.\n")
                while post_frames:
                    video_queue.put(post_frames.popleft())
                is_recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
        else:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = box.conf[0]
                if class_name == "STEAL" and confidence >= 0.75:
                    last_detected_time = time.time()
                    if not is_recording:
                        output_box.insert(tk.END, f"STEAL detected on screen 2 with confidence {confidence:.2f}\n")
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        video_path = os.path.join(save_folder,
                                                  f'steal_screen2_video_{timestamp}.avi')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps_value, (frame.shape[1], frame.shape[0]))
                        is_recording = True
                        output_box.insert(tk.END, f"กำลังบันทึกวิดีโอ steal on screen 2 ไปยัง: {video_path}\n")
                        record_thread = threading.Thread(target=write_video,
                                                         args=(video_queue, video_writer, stop_event, fps_value))
                        record_thread.start()
                        for pf in pre_frames:
                            video_queue.put(pf)
                    video_queue.put(annotated_frame)
                else:
                    if is_recording:
                        post_frames.append(annotated_frame)
        output_box.see(tk.END)
        if keyboard.is_pressed('q'):
            output_box.insert(tk.END, "หยุดการทำงานบนจอ 2 เนื่องจากกด 'q'\n")
            output_box.see(tk.END)
            break
    cv2.destroyAllWindows()

def open_gui():
    def select_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("YOLO Model", "*.pt")],
            title="เลือกไฟล์โมเดล YOLO (.pt)")
        if file_path:
            model_entry.delete(0, tk.END)
            model_entry.insert(0, file_path)

    def select_save_folder():
        folder_path = filedialog.askdirectory(
            title="เลือกโฟลเดอร์ที่ต้องการบันทึกภาพ"
        )
        if folder_path:
            save_folder_entry.delete(0, tk.END)
            save_folder_entry.insert(0, folder_path)

    def start_detection():
        model_path = model_entry.get()
        save_folder = save_folder_entry.get()

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "กรุณาเลือกไฟล์โมเดลที่ถูกต้อง")
            return

        if not os.path.exists(save_folder):
            messagebox.showerror("Error", "กรุณาเลือกโฟลเดอร์สำหรับบันทึกภาพ")
            return

        try:
            conf_value = float(conf_entry.get()) / 100
        except ValueError:
            messagebox.showerror("Error", "กรุณากรอกค่าความเชื่อมั่นเป็นตัวเลข")
            return
        try:
            fps_value = float(fps_entry.get())
        except ValueError:
            messagebox.showerror("Error", "กรุณากรอกค่า FPS เป็นตัวเลข")
            return
        stop_event.clear()
        threading.Thread(target=run_yolo_screen1,
                         args=(model_path, conf_value, fps_value, save_folder, output_box, stop_event)).start()
        threading.Thread(target=run_yolo_screen2,
                         args=(model_path, conf_value, fps_value, save_folder, output_box, stop_event)).start()

    def stop_detection():
        stop_event.set()
        output_box.insert(tk.END, "หยุดการทำงาน\n")
        output_box.see(tk.END)
    root = tk.Tk()
    root.title("YOLO Detection Settings")
    root.iconbitmap('icon.ico')
    model_label = tk.Label(root, text="เลือกไฟล์โมเดล YOLO (.pt):")
    model_label.pack(pady=10)
    model_entry = tk.Entry(root, width=50)
    model_entry.pack(pady=5)
    model_button = tk.Button(root, text="Browse", command=select_file)
    model_button.pack(pady=5)
    save_folder_label = tk.Label(root, text="เลือกโฟลเดอร์สำหรับบันทึกภาพ:")
    save_folder_label.pack(pady=10)
    save_folder_entry = tk.Entry(root, width=50)
    save_folder_entry.pack(pady=5)
    save_folder_button = tk.Button(root, text="Browse", command=select_save_folder)
    save_folder_button.pack(pady=5)
    conf_label = tk.Label(root, text="Confidence Threshold (ค่า 0-100):")
    conf_label.pack(pady=10)
    conf_entry = tk.Entry(root, width=10)
    conf_entry.insert(0, "75")
    conf_entry.pack(pady=5)
    fps_label = tk.Label(root, text="FPS (ค่า 1-60):")
    fps_label.pack(pady=10)
    fps_entry = tk.Entry(root, width=10)
    fps_entry.insert(0, "24")
    fps_entry.pack(pady=5)
    start_button = tk.Button(root, text="เริ่มการตรวจจับ", command=start_detection)
    start_button.pack(pady=10)
    stop_button = tk.Button(root, text="หยุดการตรวจจับ", command=stop_detection)
    stop_button.pack(pady=10)
    output_box = scrolledtext.ScrolledText(root, width=60, height=15, state='normal')
    output_box.pack(pady=10)
    global stop_event
    stop_event = threading.Event()
    root.mainloop()
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    open_gui()