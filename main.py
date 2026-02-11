import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from sklearn.cluster import DBSCAN
from rplidar import RPLidar
import psycopg2
import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque
import threading
import time
import datetime
import queue
import struct
import warnings

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import serial.tools.list_ports

# ================= CONFIGURATION =================
PORT_NAME = 'COM4'       
BAUD_RATE = 115200
MAX_DISTANCE = 3000      # ZOOM (3m)
MIN_DISTANCE = 100

CLUSTER_EPS = 500          # Distanța maximă (mm) între puncte pentru a fi grupate într-un singur obiect (DBSCAN).
CLUSTER_MIN_SAMPLES = 7    # Numărul minim de puncte laser necesare pentru a confirma un obiect real (elimină zgomotul/praf).
THRESHOLD = 400            # Diferența minimă (mm) față de peretele învățat pentru a considera că e "ceva nou" (sensibilitatea).
MOTION_PERSISTENCE = 5     # Numărul de cadre consecutive în care trebuie să apară un intrus pentru a declanșa alarma (evită alarme false).
MOTION_DECAY_TIME = 5.0    # Timpul (secunde) cât sistemul rămâne în alertă/înregistrează după ce mișcarea s-a oprit.
PRE_MOTION_BUFFER = 5.0    # (Teoretic) Câte secunde de istoric se păstrează înainte de declanșarea alarmei.
LEARN_RATE_DECAY = 0.9998  # Cât de greu se modifică fundalul (aproape de 1 = intrusul nu devine parte din perete dacă stă nemișcat).
ANIM_INTERVAL_MS = 60      # Viteza de actualizare a interfeței grafice (30ms ≈ 33 FPS).
LIVE_UPLOAD_INTERVAL = 0.1 # Intervalul (secunde) la care se trimit datele spre baza de date online (evită blocarea rețelei).
TRACKING_MAX_DIST = 700   # Distanța maximă (mm) în care trackerul caută un obiect vechi pentru a-l asocia cu poziția nouă (raza de predicție).

RAILWAY_URL = "#add your railway table URL"

# --- EMAIL CONFIGURATION ---  # add your sepcifications for email warning
EMAIL_SENDER = "example@gmail.com"
EMAIL_PASSWORD = "google accounts genjerated security passworf"  # Google App Password, NU parola normala
EMAIL_RECEIVER = "example@gmail.com"

# ================= KALMAN FILTER & TRACKING ENGINE =================
class KalmanObject:
    def __init__(self, x, y, id_counter):
        self.id = id_counter
        self.state = np.array([x, y, 0, 0], dtype=float)
        self.P = np.eye(4) * 500 
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 50 
        self.active_timer = 0 # Frames seen
        self.missing_frames = 0 # Frames lost

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)
        return self.state[0], self.state[1]

    def update(self, mx, my):
        z = np.array([mx, my])
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        self.active_timer += 1
        self.missing_frames = 0

class Tracker:
    def __init__(self):
        self.tracks = []
        self.id_counter = 0

    def update(self, detections, dt):
        # 1. Predict all existing tracks
        for trk in self.tracks:
            trk.predict(dt)
            trk.missing_frames += 1

        # 2. Association (Simple Euclidean Distance)
        if len(detections) > 0:
            assigned_tracks = []
            
            for dx, dy in detections:
                best_dist = TRACKING_MAX_DIST
                best_trk = None
                
                # Find closest track
                for trk in self.tracks:
                    if trk in assigned_tracks: continue
                    tx, ty = trk.state[0], trk.state[1]
                    dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_trk = trk
                
                if best_trk:
                    best_trk.update(dx, dy)
                    assigned_tracks.append(best_trk)
                else:
                    # New Object Found
                    self.id_counter += 1
                    new_trk = KalmanObject(dx, dy, self.id_counter)
                    self.tracks.append(new_trk)

        # 3. Cleanup dead tracks (lost for > 5 frames)
        self.tracks = [t for t in self.tracks if t.missing_frames < 15]
        return self.tracks

# ================= DATABASE ENGINE =================
def clean_db_data(raw_data):
    if raw_data is None: return np.zeros(360)
    try:
        binary_data = b''
        if isinstance(raw_data, (memoryview, bytes)): binary_data = bytes(raw_data)
        elif isinstance(raw_data, str) and raw_data.startswith('\\x'): binary_data = bytes.fromhex(raw_data[2:])
        else: return np.zeros(360)
        
        expected_len = 360 * 4 
        if len(binary_data) != expected_len:
            if len(binary_data) > expected_len: binary_data = binary_data[:expected_len]
            else: return np.zeros(360)
        return np.array(struct.unpack(f'<360f', binary_data))
    except: return np.zeros(360)

class CloudDatabase:
    def __init__(self, db_url):
        self.db_url = db_url
        self.history_queue = queue.Queue()
        self.live_queue = queue.Queue(maxsize=1) 
        self.running = True
        self.remote_data_buffer = None
        self.remote_fetch_active = False

        threading.Thread(target=self.worker_history, daemon=True).start()
        threading.Thread(target=self.worker_live, daemon=True).start()
        threading.Thread(target=self.worker_remote_fetch, daemon=True).start() 
        threading.Thread(target=self.init_tables, daemon=True).start()

    def _get_connection(self):
        try: return psycopg2.connect(self.db_url, connect_timeout=3)
        except: return None

    def init_tables(self):
        conn = self._get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute('''CREATE TABLE IF NOT EXISTS security_events (
                        id SERIAL PRIMARY KEY, timestamp DOUBLE PRECISION,
                        scan_data BYTEA, has_motion BOOLEAN DEFAULT TRUE)''')
                cur.execute('''CREATE TABLE IF NOT EXISTS live_feed (
                        id INT PRIMARY KEY, timestamp DOUBLE PRECISION,
                        scan_data BYTEA, status TEXT)''')
                cur.execute("INSERT INTO live_feed (id, timestamp, status) VALUES (1, 0, 'OFFLINE') ON CONFLICT (id) DO NOTHING")
                conn.commit(); conn.close()
            except: pass

    def update_live_feed(self, scan_array, status):
        try:
            if len(scan_array) != 360: scan_array = np.pad(scan_array, (0, 360 - len(scan_array)), 'constant')
            binary = scan_array.astype(np.float32).tobytes()
            if not self.live_queue.empty():
                try: self.live_queue.get_nowait()
                except: pass
            self.live_queue.put((binary, status))
        except: pass

    def worker_remote_fetch(self):
        conn = None
        while self.running:
            if self.remote_fetch_active:
                if conn is None or conn.closed != 0: conn = self._get_connection()
                if conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT scan_data, status, timestamp FROM live_feed WHERE id=1")
                            row = cur.fetchone()
                        if row:
                            arr = clean_db_data(row[0])
                            if np.sum(arr) > 0: self.remote_data_buffer = {'scan': arr, 'status': row[1], 'time': row[2]}
                    except:
                        try: conn.close(); conn = None
                        except: pass
                time.sleep(0.05) 
            else:
                if conn: 
                    try: conn.close(); conn = None
                    except: pass
                time.sleep(0.5)

    def worker_live(self):
        conn = None
        while self.running:
            try:
                binary_data, status = self.live_queue.get(timeout=1)
                if conn is None or conn.closed != 0: conn = self._get_connection()
                if conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("UPDATE live_feed SET timestamp=%s, scan_data=%s, status=%s WHERE id=1", (time.time(), binary_data, status))
                        conn.commit()
                    except:
                        try: conn.close(); conn = None
                        except: pass
            except: pass

    def save_scan_async(self, scan_array, has_motion=True):
        if not self.running: return
        try:
            binary_data = scan_array.astype(np.float32).tobytes()
            self.history_queue.put((time.time(), binary_data, has_motion))
        except: pass

    def worker_history(self):
        while self.running:
            try:
                items = []
                try:
                    items.append(self.history_queue.get(timeout=1))
                    while not self.history_queue.empty() and len(items) < 10: items.append(self.history_queue.get_nowait())
                except queue.Empty: continue
                conn = self._get_connection()
                if conn:
                    try:
                        with conn.cursor() as cur:
                            args_str = ','.join(cur.mogrify("(%s,%s,%s)", (ts, data, mot)).decode('utf-8') for ts, data, mot in items)
                            cur.execute("INSERT INTO security_events (timestamp, scan_data, has_motion) VALUES " + args_str)
                        conn.commit(); conn.close()
                    except: pass
            except: time.sleep(1)

    def get_recording_sessions(self):
        conn = self._get_connection()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT timestamp FROM security_events ORDER BY timestamp DESC LIMIT 5000')
                rows = cur.fetchall()
            conn.close()
            if not rows: return []
            ts = sorted([r[0] for r in rows])
            sessions, start_t = [], ts[0]
            for i in range(1, len(ts)):
                if ts[i] - ts[i-1] > 30: 
                    sessions.append((start_t, ts[i-1]))
                    start_t = ts[i]
            sessions.append((start_t, ts[-1]))
            return sessions[::-1]
        except: return []

    def fetch_scans_window(self, s, e):
        conn = self._get_connection()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT timestamp, scan_data, has_motion FROM security_events WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp ASC', (s, e))
                rows = cur.fetchall()
            conn.close()
            return [{'time': r[0], 'scan': clean_db_data(r[1]), 'motion': r[2]} for r in rows]
        except: return []

    def delete_session(self, start_t, end_t):
        conn = self._get_connection()
        if not conn: return False
        try:
            cur = conn.cursor()
            cur.execute('DELETE FROM security_events WHERE timestamp >= %s AND timestamp <= %s', (start_t - 0.1, end_t + 0.1))
            conn.commit(); conn.close()
            return True
        except: return False

    def delete_all_data(self):
        conn = self._get_connection()
        if not conn: return False
        try:
            cur = conn.cursor()
            cur.execute('DELETE FROM security_events')
            conn.commit(); conn.close()
            return True
        except: return False

# ================= MAIN SYSTEM =================
class LidarSecuritySystem:
    def __init__(self):
        self.cloud = CloudDatabase(RAILWAY_URL)
        self.tracker = Tracker() # Initializare Tracker
        self.mode = 'LIVE' 
        self.paused = False
        self.last_upload_time = 0
        self.email_sent_this_event = False
        self.is_armed = False 

        self.fig = plt.figure(figsize=(11, 9), facecolor='#080808') 
        self.ax = self.fig.add_axes([0.05, 0.28, 0.9, 0.65], projection='polar', facecolor='#001515') 
        self.ax.set_ylim(0, MAX_DISTANCE)
        self.ax.grid(color='#005555', alpha=0.4, linestyle='--') 
        self.ax.set_yticklabels([]); self.ax.set_xticklabels([])
        self.ax.spines['polar'].set_visible(False)

        # Plot Elements
        self.scat_walls = self.ax.scatter([], [], s=25, c='#00FF44', alpha=0.6, label='Walls', animated=True) 
        self.scat_trail = self.ax.scatter([], [], s=40, c=[], edgecolors='none', animated=True)
        self.scat_intruder = self.ax.scatter([], [], s=180, c='#FF0000', edgecolors='white', linewidth=2, zorder=10, animated=True)
        self.scat_prediction = self.ax.plot([], [], color='#FFFF00', linewidth=2, linestyle=':', animated=True)[0]

        # UI Text
        self.text_status = self.ax.text(0.5, 1.15, "SYSTEM INITIALIZING", color="white", ha="center", 
                                        transform=self.ax.transAxes, fontsize=16, fontweight='bold', fontfamily='monospace', animated=True)
        
        # --- NEW: Text urias de ALERTA (initial invizibil) ---
        self.text_alert_huge = self.ax.text(0.5, 0.92, "", color="#FF0000", ha="center", va="center",
                                    transform=self.ax.transAxes, fontsize=15, fontweight='black', 
                                    alpha=1.0, animated=True, zorder=100)
        
        # TIME & DATE
        self.text_time = self.ax.text(0.5, -0.1, "--:--:--", color="#00FFFF", ha="center", 
                                      fontsize=18, fontfamily='monospace', fontweight='bold', transform=self.ax.transAxes, animated=True)
        self.text_date = self.ax.text(0.5, -0.2, "----/--/--", color="#00AAAA", ha="center", 
                                      fontsize=12, fontfamily='monospace', transform=self.ax.transAxes, animated=True)
        
        # --- BUTTONS ---
        ax_rec = self.fig.add_axes([0.1, 0.05, 0.25, 0.07])
        self.btn_rec = Button(ax_rec, 'RECORDINGS', color='#222', hovercolor='#333')
        self.btn_rec.label.set_color('white')
        self.btn_rec.label.set_weight('bold')
        self.btn_rec.on_clicked(self.handle_recordings_button)

        ax_rem = self.fig.add_axes([0.37, 0.05, 0.25, 0.07])
        self.btn_rem = Button(ax_rem, 'GO REMOTE', color='#003366', hovercolor='#004488')
        self.btn_rem.label.set_color('white')
        self.btn_rem.label.set_weight('bold')
        self.btn_rem.on_clicked(self.handle_remote_button)

        self.ax_pause = self.fig.add_axes([0.65, 0.05, 0.15, 0.07])
        self.btn_pause = Button(self.ax_pause, 'PAUSE', color='#440000', hovercolor='#660000')
        self.btn_pause.label.set_color('white')
        self.btn_pause.label.set_weight('bold')
        self.btn_pause.on_clicked(self.handle_pause_button)
        self.ax_pause.set_visible(False) 

        # --- NEW: BUTON ARM/DISARM ---
        ax_arm = self.fig.add_axes([0.82, 0.05, 0.13, 0.07])
        self.btn_arm = Button(ax_arm, 'ARM', color='#333300', hovercolor='#555500')
        self.btn_arm.label.set_color('white')
        self.btn_arm.label.set_weight('bold')
        self.btn_arm.on_clicked(self.handle_arm_button)

        # -- Logic --
        self.scan_current = np.zeros(360)
        self.lock = threading.Lock()
        self.lidar_running = False 
        self.lidar_connected = False 
        
        self.local_background_map = np.zeros(360)
        self.display_background_map = np.zeros(360)
        self.local_motion_counter = 0
        self.local_motion_active = False
        self.local_post_timer = 0.0
        self.trail_buffer = deque(maxlen=10) 
        self.replay_data = []
        self.replay_idx = 0
        self.replay_background = np.zeros(360) 

        self.start_lidar_thread()

    def start_lidar_thread(self):
        if self.lidar_running: return
        self.lidar_running = True
        self.text_status.set_text("BOOTING LIDAR...")
        threading.Thread(target=self.lidar_worker, daemon=True).start()

    def lidar_worker(self):
        try:
            lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE, timeout=3)
            lidar.stop(); lidar.stop_motor(); time.sleep(0.5); lidar.clean_input()
            self.lidar_connected = True
            
            for scan in lidar.iter_scans(max_buf_meas=2000, min_len=5):
                if not self.lidar_running: break
                temp_scan = np.zeros(360)
                for (_, angle, dist) in scan:
                    if MIN_DISTANCE < dist < MAX_DISTANCE: temp_scan[int(angle) % 360] = dist
                with self.lock:
                    mask = temp_scan > 0
                    self.scan_current[mask] = temp_scan[mask]
            lidar.stop(); lidar.stop_motor(); lidar.disconnect()
        except:
            print("LIDAR NOT FOUND. SWITCHING TO AUTO-REMOTE.")
            self.lidar_connected = False
            self.lidar_running = False
            self.mode = 'REMOTE' 
            self.cloud.remote_fetch_active = True
            self.display_background_map = np.zeros(360)

    def get_fluent_scan(self):
        with self.lock: raw = np.copy(self.scan_current)
        padded = np.pad(raw, (1, 1), mode='wrap') 
        window = np.lib.stride_tricks.as_strided(padded, shape=(raw.shape[0], 3), strides=padded.strides*2)
        return np.median(window, axis=1)

    def process_clusters(self, angles_rad, dists):
        # FIX AICI: Returnam 3 valori goale daca nu sunt destule puncte
        if len(angles_rad) < CLUSTER_MIN_SAMPLES: return [], [], []
        x, y = dists * np.cos(angles_rad), dists * np.sin(angles_rad)
        points = np.column_stack((x, y))
        try:
            db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(points)
            out_a, out_d = [], []
            # Return also the X,Y coordinates for Tracking
            out_xy = []
            unique_labels = set(db.labels_)
            if -1 in unique_labels: unique_labels.remove(-1)
            for k in unique_labels:
                mask = (db.labels_ == k)
                cx, cy = np.mean(x[mask]), np.mean(y[mask])
                out_d.append(np.sqrt(cx**2 + cy**2))
                out_a.append(np.arctan2(cy, cx) % (2 * np.pi))
                out_xy.append((cx, cy))
            return out_a, out_d, out_xy
        except: 
            # FIX AICI: Returnam 3 valori goale in caz de eroare
            return [], [], []

    def send_email_alert(self):
        if "adresa_ta" in EMAIL_SENDER: return 

        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = "ALERTĂ SECURITATE LIDAR"

        body = f"Sistemul LIDAR a detectat o mișcare la ora {datetime.datetime.now().strftime('%H:%M:%S')}.\nEvenimentul a fost salvat!"
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print("[EMAIL] Alerta trimisa cu succes!")
        except Exception as e:
            print(f"[EMAIL ERROR] Nu s-a putut trimite: {e}")

    def handle_arm_button(self, event):
        self.is_armed = not self.is_armed
        
        self.email_sent_this_event = False 

        if self.is_armed:
            self.btn_arm.label.set_text("DISARM")
            self.btn_arm.color = '#550000' # Rosu inchis (Armed)
            self.btn_arm.hovercolor = '#770000'
        else:
            self.btn_arm.label.set_text("ARM")
            self.btn_arm.color = '#333300' # Galben inchis (Disarmed)
            self.btn_arm.hovercolor = '#555500'
        
        # Forțăm redesenarea pentru a actualiza vizual butonul imediat
        self.fig.canvas.draw_idle()
        
    def update(self, frame):
        ts_now = time.time()
        dt = ANIM_INTERVAL_MS / 1000.0
        
        if self.mode == 'REMOTE' and self.btn_rem.label.get_text() == 'GO REMOTE':
            self.btn_rem.label.set_text("GO LIVE")

        # --- 1. LOCAL LOGIC (Motion Detection) ---
        if self.lidar_connected:
            local_data = self.get_fluent_scan()
            if np.sum(self.local_background_map) == 0: self.local_background_map = np.copy(local_data)
            else:
                valid_mask = local_data > 0
                self.local_background_map[valid_mask] = np.maximum(self.local_background_map[valid_mask] * LEARN_RATE_DECAY, local_data[valid_mask])
                diff = self.local_background_map - local_data
                intruder_mask = (local_data > 0) & (self.local_background_map > 0) & (diff > THRESHOLD)
                rads = np.deg2rad(np.arange(360))
                l_obj_a, l_obj_d, l_obj_xy = self.process_clusters(rads[intruder_mask], local_data[intruder_mask])
                
                # --- KALMAN TRACKING UPDATE ---
                self.tracker.update(l_obj_xy, dt)

                local_status = "SECURE"
                if len(l_obj_a) > 0:
                    self.local_motion_counter += 1
                    if self.local_motion_counter >= MOTION_PERSISTENCE:
                        local_status = "INTRUDER"
                        self.local_motion_active = True
                        self.local_post_timer = MOTION_DECAY_TIME
                        self.cloud.save_scan_async(local_data, True)
                        
                        # --- MODIFICARE AICI: Trimite email doar daca e ARMED ---
                        if self.is_armed and not self.email_sent_this_event:
                            threading.Thread(target=self.send_email_alert, daemon=True).start()
                            self.email_sent_this_event = True
                else:
                    self.local_motion_counter = max(0, self.local_motion_counter - 1)
                    if self.local_motion_active:
                        local_status = "BUFFERING"
                        self.local_post_timer -= dt
                        self.cloud.save_scan_async(local_data, False)
                        if self.local_post_timer <= 0: 
                            self.local_motion_active = False
                            self.email_sent_this_event = False 
                
                if ts_now - self.last_upload_time > LIVE_UPLOAD_INTERVAL:
                    self.cloud.update_live_feed(local_data, local_status); self.last_upload_time = ts_now

        # --- 2. DISPLAY LOGIC ---
        display_data, current_display_status = None, "OFFLINE"

        if self.mode == 'LIVE':
            if not self.lidar_connected:
                self.text_status.set_text("NO HARDWARE")
                return self.scat_walls, self.scat_intruder, self.scat_trail, self.scat_prediction, self.text_status, self.text_time, self.text_date, self.text_alert_huge
            display_data = self.get_fluent_scan() 
            now = datetime.datetime.now()
            self.text_time.set_text(now.strftime("%H:%M:%S"))
            self.text_date.set_text(now.strftime("%Y-%m-%d"))

        elif self.mode == 'REMOTE':
            remote_packet = self.cloud.remote_data_buffer
            if remote_packet:
                display_data, current_display_status = remote_packet['scan'], remote_packet['status']
                dt_obj = datetime.datetime.fromtimestamp(remote_packet['time'])
                self.text_time.set_text(dt_obj.strftime("%H:%M:%S"))
                self.text_date.set_text(dt_obj.strftime("%Y-%m-%d"))
            else:
                self.text_status.set_text("CONNECTING...")
                return self.scat_walls, self.scat_intruder, self.scat_trail, self.scat_prediction, self.text_status, self.text_time, self.text_date, self.text_alert_huge

        if display_data is not None:
            if np.sum(self.display_background_map) == 0: self.display_background_map = np.copy(display_data)
            valid_mask = display_data > 0
            self.display_background_map[valid_mask] = np.maximum(self.display_background_map[valid_mask] * LEARN_RATE_DECAY, display_data[valid_mask])
            diff = self.display_background_map - display_data
            intruder_mask = (display_data > 0) & (self.display_background_map > 0) & (diff > THRESHOLD)
            rads = np.deg2rad(np.arange(360))
            obj_a, obj_d, _ = self.process_clusters(rads[intruder_mask], display_data[intruder_mask])
            
            if self.mode == 'LIVE':
                if len(obj_a) > 0 and self.local_motion_counter >= MOTION_PERSISTENCE: current_display_status = "INTRUDER"
                elif self.local_motion_active: current_display_status = "BUFFERING"
                else: current_display_status = "SECURE"

            self.text_status.set_text(f"{'REMOTE: ' if self.mode=='REMOTE' else ''}{current_display_status}")
            self.text_status.set_color('#FF0000' if "INTRUDER" in current_display_status else ('#FFA500' if "BUFFER" in current_display_status else '#00FF00'))
            self.ax.set_facecolor('#150000' if "INTRUDER" in current_display_status else '#001515')
            
            self.scat_walls.set_offsets(np.c_[np.deg2rad(np.arange(360)), self.display_background_map])
            if len(obj_a) > 0:
                self.scat_intruder.set_offsets(np.c_[obj_a, obj_d])
                self.trail_buffer.append(np.c_[obj_a, obj_d])
            else: self.scat_intruder.set_offsets(np.empty((0, 2))); self.trail_buffer.append(np.empty((0, 2)))

            # --- RENDER PREDICTION LINES (YELLOW) ---
            if self.mode == 'LIVE' and len(self.tracker.tracks) > 0:
                pred_x, pred_y = [], []
                for t in self.tracker.tracks:
                    tx, ty = t.state[0], t.state[1]
                    vx, vy = t.state[2], t.state[3]
                    future_x, future_y = tx + vx*0.5, ty + vy*0.5 
                    
                    dist_now = np.sqrt(tx**2 + ty**2)
                    ang_now = np.arctan2(ty, tx) % (2*np.pi)
                    dist_fut = np.sqrt(future_x**2 + future_y**2)
                    ang_fut = np.arctan2(future_y, future_x) % (2*np.pi)
                    
                    pred_x.extend([ang_now, ang_fut, np.nan]) 
                    pred_y.extend([dist_now, dist_fut, np.nan])
                
                self.scat_prediction.set_data(pred_x, pred_y)
            else:
                self.scat_prediction.set_data([], [])

        elif self.mode == 'REPLAY':
            if self.replay_data:
                frame_data = self.replay_data[min(self.replay_idx, len(self.replay_data)-1)]
                scan = frame_data['scan']
                if self.replay_idx == 0: self.replay_background = np.copy(scan)
                else: self.replay_background[scan > self.replay_background] = scan[scan > self.replay_background]
                diff = self.replay_background - scan
                rads = np.deg2rad(np.arange(360))
                viz_obj_a, viz_obj_d, _ = self.process_clusters(rads[(scan > 0) & (diff > THRESHOLD)], scan[(scan > 0) & (diff > THRESHOLD)])
                
                self.scat_walls.set_offsets(np.c_[rads, self.replay_background])
                if len(viz_obj_a) > 0: self.scat_intruder.set_offsets(np.c_[viz_obj_a, viz_obj_d])
                else: self.scat_intruder.set_offsets(np.empty((0, 2)))
                
                self.text_status.set_text("PLAYBACK: ALARM" if frame_data['motion'] else "PLAYBACK: BUFFER")
                dt_obj = datetime.datetime.fromtimestamp(frame_data['time'])
                self.text_time.set_text(dt_obj.strftime("%H:%M:%S"))
                self.text_date.set_text(dt_obj.strftime("%Y-%m-%d"))

                if not self.paused: self.replay_idx = min(self.replay_idx + 1, len(self.replay_data)-1)

        pts = [p for p in self.trail_buffer if len(p) > 0]
        if pts:
            self.scat_trail.set_offsets(np.vstack(pts))
            self.scat_trail.set_facecolors(np.vstack([np.full((len(p), 4), [1.0, 0.5, 0.0, (i+1)/20]) for i, p in enumerate(pts)]))
        else: self.scat_trail.set_offsets(np.empty((0, 2)))
        
        # --- UPDATE BIG ALERT TEXT ---
        if self.is_armed and "INTRUDER" in current_display_status:
            self.text_alert_huge.set_text("ALERT")
        else:
            self.text_alert_huge.set_text("")

        return self.scat_walls, self.scat_intruder, self.scat_trail, self.scat_prediction, self.text_status, self.text_time, self.text_date, self.text_alert_huge
    
    def handle_remote_button(self, event):
        if self.mode == 'REMOTE':
            self.mode = 'LIVE'
            self.btn_rem.label.set_text("GO REMOTE")
            self.btn_rec.label.set_text("RECORDINGS")
            self.cloud.remote_fetch_active = False
            self.ax_pause.set_visible(False)
            self.start_lidar_thread() 
        else:
            self.mode = 'REMOTE'
            self.btn_rem.label.set_text("GO LIVE")
            self.cloud.remote_fetch_active = True
            self.display_background_map = np.zeros(360) 
            self.ax_pause.set_visible(False)

    def handle_recordings_button(self, event):
        if self.mode == 'REPLAY':
            self.mode = 'LIVE'
            self.btn_rec.label.set_text("RECORDINGS")
            self.btn_rem.label.set_text("GO REMOTE")
            self.ax_pause.set_visible(False)
            self.lidar_running = True; self.start_lidar_thread()
        else:
            self.lidar_running = False 
            sessions = self.cloud.get_recording_sessions()
            root = tk.Tk(); root.withdraw()
            selection = self.show_selector(sessions)
            root.destroy()
            
            if selection:
                self.mode = 'REPLAY'
                self.btn_rec.label.set_text("EXIT REPLAY")
                self.ax_pause.set_visible(True)
                self.paused = False
                self.replay_data = self.cloud.fetch_scans_window(selection[0], selection[1])
                self.replay_idx = 0; self.replay_background = np.zeros(360)
            else:
                self.lidar_running = True; self.start_lidar_thread() 

    def handle_pause_button(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text("RESUME" if self.paused else "PAUSE")

    def show_selector(self, sessions):
        res = {"val": None}
        win = tk.Toplevel(); win.title("SECURITY LOGS"); win.geometry("500x400")
        win.configure(bg="#222")
        
        lbl = tk.Label(win, text="AVAILABLE SESSIONS", bg="#222", fg="white", font=("Arial", 12, "bold"))
        lbl.pack(pady=10)

        frame_lb = tk.Frame(win); frame_lb.pack(fill="both", expand=True, padx=15)
        lb = tk.Listbox(frame_lb, bg="#111", fg="#00FF00", font=("Consolas", 10), selectbackground="#004400")
        lb.pack(side="left", fill="both", expand=True)
        sc = tk.Scrollbar(frame_lb, command=lb.yview); sc.pack(side="right", fill="y")
        lb.config(yscrollcommand=sc.set)

        for s, e in sessions:
            dur = e - s
            lb.insert(tk.END, f" DATE: {datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')} | TIME: {datetime.datetime.fromtimestamp(s).strftime('%H:%M:%S')} | DUR: {dur:.1f}s")

        def load():
            if lb.curselection(): res["val"] = sessions[lb.curselection()[0]]; win.destroy()
        
        def delete_one():
            if not lb.curselection(): return
            idx = lb.curselection()[0]
            sel = sessions[idx]
            if messagebox.askyesno("DELETE", "Delete this recording session?", parent=win):
                if self.cloud.delete_session(sel[0], sel[1]):
                    lb.delete(idx); sessions.pop(idx)
        
        def delete_all():
            if messagebox.askyesno("NUKE", "DELETE ALL DATA FOREVER?", parent=win):
                self.cloud.delete_all_data(); win.destroy()

        btn_frame = tk.Frame(win, bg="#222"); btn_frame.pack(fill="x", pady=10, padx=15)
        tk.Button(btn_frame, text="PLAY SESSION", command=load, bg="#004400", fg="white", font=("Arial", 10, "bold"), width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="DELETE SELECTED", command=delete_one, bg="#444", fg="white", font=("Arial", 10), width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="DELETE ALL", command=delete_all, bg="#550000", fg="white", font=("Arial", 10, "bold")).pack(side="right")

        win.transient(); win.grab_set(); win.wait_window()
        return res["val"]

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=ANIM_INTERVAL_MS, blit=True, cache_frame_data=False)
        plt.show()
        self.lidar_running = False; self.cloud.running = False

if __name__ == "__main__":
    LidarSecuritySystem().run()
