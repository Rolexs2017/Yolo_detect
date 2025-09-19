import cv2
import threading
import time

class LowLatencyCamera:
    def __init__(self, sources, buffer_size=2):
        self.cap = cv2.VideoCapture(sources)
        self.latest_frame = None
        self.running = False
        self.thread = None
        
        # Оптимальные настройки для минимальной задержки
        self._setup_camera()
        
    def _setup_camera(self):
        
        # Устанавливаем FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Пытаемся уменьшить буфер
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
    
    def start(self):
        """Запуск непрерывного захвата в отдельном потоке"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1)  # Даем время на запуск
    
    def _capture_loop(self):
        """Основной цикл захвата"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            time.sleep(0.001)  # Минимальная пауза
    
    def get_frame(self):
        """Получение последнего кадра"""
        return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Остановка захвата"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()