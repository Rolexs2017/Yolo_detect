import sys
import os
import cv2
from ultralytics import YOLO
import time
import numpy as np
import threading
from datetime import datetime
from datetime import date
from collections import defaultdict, deque
from opcua import Server
from Modbus_new import ModbusManager
#from kam_sdk import CameraCapture
from Low_latency import LowLatencyCamera
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AI_Detection_System:
    def __init__(self):
        # Конфигурация
        self.CAMERA_INDEX = 0
        self.size_tail = 512
        self.speed_control = False
        self.det = True
        self.gray = False
        self.device = 'cuda'
        self.model_path = "yolo12n.pt"
        self.model_gray_path = "best_n_gray.pt"
        #self.model_gray_path = "best_n_standart.pt"
        self.BACKGROUND_PATH = "background.jpg"
        
        # Переменные состояния
        self.frame_count = 0
        self.speed = 0
        self.metr = 0
        self.get = 0
        self.current_fps = 0
        self.background = None
        self.start_time = time.time()
        
        # Инициализация компонентов
        self.model = None
        self.camera = None
        self.modbus_manager = None
        self.opc_server = None
        self.variables = {}
        self.in_registers = []
        self.out_registers = []
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        # Очередь для кадров
        self.frame_queue = deque(maxlen=5)
   
    def initialize_system(self):
        """Инициализация всей системы"""
        try:
            self._create_dir()
            self._initialize_model()
            self._initialize_camera()
            self._initialize_modbus()
            #self._initialize_opc_server()
            logger.info("Система успешно инициализирована")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации системы: {e}")
            return False
            
    def _create_dir(self):   
        # Создание директорий
        self.dir_name = datetime.now().strftime("%d_%m_%y")
        self.output_dir = f"D://{self.dir_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Директории созданны ")
    
    def _initialize_model(self):
        """Инициализация модели YOLO"""
        model_path = self.model_gray_path if self.gray else self.model_path
        self.model = YOLO(model_path)
        logger.info(f"Модель загружена: {'Gray' if self.gray else 'Color'}")
    
    def _initialize_camera(self):
        """Инициализация камеры"""
        self.camera = LowLatencyCamera(self.CAMERA_INDEX)
        logger.info("Камера инициализирована")
    
    def _initialize_modbus(self):
        """Инициализация Modbus"""
        try:
            self.modbus_manager = ModbusManager()
            self.modbus_manager.host = "192.168.3.38"
            self.modbus_manager.port = 502
            self.modbus_manager.connect()
            self.modbus_manager.start()
            logger.info("Modbus подключен")
        except Exception as e:
            logger.warning(f"Modbus не подключен: {e}")
            self.modbus_manager = None
    
    def _initialize_opc_server(self):
        """Инициализация OPC сервера"""
        try:
            self.opc_server = Server()
            self.opc_server.set_endpoint("opc.tcp://192.168.3.167:4840/ivcore-ai-detect/server/")
            uri = "http://ivcore-ai-detect.com"
            idx = self.opc_server.register_namespace(uri)
            objects = self.opc_server.get_objects_node()
            myobj = objects.add_object(idx, "Classes")
            
            if self.model:
                classes = self.model.names
                for class_id, class_name in classes.items():
                    var = myobj.add_variable(idx, class_name, 0.0)
                    var.set_writable()
                    self.variables[class_name] = var
            
            self.opc_server.start()
            logger.info("OPC сервер запущен")
        except Exception as e:
            logger.warning(f"OPC сервер не запущен: {e}")
            self.opc_server = None
    
    def capture_frame(self):
        """Захват кадра с камеры"""
        if not self.camera:
            return None
        try:
            #frame = self.camera.capture_frame()
            frame = self.camera.get_frame()
            if frame is not None:
                frame = cv2.flip(frame, 0)
                if self.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        except Exception as e:
            logger.error(f"Ошибка захвата кадра: {e}")
            return None
    
    def split_frame_to_tiles(self, frame, tile_size):
        """Разделение кадра на тайлы"""
        if frame is None:
            return [], [], 0, 0
            
        tiles = []
        tile_positions = []  # Сохраняем позиции тайлов для восстановления
        height, width = frame.shape[:2]
        
        y_count = 0
        for y in range(0, height, tile_size):
            y_count += 1
            for x in range(0, width, tile_size):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                tile = frame[y:y_end, x:x_end]
                tiles.append(tile)
                tile_positions.append((y, x, y_end, x_end))
        
        return tiles, tile_positions, len(tiles), y_count
    
    def detect_objects(self, tiles, original_tiles):
        """Детекция объектов с сохранением только тайлов с детекцией"""
        if not tiles or not self.model:
            return tiles, [], []
        
        try:
            timestamp = datetime.now().strftime("%H_%M_%S")
            detections = []
            detected_tile_indices = []
            detected_tiles_with_box = []
            detected_tiles_without_box = []
            
            # Выполнение детекции
            results = self.model(
                tiles, 
                verbose=False, 
                conf=0.7, 
                iou=0.5, 
                imgsz=self.size_tail, 
                device=self.device
            )
            
            # Обработка результатов по тайлам
            for idx, (result, original_tile) in enumerate(zip(results, original_tiles)):
                tile_detections = []
                
                # Проверка наличия детекций в тайле
                if result.boxes is not None and len(result.boxes) > 0:
                    # Сбор информации о детекциях
                    for box in result.boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names.get(cls, f"class_{cls}")
                        tile_detections.append(class_name)
                        detections.append(class_name)
                    
                    # Если есть детекции, сохраняем тайл
                    if tile_detections:
                        
                        detected_tile_indices.append(idx)
                        # Сохраняем тайл с детекцией
                        self._save_detected_tile(
                            result.plot(), 
                            original_tile, 
                            tile_detections, 
                            timestamp, 
                            idx,
                        )
            
            return tiles, detections, detected_tile_indices
            
        except Exception as e:
            logger.error(f"Ошибка детекции: {e}")
            return tiles, [], []
    
    def _save_detected_tile(self, tile_with_box, tile_without_box, detections, timestamp, tile_index):
        """Сохранение тайла с детекцией"""
        try:
            metr = self.in_registers[3]/100
            # Создаем уникальные имена файлов для каждого тайла
            filename_box = f"det_box_tile_{tile_index}_{timestamp}.jpg"
            filename_no_box = f"det_tile_{tile_index}_{timestamp}.jpg"
            
            output_path_box = os.path.join(self.output_dir, filename_box)
            output_path_no_box = os.path.join(self.output_dir, filename_no_box)
            
            # Сохранение тайлов
            cv2.imwrite(output_path_box, tile_with_box)
            cv2.imwrite(output_path_no_box, tile_without_box)
            
            # Запись в лог
            text = ', '.join(detections)
            log_path = os.path.join(self.output_dir, 'detection_log.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {metr}  -  {text}\n")
                #f.write(f"{datetime.now().strftime('%H:%M:%S')} - Tile {tile_index}: {text}\n")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения тайла с детекцией: {e}")
    
    def update_statistics(self, detections):
        """Обновление статистики"""
        try:
            class_counts = defaultdict(int)
            
            # Подсчет детекций
            for class_name in detections:
                class_counts[class_name] += 1
            
            # Обновление OPC переменных
            if self.opc_server and self.variables:
                with self.lock:
                    for class_name, count in class_counts.items():
                        if class_name in self.variables:
                            try:
                                self.variables[class_name].set_value(float(count))
                            except Exception as e:
                                logger.warning(f"Ошибка обновления OPC переменной {class_name}: {e}")                
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")        
    
    def apply_background_filter(self, frame):
        """Применение фильтра фона"""
        if frame is None or self.background is None:
            return frame
            
        try:
            corrected = np.divide(
                frame.astype(np.float32), 
                self.background.astype(np.float32) + 1e-6
            )
            corrected = np.clip(corrected, 0, 2)
            corrected = (corrected * 255).astype(np.uint8)
            return corrected
        except Exception as e:
            logger.error(f"Ошибка фильтрации фона: {e}")
            return frame
    
    def calculate_fps(self):
        """Расчет FPS"""
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed_time = time.time() - self.start_time
            self.current_fps = 10 / elapsed_time if elapsed_time > 0 else 0
            self.start_time = time.time()
        return self.current_fps
    
    def add_text_overlay(self, frame, texts):
        """Добавление текстовой информации на кадр (без фона)"""
        if frame is None:
            return frame
            
        positions = [(10, 30 + i * 30) for i in range(len(texts))]
        colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Убедимся, что цветов хватает
        while len(colors) < len(texts):
            colors.append((255, 255, 255))
        
        for text, pos, color in zip(texts, positions, colors):
            cv2.putText(
                frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        
        return frame
    
    def create_visualization_grid(self, tiles, tile_count, y_count, tile_size=200):
        """Создание сетки визуализации из тайлов"""
        if not tiles:
            return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        x_count = max(1, (tile_count + y_count - 1) // y_count if y_count > 0 else tile_count)
        
        result_height = max(1, y_count * tile_size)
        result_width = max(1, x_count * tile_size)
        
        """Создаем цветное изображение для визуализации"""
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
        
        for idx, tile in enumerate(tiles):
            if tile is None:
                continue
                
            row = idx // x_count
            col = idx % x_count
            
            if row >= y_count or col >= x_count:
                continue
            
            try:
                # Преобразуем тайл в цветной формат если нужно
                if len(tile.shape) == 2:  # Серое изображение
                    display_tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                else:
                    display_tile = tile
                    
                resized_tile = cv2.resize(display_tile, (tile_size, tile_size))
                y_start = row * tile_size
                y_end = (row + 1) * tile_size
                x_start = col * tile_size
                x_end = (col + 1) * tile_size
                
                if y_end <= result_height and x_end <= result_width:
                    result[y_start:y_end, x_start:x_end] = resized_tile
                    
            except Exception as e:
                logger.warning(f"Ошибка обработки тайла {idx} для визуализации: {e}")
        
        return result
    
    def handle_user_input(self, key):
        """Обработка пользовательского ввода"""
        if key == ord('g'):
            self.gray = not self.gray
            self._initialize_model()
            logger.info(f"Режим {'серого' if self.gray else 'цветного'} изображения")
        elif key == ord('s') and self.background is None:
            return 'save_background'
        elif key ==ord('f'):
            return 'save_frame'
        elif key == ord('q'):
            return 'quit'
        return None
    
    def save_background(self, frame):
        """Сохранение фона"""
        if frame is not None:
            self.background = frame.copy()
            cv2.imwrite(self.BACKGROUND_PATH, self.background)
            logger.info(f"Фон сохранен как '{self.BACKGROUND_PATH}'")
    
    def update_modbus_data(self):
        """Обновление данных с Modbus"""
        try:
            if self.modbus_manager and self.modbus_manager.connected:
                self.in_registers = self.modbus_manager.get()
                self.modbus_manager.put(self.out_registers)
        except Exception as e:
            logger.error(f"Ошибка обновления Modbus данных: {e}")
    
    def save_frame(self, frame):
        """Сохранение кадра"""
        if frame is not None and self.get == 1:
            timestamp = datetime.now().strftime("%H_%M_%S")
            filename = f"save_{timestamp}.jpg"
            output_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(output_path, frame)
            self.get = 0
            logger.info("Кадр сохранен")
    
    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.camera:
                self.camera.close()
            if self.opc_server:
                self.opc_server.stop()
            if self.modbus_manager:
                self.modbus_manager.stop()
                self.modbus_manager.disconnect()
            cv2.destroyAllWindows()
            logger.info("Ресурсы освобождены")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")

def main():
    """Основная функция"""
    date_old = date.today()
    system = AI_Detection_System()
    
    if not system.initialize_system():
        logger.error("Не удалось инициализировать систему")
        return
    
    try:
        logger.info("Система запущена. Нажмите 'q' для выхода")

        
        while True:
            
            if date_old != date.today():
                date_old = date.today()
                system._create_dir()
            
            key = cv2.waitKey(1) & 0xFF
            
            # Обработка пользовательского ввода
            action = system.handle_user_input(key)
            if action == 'quit':
                break
            elif action == 'save_background':
                frame = system.capture_frame()
                system.save_background(frame)
            elif action == 'save_frame':
                system.get = 1
                
            if system.get == 1:
                frame = system.capture_frame()
                system.save_frame(frame)
                
            
            # Обновление данных Modbus
            system.update_modbus_data()
            

            
            # Захват кадра
            frame = system.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Сохранение кадра при необходимости
            system.save_frame(frame)
            
            # Применение фильтра фона
            if system.background is not None:
                frame = system.apply_background_filter(frame)
            
            # Разделение на тайлы
            tiles, tile_positions, tile_count, y_count = system.split_frame_to_tiles(frame, system.size_tail)

            yolo_start = time.time()
            # Детекция объектов
            if system.det and system.model:
                processed_tiles, detections, detected_indices = system.detect_objects(tiles, tiles)
                system.update_statistics(detections)
            yolo_time = time.time() - yolo_start
            
            # Создание визуализации всех тайлов для отображения
            if tiles:
                # Создаем сетку из всех тайлов для отображения
                result_image = system.create_visualization_grid(tiles, tile_count, y_count)
                
                # Добавление текстовой информации
                fps = system.calculate_fps()
                metr = system.in_registers[3]/100 
                speed = (system.in_registers[4]/10)/0.0314159
                texts = [
                    f"Time - {datetime.now().strftime('%H:%M:%S')}",
                    f"FPS: {fps:.2f}",
                    f"Meters - {metr:.2f}, Speed - {speed:.2f}",
                    #f"Speed - {speed:.2f}",
                    f"Tiles - {tile_count}, Yolo - {yolo_time:.3f}",
                    #f"Yolo - {yolo_time:.3f}",
                    "'q' - exit"
                ]
                #if system.background is None:
                #    texts.insert(-1, "'s' - save background")
                
                result_image = system.add_text_overlay(result_image, texts)
                
                # Отображение результата
                cv2.imshow('YOLO Detection', result_image)
            
            # Проверка закрытия окна
            if cv2.getWindowProperty('YOLO Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
                
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    except Exception as e:
        logger.error(f"Ошибка в основном цикле: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
