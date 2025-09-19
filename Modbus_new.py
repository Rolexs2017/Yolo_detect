from pyModbusTCP.client import ModbusClient
import threading
import time

class ModbusManager:
    """Менеджер Modbus TCP соединения"""
    def __init__(self, host="127.0.0.1", port=502):
        self.host = host
        self.port = port
        self.lock = threading.Lock()  # Создаем блокировку
        self.client = None
        self.connected = False
        self.running = False
        self.in_registers = []
        self.out_registers = []
        
    def connect(self):
            
        try:
            with self.lock:
                self.client = ModbusClient(host=self.host, port=self.port, auto_open=True, timeout=3)
                self.client.open()
                self.connected = bool(self.client.is_open)
                return self.connected
        except Exception as e:
            print(f"Ошибка подключения к Modbus: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Отключение от Modbus сервера"""
        if self.client and self.connected:
            with self.lock:
                self.client.close()
                self.connected = False
                
    def start(self):
        with self.lock:
            self.running = True
            self.thread = threading.Thread(target=self.job_loop)
            self.thread.daemon = True
            self.thread.start()
            time.sleep(1)
            
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def job_loop(self):
        while self.running:
            self.get_register()
            self.put_register()
            time.sleep(0.001)
            
    def get(self):
        return self.in_registers
        
    def put(self, registers):
        self.out_registers = registers
        
    
    def put_register(self):
        """Отправка данных детекции по Modbus"""
        if not self.connected or not self.client:
            return False
            
        try:
            for register in self.out_registers:
                try:
                    self.client.write_single_register(register, self.out_registers[register])
                except:
                    print(f"Ошибка записи регистра {register}")               
            return True
        except Exception as e:
            print(f"Общая ошибка Modbus: - {e}")
            return False
            
    def get_register(self):
        if not self.connected or not self.client:
            return False
        try:
            col = 100
            dat = self.client.read_input_registers(0, col)
            self.in_registers = dat
            return True
        except Exception as e:
            print("Ошибка чтения регистров")
            return False
            
  
