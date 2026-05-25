import RPi.GPIO as GPIO
import time

BUTTON_PIN = 2

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Aguardando botão no GPIO 2... (Ctrl+C para sair)")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("Botão pressionado!")
            time.sleep(0.2)  # debounce
        else:
            print("Não pressionado")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nEncerrando...")
finally:
    GPIO.cleanup()