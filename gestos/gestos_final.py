import os
import time
import subprocess
import threading
import wave
import json
import cv2
import numpy as np
import mediapipe as mp
from tflite_runtime.interpreter import Interpreter

# Importações condicionais de hardware
try:
    import RPi.GPIO as GPIO
    _GPIO_DISPONIVEL = True
except ImportError:
    _GPIO_DISPONIVEL = False
    print("⚠️  RPi.GPIO não disponível — botão de alternância de áudio desativado.")

try:
    from vosk import Model, KaldiRecognizer
    _VOSK_DISPONIVEL = True
except ImportError:
    _VOSK_DISPONIVEL = False
    print("⚠️  Vosk não disponível — confirmação será feita via teclado.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

# Modelo de gestos
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "gestos.tflite")
LABELS       = ["fist", "stop", "thumb_index"]
THRESHOLD    = 0.70
MARGEM_BBOX  = 0.10
TEMPO_ESPERA = 3.0

# Reconhecimento facial (venv separado)
RECONHECIMENTO_FACIAL_PATH = "/home/tcc-unisanta/Documents/reconhecimento_facial/reconhecimento_facial.py"
VENV_PYTHON                = "/home/tcc-unisanta/Documents/reconhecimento_facial/venv39/bin/python"

# Áudio — nomes fixos de card (não mudam no reboot), idêntico ao reconhecimento facial
I2S_DEVICE = "hw:sndrpigooglevoi,0"   # INMP441 (entrada) + MAX98357A (saída I2S)
P2_DEVICE  = "plughw:Headphones,0"    # saída analógica P2

# Microfone — confirmado: arecord -D hw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2
MIC_RATE     = 48000
MIC_CHANNELS = 2
MIC_DTYPE    = 'int32'   # S32_LE
DURACAO_PAD  = 5         # segundos de gravação por resposta

# GPIO
BOTAO_GPIO            = 5   # alterna saída de áudio
BOTAO_CONFIRMACAO_GPIO = 6  # confirmação de gesto (fallback sem microfone)
TIMEOUT_BOTAO_CONFIRMACAO = 10  # segundos aguardando pressão do botão GPIO 6

# Piper TTS
MODELO_VOZ_SAIDA = "/home/tcc-unisanta/Documents/gestos/pt_BR-faber-medium.onnx"
PIPER_BIN        = "/home/tcc-unisanta/Documents/reconhecimento_facial/piper/piper"

# Vosk STT
MODELO_VOZ_ENTRADA = "/home/tcc-unisanta/Documents/gestos/vosk-model-small-pt-0.3"

# Palavras-chave para sim/não — idêntico ao reconhecimento facial
POSITIVOS = [
    "sim", "correto", "certo", "isso", "exato", "confirmo", "confirmar",
    "pode", "ok", "claro", "positivo", "tá", "ta"
]
NEGATIVOS = [
    "não", "nao", "errado", "errei", "cancela", "cancelar", "negativo"
]

# Nomes dos gestos em português para a fala de confirmação
FUNCIONALIDADES_GESTOS = {
    "fist":        "reconhecimento facial",
    "stop":        "reconhecimento de texto",
    "thumb_index": "leitura da bateria",
}


# ══════════════════════════════════════════════════════════════════════════════
# GERENCIADOR DE ÁUDIO  (idêntico ao reconhecimento facial)
# ══════════════════════════════════════════════════════════════════════════════

class GerenciadorAudio:
    """
    Alterna a saída entre MAX98357A (I2S) e P2 analógico via botão GPIO 5.
    Usa nomes fixos de card para não depender de numeração do ALSA.
    """

    def __init__(self):
        self._saida_atual = I2S_DEVICE
        self._lock        = threading.Lock()
        self._monitorando = False

        if _GPIO_DISPONIVEL:
            self._ultimo_state = GPIO.HIGH
            self._configurar_gpio()
            self._iniciar_monitoramento()
            print(f"🔊 Saída padrão: MAX98357A [{I2S_DEVICE}]")
            print(f"🔘 GPIO {BOTAO_GPIO} — pressione para alternar entre MAX e P2")
        else:
            print(f"🔊 Saída padrão: MAX98357A [{I2S_DEVICE}] (botão GPIO indisponível)")

    def _configurar_gpio(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BOTAO_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _monitorar(self):
        while self._monitorando:
            state = GPIO.input(BOTAO_GPIO)
            if state == GPIO.LOW and self._ultimo_state == GPIO.HIGH:
                self._alternar()
                time.sleep(0.3)   # debounce
            self._ultimo_state = state
            time.sleep(0.01)

    def _iniciar_monitoramento(self):
        self._monitorando = True
        self._thread = threading.Thread(target=self._monitorar, daemon=True)
        self._thread.start()

    def _alternar(self):
        with self._lock:
            if self._saida_atual == I2S_DEVICE:
                self._saida_atual = P2_DEVICE
                modo = "P2 analógico"
            else:
                self._saida_atual = I2S_DEVICE
                modo = "MAX98357A (I2S)"
            print(f"\n🔘 Saída → {modo} [{self._saida_atual}]")

    @property
    def saida(self):
        with self._lock:
            return self._saida_atual

    def cleanup(self):
        self._monitorando = False
        if _GPIO_DISPONIVEL:
            GPIO.cleanup()


# ══════════════════════════════════════════════════════════════════════════════
# VOZ SAÍDA — PIPER TTS  (idêntico ao reconhecimento facial)
# ══════════════════════════════════════════════════════════════════════════════

class VozSaida:
    """
    Fluxo:
      1. piper gera voz.wav (22050 Hz mono 16-bit)
      2. sox converte + normaliza:
           - MAX98357A: 48kHz S32_LE estéreo → gain -3
           - P2:        44100Hz S16_LE estéreo → norm -1 (evita clipping)
      3. aplay reproduz no dispositivo ativo
    """

    def __init__(self, modelo=MODELO_VOZ_SAIDA, gerenciador_audio=None):
        self.modelo    = modelo
        self.ger_audio = gerenciador_audio
        self._ok       = False

        if not os.path.exists(PIPER_BIN):
            print(f"⚠️  Piper não encontrado em '{PIPER_BIN}' — TTS desativado.")
        elif not os.path.exists(modelo):
            print(f"⚠️  Modelo TTS não encontrado: '{modelo}' — TTS desativado.")
        else:
            self._ok = True
            print(f"✅ Piper TTS pronto [{modelo}]")

    def falar(self, texto: str):
        print(f"🔊 {texto}")
        if not self._ok:
            return
        try:
            saida = self.ger_audio.saida if self.ger_audio else I2S_DEVICE

            # 1. Gera voz.wav com Piper (22050 Hz mono 16-bit)
            subprocess.run(
                f'echo "{texto}" | {PIPER_BIN} --model {self.modelo} --output_file voz.wav',
                shell=True, check=True, stderr=subprocess.DEVNULL
            )

            # 2. Converte conforme dispositivo ativo
            if saida == I2S_DEVICE:
                # MAX98357A: 48kHz S32_LE estéreo — gain -3 evita clipping
                subprocess.run(
                    "sox voz.wav -r 48000 -e signed -b 32 -c 2 voz_saida.wav gain -3",
                    shell=True, check=True
                )
            else:
                # P2: normaliza primeiro (norm sem argumento = 0 dBFS),
                # depois converte para 44100 Hz S16_LE estéreo
                subprocess.run(
                    "sox voz.wav voz_norm.wav norm",
                    shell=True, check=True
                )
                subprocess.run(
                    "sox voz_norm.wav -r 44100 -e signed -b 16 -c 2 voz_saida.wav",
                    shell=True, check=True
                )

            # 3. Reproduz
            subprocess.run(
                f"aplay -D {saida} voz_saida.wav",
                shell=True, check=True
            )

        except subprocess.CalledProcessError as e:
            print(f"⚠️  Erro TTS: {e}")
        except Exception as e:
            print(f"⚠️  Erro inesperado no TTS: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# VOZ ENTRADA — VOSK STT  (idêntico ao reconhecimento facial)
# ══════════════════════════════════════════════════════════════════════════════

class VozEntrada:
    """
    Gravação via arecord — idêntico ao reconhecimento facial:
      arecord -D hw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2 -d N arquivo.wav
    Conversão para Vosk: gain 10 → 16kHz mono 16-bit via sox.
    """

    def __init__(self, modelo=MODELO_VOZ_ENTRADA, voz_saida=None):
        self.voz = voz_saida
        self._ok = False

        if not _VOSK_DISPONIVEL:
            print("⚠️  Vosk indisponível — STT desativado.")
            return

        if not os.path.exists(modelo):
            alt = modelo + "-0.3"
            if os.path.exists(alt):
                modelo = alt
            else:
                print(
                    f"⚠️  Modelo Vosk não encontrado: {modelo} — STT desativado.\n"
                    "    Baixe em: https://alphacephei.com/vosk/models"
                )
                return

        print(f"⏳ Carregando Vosk [{modelo}]...")
        self.model = Model(modelo)
        self._ok   = True
        print("✅ Vosk carregado.")

    @property
    def disponivel(self) -> bool:
        return self._ok

    def gravar(self, duracao=DURACAO_PAD) -> str:
        """Grava via arecord — idêntico ao reconhecimento facial."""
        print(f"🎤 Gravando {duracao}s...")
        subprocess.run(
            f"arecord -D {I2S_DEVICE} -f S32_LE -r {MIC_RATE} -c {MIC_CHANNELS} "
            f"-d {duracao} entrada_raw.wav",
            shell=True, check=True
        )
        print("✅ Gravação concluída.")

        # Converte: mixdown estéreo→mono, gain 20 (mais alto para captar melhor),
        # filtros de voz, 16kHz 16-bit para Vosk
        subprocess.run(
            "sox entrada_raw.wav -r 16000 -e signed -b 16 -c 1 entrada.wav "
            "gain 20 remix - highpass 80 lowpass 3400",
            shell=True, check=True
        )
        return "entrada.wav"

    def transcrever(self, arquivo_wav: str) -> str:
        rec       = KaldiRecognizer(self.model, 16000)
        resultado = ""
        with wave.open(arquivo_wav, "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    resultado += json.loads(rec.Result()).get("text", "")
        resultado += json.loads(rec.FinalResult()).get("text", "")
        return resultado.strip()

    def ouvir(self, prompt="", duracao=DURACAO_PAD, tentativas=2) -> str:
        """
        Fala o prompt UMA vez (se fornecido), depois grava.
        Se não entender, fala mensagem curta e grava de novo — sem repetir o prompt.
        Idêntico ao reconhecimento facial.
        """
        for i in range(tentativas):
            if i == 0 and prompt and self.voz:
                self.voz.falar(prompt)
            texto = self.transcrever(self.gravar(duracao))
            if texto:
                print(f"🗣️  Você disse: \"{texto}\"")
                return texto.lower()
            if i < tentativas - 1 and self.voz:
                self.voz.falar("Não entendi. Pode repetir?")
        if self.voz:
            self.voz.falar("Não consegui entender.")
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE IMAGEM
# ══════════════════════════════════════════════════════════════════════════════

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def corrigir(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_CLAHE.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def bbox_quadrado(landmarks, w_frame: int, h_frame: int):
    xs = [lm.x * w_frame for lm in landmarks]
    ys = [lm.y * h_frame for lm in landmarks]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    lado   = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    metade = lado // 2 + int(lado * MARGEM_BBOX)
    x1, y1 = max(0, cx - metade), max(0, cy - metade)
    x2, y2 = min(w_frame, cx + metade), min(h_frame, cy + metade)
    return x1, y1, x2, y2


# ══════════════════════════════════════════════════════════════════════════════
# AÇÕES DOS GESTOS
# ══════════════════════════════════════════════════════════════════════════════

def acao_fist(voz: VozSaida):
    """Ativa o sistema de reconhecimento facial no venv39."""
    voz.falar("Gesto reconhecido. Reconhecimento facial ativado.")
    print(f"\n🚀 Iniciando: {RECONHECIMENTO_FACIAL_PATH}")

    if not os.path.exists(VENV_PYTHON):
        voz.falar("Erro. Ambiente Python do reconhecimento facial não encontrado.")
        print(f"⚠️  Python não encontrado: {VENV_PYTHON}")
        return

    if not os.path.exists(RECONHECIMENTO_FACIAL_PATH):
        voz.falar("Erro. Script de reconhecimento facial não encontrado.")
        print(f"⚠️  Script não encontrado: {RECONHECIMENTO_FACIAL_PATH}")
        return

    try:
        subprocess.run([VENV_PYTHON, RECONHECIMENTO_FACIAL_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Erro ao executar reconhecimento facial: {e}")
        voz.falar("Ocorreu um erro ao executar o reconhecimento facial.")


def acao_stop(voz: VozSaida):
    """Placeholder — OCR será implementado por outro aluno."""
    voz.falar("Gesto reconhecido. Reconhecimento de texto ativado.")
    print("📄 OCR: funcionalidade a ser implementada.")


def acao_thumb_index(voz: VozSaida):
    """Placeholder — leitura de bateria será implementada por outro aluno."""
    voz.falar("Gesto reconhecido. Leitura da carga da bateria.")
    print("🔋 Bateria: funcionalidade a ser implementada.")


# Tabela de despacho
ACOES = {
    "fist":        acao_fist,
    "stop":        acao_stop,
    "thumb_index": acao_thumb_index,
}


# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SistemaGestoSnapshot:

    def __init__(self):
        self.ger_audio = GerenciadorAudio()
        self.voz       = VozSaida(gerenciador_audio=self.ger_audio)
        self.stt       = VozEntrada(voz_saida=self.voz)
        self._carregar_mediapipe()
        self._carregar_modelo(MODEL_PATH)
        self._configurar_botao_confirmacao()   # ← GPIO 6

    # ── Inicialização ──────────────────────────────────────────────────────────

    def _carregar_mediapipe(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def _configurar_botao_confirmacao(self):
        """
        Configura GPIO 6 como botão de confirmação de gesto.
        Usado como fallback quando o microfone (Vosk) não está disponível.
        O GPIO.setmode já foi definido por GerenciadorAudio (GPIO.BCM).
        """
        if _GPIO_DISPONIVEL:
            GPIO.setup(BOTAO_CONFIRMACAO_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(
                f"🔘 GPIO {BOTAO_CONFIRMACAO_GPIO} — botão de confirmação de gesto configurado "
                f"(fallback sem microfone, timeout {TIMEOUT_BOTAO_CONFIRMACAO}s)."
            )
        else:
            print(
                f"⚠️  GPIO não disponível — botão de confirmação GPIO "
                f"{BOTAO_CONFIRMACAO_GPIO} desativado."
            )

    def _carregar_modelo(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        _, self.h_model, self.w_model, _ = self.inp[0]["shape"]

    # ── Câmera ────────────────────────────────────────────────────────────────

    def _buscar_camera(self):
        """Tenta índices 0-9 para encontrar câmera ativa."""
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        return None

    def _capturar_frame(self, cap) -> np.ndarray | None:
        """Countdown e captura de frame com buffer limpo."""
        for i in range(int(TEMPO_ESPERA), 0, -1):
            print(f"⏱️  Tirando foto em {i}s...", end="\r")
            time.sleep(1)
        for _ in range(5):   # descarta frames antigos do buffer
            cap.read()
        ret, frame = cap.read()
        return frame if ret else None

    # ── Inferência ────────────────────────────────────────────────────────────

    def _inferir(self, frame: np.ndarray):
        """
        Detecta mão com MediaPipe e classifica o gesto com TFLite.
        Retorna (label, confiança) ou (None, 0.0) se não houver mão.
        """
        h_f, w_f  = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, 0.0

        lm = results.multi_hand_landmarks[0]
        x1, y1, x2, y2 = bbox_quadrado(lm.landmark, w_f, h_f)
        recorte = frame[y1:y2, x1:x2]

        if recorte.size == 0:
            return None, 0.0

        recorte   = cv2.rotate(recorte, cv2.ROTATE_90_CLOCKWISE)
        img_res   = cv2.resize(recorte, (self.w_model, self.h_model))
        img_rgb   = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(np.float32(img_rgb), axis=0)

        self.interp.set_tensor(self.inp[0]['index'], img_input)
        self.interp.invoke()

        probs_raw = self.interp.get_tensor(self.out[0]['index'])[0].astype(np.float32)
        probs     = np.exp(probs_raw - probs_raw.max())
        probs    /= probs.sum()

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        return LABELS[idx], conf

    # ── Confirmação por botão GPIO 6 ─────────────────────────────────────────

    def _poll_botao(self, resultado: list, parar: threading.Event):
        """
        Roda em thread separada. Faz polling no GPIO 6 até detectar
        pressão (retorna 's') ou até parar.Event ser acionado.
        Debounce simples: exige que o pino fique LOW por 3 leituras consecutivas.
        """
        if not _GPIO_DISPONIVEL:
            return

        estado_anterior = GPIO.HIGH
        contagem_low    = 0

        while not parar.is_set():
            estado_atual = GPIO.input(BOTAO_CONFIRMACAO_GPIO)

            if estado_atual == GPIO.LOW:
                contagem_low += 1
            else:
                contagem_low = 0

            # Borda HIGH→LOW confirmada após 3 leituras (~30 ms)
            if contagem_low == 3 and estado_anterior == GPIO.HIGH:
                print(f"\n✅ Botão GPIO {BOTAO_CONFIRMACAO_GPIO} pressionado — confirmado.")
                resultado.append("s")
                parar.set()
                return

            estado_anterior = estado_atual if contagem_low == 0 else estado_anterior
            time.sleep(0.01)

    # ── Confirmação por voz E botão GPIO 6 em paralelo ───────────────────────

    def _confirmar_por_voz(self, tentativas: int = 3) -> str:
        """
        Voz (Vosk) e botão GPIO 6 correm em paralelo — quem responder
        primeiro vence. O botão sempre está ativo, independente do microfone.

        Retorna 's' (confirmado), 'n' (negado por voz) ou '' (sem resposta).
        """
        resultado_botao: list = []
        parar_botao           = threading.Event()

        # Inicia polling do botão em background
        t_botao = threading.Thread(
            target=self._poll_botao,
            args=(resultado_botao, parar_botao),
            daemon=True
        )
        t_botao.start()
        print(f"🔘 GPIO {BOTAO_CONFIRMACAO_GPIO} ativo — pressione para confirmar.")

        resposta_voz = ""

        if self.stt.disponivel:
            for i in range(tentativas):
                # Verifica se o botão já foi pressionado antes de gravar
                if parar_botao.is_set():
                    break

                arquivo = self.stt.gravar(duracao=4)

                # Verifica novamente após a gravação (botão pode ter sido
                # pressionado durante os 4 s de gravação)
                if parar_botao.is_set():
                    break

                texto = self.stt.transcrever(arquivo)
                if texto:
                    print(f"🗣️  Você disse: \"{texto.lower()}\"")
                    for p in POSITIVOS:
                        if p in texto.lower():
                            resposta_voz = "s"
                            break
                    if not resposta_voz:
                        for p in NEGATIVOS:
                            if p in texto.lower():
                                resposta_voz = "n"
                                break
                    if resposta_voz:
                        break

                if i < tentativas - 1 and not parar_botao.is_set():
                    self.voz.falar("Não entendi. Diga sim ou não.")
        else:
            # Sem microfone: aguarda apenas o botão pelo tempo máximo
            print("⚠️  Microfone indisponível — aguardando apenas o botão.")
            parar_botao.wait(timeout=TIMEOUT_BOTAO_CONFIRMACAO)
            if not parar_botao.is_set():
                print(f"⏱️  Tempo esgotado sem pressionar GPIO {BOTAO_CONFIRMACAO_GPIO}.")
                self.voz.falar("Tempo esgotado. Encerrando.")

        # Encerra o polling do botão
        parar_botao.set()
        t_botao.join(timeout=0.5)

        # Botão tem prioridade se foi pressionado
        if resultado_botao:
            return resultado_botao[0]

        return resposta_voz

    # ── Fluxo principal ───────────────────────────────────────────────────────

    def capturar_e_processar(self, tentativas_maximas: int = 2):
        """
        Ciclo completo:
          saudação → câmera → countdown → captura → corrigir →
          inferência → fala resultado → confirmação por voz (ou botão GPIO 6) → executa ação
        Dá até `tentativas_maximas` chances se não detectar mão.
        """
        self.voz.falar("Reconhecimento de gestos ativado.")
        print("\n" + "═"*50)
        print("  SISTEMA DE RECONHECIMENTO DE GESTOS")
        print("═"*50)

        for tentativa in range(1, tentativas_maximas + 1):

            # ── Abre câmera ───────────────────────────────────────────────────
            print(f"\n🔍 Procurando câmera... (tentativa {tentativa}/{tentativas_maximas})")
            cap = self._buscar_camera()

            if cap is None:
                self.voz.falar(
                    "Erro. Nenhuma câmera encontrada. "
                    "Verifique se a câmera está conectada."
                )
                return

            # ── Countdown e captura ───────────────────────────────────────────
            print("✅ Câmera encontrada! Prepare o gesto.")
            self.voz.falar("Câmera encontrada. Prepare o gesto.")

            frame = self._capturar_frame(cap)
            cap.release()

            if frame is None:
                self.voz.falar("Falha ao capturar imagem. Encerrando.")
                return

            print("\n⚡ Foto capturada! Processando...")
            frame       = corrigir(frame)
            label, conf = self._inferir(frame)
            porcentagem = int(conf * 100)

            # ── Mão não detectada ─────────────────────────────────────────────
            if label is None:
                print("⚠️  Nenhuma mão detectada.")
                if tentativa < tentativas_maximas:
                    self.voz.falar(
                        "Nenhuma mão foi detectada na imagem. "
                        "Por favor, posicione sua mão na frente da câmera e tente novamente."
                    )
                    continue   # dá mais uma chance
                else:
                    self.voz.falar(
                        "Não foi possível detectar uma mão. "
                        "Encerrando o reconhecimento de gestos."
                    )
                    return

            # ── Gesto detectado ───────────────────────────────────────────────
            print("\n" + "═"*50)
            print(f"  Gesto: {label}  |  Confiança: {porcentagem}%")
            print("═"*50)

            nome_gesto = FUNCIONALIDADES_GESTOS.get(label, label)
            confiavel  = conf >= THRESHOLD

            # Fala o resultado e pede confirmação
            if confiavel:
                self.voz.falar(
                    f"Gesto reconhecido. Deseja ativar {nome_gesto}? "
                    "Diga sim para confirmar ou não para cancelar."
                )
            else:
                self.voz.falar(
                    f"Detectei o gesto para {nome_gesto}, "
                    f"porém com baixa confiança de {porcentagem} porcento. "
                    "Deseja confirmar? Diga sim ou não."
                )

            # ── Confirmação (voz ou botão GPIO 6 como fallback) ───────────────
            resposta = self._confirmar_por_voz()

            if resposta == "s":
                print("✅ Confirmado.")
                ACOES[label](self.voz)
            elif resposta == "n":
                print("❌ Cancelado pelo usuário.")
                self.voz.falar("Gesto cancelado. Até logo.")
            else:
                print("❓ Sem confirmação.")
                self.voz.falar("Não foi possível entender a confirmação. Encerrando.")

            return   # encerra após uma detecção (confirmada ou não)

    def finalizar(self):
        self.ger_audio.cleanup()
        print("👋 Sistema encerrado.")


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sistema = SistemaGestoSnapshot()
    try:
        sistema.capturar_e_processar()
    except KeyboardInterrupt:
        print("\n🛑 Cancelado pelo usuário.")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro: {e}")
    finally:
        sistema.finalizar()