import cv2
import numpy as np
import os
import sqlite3
import subprocess
import sounddevice as sd
import wave
import json
import time
import threading
import RPi.GPIO as GPIO
from vosk import Model, KaldiRecognizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIGURAÇÕES GLOBAIS
# =========================

# Diretório base do próprio script — garante que caminhos relativos
# funcionem independente de qual diretório chamou este arquivo
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELO_VOZ_SAIDA   = os.path.join(_BASE_DIR, "pt_BR-faber-medium.onnx")
PIPER_BIN          = os.path.join(_BASE_DIR, "piper", "piper")
MODELO_VOZ_ENTRADA = os.path.join(_BASE_DIR, "vosk-model-small-pt")
MODELO_FACE_CORAL  = os.path.join(_BASE_DIR, "mobilefacenet_edgetpu.tflite")
MODELO_FACE_CPU    = os.path.join(_BASE_DIR, "mobilefacenet.tflite")
DB_NAME            = os.path.join(_BASE_DIR, "faces_database.db")

# Áudio — nomes fixos de card (não mudam no reboot)
I2S_DEVICE = "hw:sndrpigooglevoi,0"   # INMP441 (entrada) + MAX98357A (saída I2S)
P2_DEVICE  = "plughw:Headphones,0"    # saída analógica P2

# Gravação — confirmado: arecord -D hw:sndrpigooglevoi,0 -f S32_LE -r 48000 -c 2
MIC_RATE     = 48000
MIC_CHANNELS = 2
MIC_DTYPE    = 'int32'   # S32_LE

DURACAO_PAD         = 5
SIMILARIDADE_MINIMA = 0.75

# GPIO
BOTAO_GPIO                = 5   # alterna saída de áudio
BOTAO_CONFIRMACAO_GPIO    = 6   # confirmação sim/não (botão físico paralelo à voz)
TIMEOUT_BOTAO_CONFIRMACAO = 10  # segundos aguardando o botão quando mic falhar


# =========================
# GERENCIADOR DE ÁUDIO
# =========================
class GerenciadorAudio:
    """
    Alterna a saída de áudio entre MAX98357A (I2S) e P2 analógico via botão GPIO.
    Usa nomes fixos de card para não depender de numeração do ALSA.
    """

    def __init__(self):
        self._saida_atual  = I2S_DEVICE
        self._lock         = threading.Lock()
        self._ultimo_state = GPIO.HIGH
        self._monitorando  = False
        self._configurar_gpio()
        self._iniciar_monitoramento()
        print(f"🔊 Saída padrão: MAX98357A [{I2S_DEVICE}]")
        print(f"🔘 GPIO {BOTAO_GPIO} — pressione para alternar entre MAX e P2")

    def _configurar_gpio(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BOTAO_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _monitorar(self):
        import time
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
        GPIO.cleanup()


# =========================
# BANCO DE DADOS
# =========================
class BancoDados:
    def __init__(self, db_name=DB_NAME):
        self.db_name = db_name
        self._init()

    def _init(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS usuarios (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome      TEXT    NOT NULL,
                    embedding BLOB    NOT NULL
                )
            ''')
            conn.commit()

    def salvar(self, nome, embedding):
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(
                "INSERT INTO usuarios (nome, embedding) VALUES (?, ?)",
                (nome, emb_bytes)
            )
            conn.commit()
        print(f"💾 '{nome}' salvo no banco.")

    def atualizar(self, nome_antigo, nome_novo, embedding):
        """Sobrescreve o embedding de um usuário existente."""
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(
                "UPDATE usuarios SET nome = ?, embedding = ? WHERE nome = ?",
                (nome_novo, emb_bytes, nome_antigo)
            )
            conn.commit()
        print(f"🔄 '{nome_antigo}' atualizado para '{nome_novo}'.")

    def carregar_todos(self):
        with sqlite3.connect(self.db_name) as conn:
            rows = conn.execute("SELECT nome, embedding FROM usuarios").fetchall()
        return [
            {"nome": r[0], "embedding": np.frombuffer(r[1], dtype=np.float32)}
            for r in rows
        ]


# =========================
# VOZ SAÍDA — PIPER TTS
# =========================
class VozSaida:
    """
    Fluxo:
      1. piper gera voz.wav  (22050 Hz mono 16-bit)
      2. sox converte + normaliza para:
           - MAX98357A: 48kHz S32_LE estéreo  → voz_saida.wav  (gain -3)
           - P2:        44100Hz S16_LE estéreo → voz_saida.wav
             CORREÇÃO: ordem correta sox: infile → outfile → efeitos
      3. aplay reproduz no dispositivo ativo

    CORREÇÃO 1: PIPER_BIN e MODELO_VOZ_SAIDA agora usam caminhos absolutos
                baseados em _BASE_DIR, funcionando independente do CWD.
    CORREÇÃO 2: comando sox P2 corrigido — 'norm' é um efeito e deve vir
                APÓS o arquivo de saída:
                  ERRADO:  sox voz.wav norm -1 -r 44100 ... voz_saida.wav
                  CORRETO: sox voz.wav voz_saida.wav norm (ou duas etapas)
                Usamos duas etapas para clareza e compatibilidade:
                  1) sox voz.wav voz_norm.wav norm
                  2) sox voz_norm.wav -r 44100 -e signed -b 16 -c 2 voz_saida.wav
    """

    def __init__(self, modelo=MODELO_VOZ_SAIDA, gerenciador_audio=None):
        self.modelo    = modelo
        self.ger_audio = gerenciador_audio
        self._ok       = False

        if not os.path.exists(PIPER_BIN):
            print(f"⚠️  Piper não encontrado em '{PIPER_BIN}' — TTS desativado.")
        elif not os.path.exists(modelo):
            print(f"⚠️  Modelo não encontrado: '{modelo}' — TTS desativado.")
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
                # CORREÇÃO: 'norm' é efeito — deve vir depois do outfile.
                # Etapa 1: normaliza o pico para 0 dBFS
                subprocess.run(
                    "sox voz.wav voz_norm.wav norm",
                    shell=True, check=True
                )
                # Etapa 2: converte para 44100 Hz S16_LE estéreo
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


# =========================
# VOZ ENTRADA — VOSK STT
# =========================
class VozEntrada:
    """
    Gravação: hw:sndrpigooglevoi,0 | S32_LE | 48kHz | estéreo (ou mono no fallback)
    Conversão para Vosk: gain 10 → 16kHz mono 16-bit

    CORREÇÃO: _resolver_device() agora retorna (índice, max_canais).
    gravar() usa min(MIC_CHANNELS, max_canais) para nunca pedir mais canais
    do que o dispositivo suporta — elimina o erro PortAudio
    "Invalid number of channels" sem precisar de try/except como fallback.
    """

    def __init__(self, modelo=MODELO_VOZ_ENTRADA, voz_saida=None):
        self.voz = voz_saida

        if not os.path.exists(modelo):
            alt = modelo + "-0.3"
            if os.path.exists(alt):
                modelo = alt
            else:
                raise FileNotFoundError(
                    f"Modelo Vosk não encontrado: {modelo}\n"
                    "Baixe em: https://alphacephei.com/vosk/models"
                )

        print(f"⏳ Carregando Vosk [{modelo}]...")
        self.model = Model(modelo)
        print("✅ Vosk carregado.")

    def _resolver_device(self):
        """
        Retorna (índice, max_canais_entrada) do sndrpigooglevoi.
        Fallback: card 3 com 2 canais (confirmado por aplay -l no hardware).
        """
        for i, d in enumerate(sd.query_devices()):
            if "sndrpigooglevoi" in d['name'] and d['max_input_channels'] > 0:
                return i, int(d['max_input_channels'])
        return 3, 2   # fallback: card 3, estéreo

    def gravar(self, duracao=DURACAO_PAD):
        print(f"🎤 Gravando {duracao}s...")
        device, max_ch = self._resolver_device()

        canais = min(MIC_CHANNELS, max_ch)

        audio = sd.rec(
            int(MIC_RATE * duracao),
            samplerate=MIC_RATE,
            channels=canais,
            dtype=MIC_DTYPE,
            device=device
        )
        sd.wait()
        print("✅ Gravação concluída.")

        with wave.open("entrada_raw.wav", 'wb') as wf:
            wf.setnchannels(canais)
            wf.setsampwidth(4)        # S32 = 4 bytes
            wf.setframerate(MIC_RATE)
            wf.writeframes(audio.tobytes())

        # remix - faz mixdown estéreo→mono; se já for mono é inofensivo
        subprocess.run(
            "sox entrada_raw.wav -r 16000 -e signed -b 16 -c 1 entrada.wav "
            "gain 10 remix - highpass 80 lowpass 3400",
            shell=True, check=True
        )
        return "entrada.wav"

    def transcrever(self, arquivo_wav):
        rec = KaldiRecognizer(self.model, 16000)
        resultado = ""
        with wave.open(arquivo_wav, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    resultado += json.loads(rec.Result()).get("text", "")
        resultado += json.loads(rec.FinalResult()).get("text", "")
        return resultado.strip()

    def ouvir(self, prompt, duracao=DURACAO_PAD, tentativas=2):
        """
        Fala o prompt UMA vez, depois grava.
        Se não entender, fala mensagem CURTA e grava de novo — sem repetir o prompt.
        """
        for i in range(tentativas):
            if i == 0 and self.voz:
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

    def gravar_sem_prompt(self, duracao=DURACAO_PAD):
        """Grava diretamente sem falar nada — usado após mensagem de erro."""
        return self.transcrever(self.gravar(duracao))


# =========================
# DETECTOR DE ROSTO — MEDIAPIPE
# =========================
class DetectorMediaPipe:
    """
    Detecta rostos com MediaPipe Face Detection.
    Fallback automático para HaarCascade se mediapipe não estiver instalado.
    """
    CONFIANCA_MIN = 0.7

    def __init__(self):
        self._detector   = None
        self._disponivel = False
        self._carregar()

    def _carregar(self):
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.CONFIANCA_MIN
            )
            self._disponivel = True
            print("✅ MediaPipe FaceDetection carregado.")
        except ImportError:
            print("⚠️  mediapipe não instalado — usando HaarCascade.")
        except Exception as e:
            print(f"⚠️  Erro MediaPipe ({e}) — usando HaarCascade.")

    @property
    def disponivel(self):
        return self._disponivel

    def detectar(self, frame_bgr):
        if not self._disponivel:
            return []
        h, w, _ = frame_bgr.shape
        res = self._detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return []
        faces = []
        for det in res.detections:
            bb = det.location_data.relative_bounding_box
            x  = max(0, int(bb.xmin * w))
            y  = max(0, int(bb.ymin * h))
            fw = min(int(bb.width  * w), w - x)
            fh = min(int(bb.height * h), h - y)
            faces.append((x, y, fw, fh))
        return faces

    def fechar(self):
        if self._detector:
            self._detector.close()


# =========================
# CÂMERA + FACE EMBEDDING
# =========================
class CameraFace:
    ENTRADA_W = 112
    ENTRADA_H = 112

    def __init__(self, voz_saida=None):
        self.voz      = voz_saida
        self.detector = DetectorMediaPipe()
        self.haar     = None
        self._coral   = False
        self.interp   = self._carregar_modelo()

        if not self.detector.disponivel:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("🔄 Fallback: HaarCascade ativo.")

    def _carregar_modelo(self):
        if os.path.exists(MODELO_FACE_CORAL):
            interp = self._tentar_coral(MODELO_FACE_CORAL)
            if interp:
                self._coral = True
                return interp
        if os.path.exists(MODELO_FACE_CPU):
            return self._tentar_cpu(MODELO_FACE_CPU)
        print("⚠️  Nenhum modelo TFLite — embedding simulado ativo.")
        return None

    def _tentar_coral(self, caminho):
        try:
            from pycoral.utils.edgetpu import make_interpreter
            interp = make_interpreter(caminho)
            interp.allocate_tensors()
            print(f"✅ MobileFaceNet → Coral EdgeTPU [{caminho}]")
            return interp
        except ImportError:
            print("⚠️  pycoral não instalado — tentando CPU.")
        except Exception as e:
            print(f"⚠️  Coral indisponível ({e}) — tentando CPU.")
        return None

    def _tentar_cpu(self, caminho):
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=caminho)
            interp.allocate_tensors()
            print(f"✅ MobileFaceNet → CPU [{caminho}]")
            return interp
        except ImportError:
            print("⚠️  tflite_runtime não instalado — pip install tflite-runtime")
        except Exception as e:
            print(f"⚠️  Erro CPU: {e}")
        return None

    def _pre_processar(self, face_bgr):
        rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res  = cv2.resize(rgb, (self.ENTRADA_W, self.ENTRADA_H))
        norm = (res.astype(np.float32) - 127.5) / 128.0
        return np.expand_dims(norm, axis=0)

    def _gerar_embedding(self, face_bgr):
        if self.interp is not None:
            try:
                inp = self.interp.get_input_details()[0]
                out = self.interp.get_output_details()[0]
                self.interp.set_tensor(inp["index"], self._pre_processar(face_bgr))
                self.interp.invoke()
                emb = self.interp.get_tensor(out["index"])[0]
                return (emb / (np.linalg.norm(emb) + 1e-10)).astype(np.float32)
            except Exception as e:
                print(f"⚠️  Erro inferência: {e}")
        # fallback simulado (sem modelo carregado)
        emb = np.random.rand(192).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-10)

    def _detectar_faces(self, frame):
        if self.detector.disponivel:
            faces = self.detector.detectar(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det  = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            faces = list(det) if len(det) > 0 else []
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def capturar_embedding(self, tentativas=5):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            if self.voz:
                self.voz.falar("Câmera não encontrada.")
            return None

        print("📸 Procurando rosto...")
        embedding = None

        for i in range(tentativas):
            ret, frame = cap.read()
            if not ret:
                continue
            faces = self._detectar_faces(frame)
            if not faces:
                det_nome = "MediaPipe" if self.detector.disponivel else "HaarCascade"
                print(f"  [{i+1}/{tentativas}] {det_nome}: nenhum rosto detectado...")
                continue
            x, y, w, h = faces[0]
            embedding   = self._gerar_embedding(frame[y:y+h, x:x+w])
            if embedding is not None:
                backend  = "Coral" if self._coral else "CPU"
                detector = "MediaPipe" if self.detector.disponivel else "HaarCascade"
                print(f"  ✅ Rosto capturado [{detector} + {backend}] — {len(embedding)}d")
                break

        cap.release()
        self.detector.fechar()
        if embedding is None and self.voz:
            self.voz.falar("Não foi possível detectar um rosto.")
        return embedding


# =========================
# INTERPRETADOR DE FALA
# =========================
class Interpretador:
    # "cada" incluído pois o Vosk pt-BR trunca "cadastrar" com frequência
    CADASTRO  = [
        "cadastr", "registr", "adicionar", "novo", "nova", "criar", "salvar",
        "cada", "cadas", "cadast"
    ]
    RECONHEC  = [
        "reconhec", "identific", "verificar", "entrar", "logar", "login", "quem",
        "recon", "reconhe"
    ]
    POSITIVOS = [
        "sim", "correto", "certo", "isso", "exato", "confirmo", "confirmar",
        "pode", "ok", "claro", "positivo", "tá", "ta"
    ]
    NEGATIVOS = [
        "não", "nao", "errado", "errei", "cancela", "cancelar", "negativo"
    ]
    PREFIXOS  = [
        "meu nome é", "meu nome e", "me chamo",
        "sou o", "sou a", "sou",
    ]

    @staticmethod
    def opcao(texto):
        for p in Interpretador.CADASTRO:
            if p in texto:
                return "c"
        for p in Interpretador.RECONHEC:
            if p in texto:
                return "r"
        return None

    @staticmethod
    def confirmacao(texto):
        for p in Interpretador.POSITIVOS:
            if p in texto:
                return "s"
        for p in Interpretador.NEGATIVOS:
            if p in texto:
                return "n"
        return None

    @staticmethod
    def nome(texto):
        limpo = texto
        for prep in sorted(Interpretador.PREFIXOS, key=len, reverse=True):
            limpo = limpo.replace(prep, "").strip()
        nome = " ".join(w.capitalize() for w in limpo.split() if w)
        return nome if nome else texto.title()


# =========================
# SISTEMA PRINCIPAL
# =========================
class SistemaReconhecimento:
    def __init__(self):
        self.ger_audio = GerenciadorAudio()
        self.db        = BancoDados()
        self.tts       = VozSaida(gerenciador_audio=self.ger_audio)
        self.stt       = VozEntrada(voz_saida=self.tts)
        self.camera    = CameraFace(voz_saida=self.tts)
        self.interp    = Interpretador()
        self._configurar_botao_confirmacao()

    def _configurar_botao_confirmacao(self):
        """
        Configura GPIO 6 como botão de confirmação sim/não.
        GPIO.setmode já foi definido por GerenciadorAudio (BCM).
        """
        GPIO.setup(BOTAO_CONFIRMACAO_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(
            f"🔘 GPIO {BOTAO_CONFIRMACAO_GPIO} — botão de confirmação sim/não configurado "
            f"(paralelo à voz, timeout {TIMEOUT_BOTAO_CONFIRMACAO}s)."
        )

    def _obter_opcao(self):
        self.tts.falar(
            "Diga 'cadastrar' para registrar um novo usuário, "
            "ou 'reconhecer' para identificar um usuário existente."
        )
        for tentativa in range(1, 4):
            texto = self.transcrever_agora(duracao=5)
            opcao = self.interp.opcao(texto)
            if opcao:
                acao = "cadastrar novo usuário" if opcao == "c" else "reconhecer usuário"
                self.tts.falar(f"Entendido. Vou {acao}.")
                return opcao
            if tentativa < 3:
                self.tts.falar(
                    f"Não entendi. Tentativa {tentativa} de 3. "
                    "Diga 'cadastrar' ou 'reconhecer'."
                )
        return None

    def transcrever_agora(self, duracao=DURACAO_PAD):
        """Grava e transcreve sem falar nada antes."""
        arquivo = self.stt.gravar(duracao)
        texto   = self.stt.transcrever(arquivo)
        if texto:
            print(f"🗣️  Você disse: \"{texto}\"")
        return texto.lower() if texto else ""

    def _poll_botao(self, resultado: list, parar: threading.Event):
        """
        Roda em thread separada. Faz polling no GPIO 6 até detectar
        pressão (LOW confirmado por 3 leituras consecutivas ~30ms) ou
        até parar.Event ser acionado.
        """
        estado_anterior = GPIO.HIGH
        contagem_low    = 0
        while not parar.is_set():
            estado_atual = GPIO.input(BOTAO_CONFIRMACAO_GPIO)
            if estado_atual == GPIO.LOW:
                contagem_low += 1
            else:
                contagem_low = 0
            if contagem_low == 3 and estado_anterior == GPIO.HIGH:
                print(f"\n✅ Botão GPIO {BOTAO_CONFIRMACAO_GPIO} pressionado — confirmado.")
                resultado.append("s")
                parar.set()
                return
            estado_anterior = estado_atual if contagem_low == 0 else estado_anterior
            time.sleep(0.01)

    def _confirmar(self, tentativas=3):
        """
        Ouve sim/não por voz (Vosk) e botão GPIO 6 em paralelo.
        Quem responder primeiro vence.
        Retorna 's' (sim), 'n' (não) ou None se não houver resposta.
        """
        resultado_botao: list = []
        parar_botao           = threading.Event()

        t_botao = threading.Thread(
            target=self._poll_botao,
            args=(resultado_botao, parar_botao),
            daemon=True
        )
        t_botao.start()
        print(f"🔘 GPIO {BOTAO_CONFIRMACAO_GPIO} ativo — pressione para confirmar.")

        resposta_voz = None

        for _ in range(tentativas):
            if parar_botao.is_set():
                break

            texto = self.transcrever_agora(duracao=4)

            if parar_botao.is_set():
                break

            conf = self.interp.confirmacao(texto)
            if conf:
                resposta_voz = conf
                break

            if not parar_botao.is_set():
                self.tts.falar("Diga sim ou não.")

        parar_botao.set()
        t_botao.join(timeout=0.5)

        if resultado_botao:
            return resultado_botao[0]

        return resposta_voz

    # ------------------------------------------------------------------
    # CADASTRO
    # ------------------------------------------------------------------
    def _fluxo_cadastro(self, embedding):
        """
        Antes de salvar, verifica se o rosto já existe no banco
        (sim >= SIMILARIDADE_MINIMA). Se existir, pergunta se quer
        sobrescrever — evitando registros duplicados.
        """
        # 1. Verificação de duplicata
        usuarios = self.db.carregar_todos()
        if usuarios:
            melhor_sim, melhor_usuario = max(
                ((cosine_similarity(
                    u["embedding"].reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0], u) for u in usuarios),
                key=lambda x: x[0]
            )
            if melhor_sim >= SIMILARIDADE_MINIMA:
                nome_existente = melhor_usuario["nome"]
                porcentagem    = int(melhor_sim * 100)
                self.tts.falar(
                    f"Este rosto já está cadastrado como {nome_existente}, "
                    f"com {porcentagem} porcento de similaridade. "
                    "Deseja sobrescrever? Diga sim ou não."
                )
                conf = self._confirmar()
                if conf != "s":
                    self.tts.falar("Cadastro cancelado.")
                    return

                # Sobrescreve: pede novo nome e atualiza
                self.tts.falar("Ok. Diga o novo nome para sobrescrever.")
                novo_nome = self._capturar_nome()
                if not novo_nome:
                    return
                self.db.atualizar(nome_existente, novo_nome, embedding)
                self.tts.falar(f"Cadastro de {nome_existente} atualizado para {novo_nome}.")
                return

        # 2. Rosto novo — fluxo normal
        self.tts.falar("Iniciando cadastro. Diga o nome completo do usuário.")
        nome = self._capturar_nome()
        if not nome:
            return
        self.db.salvar(nome, embedding)
        self.tts.falar(f"Usuário {nome} cadastrado com sucesso!")

    def _capturar_nome(self):
        """
        Tenta capturar e confirmar o nome em até 3 tentativas.
        Retorna o nome confirmado ou string vazia em caso de falha.
        """
        for tentativa in range(1, 4):
            if tentativa > 1:
                self.tts.falar("Não ouvi. Diga o nome novamente.")

            texto_nome = self.transcrever_agora(duracao=6)
            if not texto_nome:
                continue

            nome = self.interp.nome(texto_nome)
            self.tts.falar(f"Ouvi: {nome}. Correto? Diga sim ou não.")

            conf = self._confirmar()
            if conf == "s":
                return nome
            elif conf == "n":
                self.tts.falar("Ok. Diga o nome novamente.")
            else:
                self.tts.falar("Sem confirmação. Tentando novamente.")

        self.tts.falar("Não foi possível capturar o nome. Encerrando.")
        return ""

    # ------------------------------------------------------------------
    # RECONHECIMENTO
    # ------------------------------------------------------------------
    def _fluxo_reconhecimento(self, embedding):
        self.tts.falar("Iniciando reconhecimento. Aguarde.")
        usuarios = self.db.carregar_todos()

        if not usuarios:
            self.tts.falar("O banco de dados está vazio. Nenhum usuário cadastrado ainda.")
            return

        total = len(usuarios)
        self.tts.falar(
            f"Comparando com {total} usuário{'s' if total > 1 else ''} "
            f"cadastrado{'s' if total > 1 else ''}."
        )

        # Calcula similaridades de todos de uma vez
        embeddings_db = np.array([u["embedding"] for u in usuarios])
        sims = cosine_similarity(embedding.reshape(1, -1), embeddings_db)[0]

        melhor_idx  = int(np.argmax(sims))
        melhor_sim  = float(sims[melhor_idx])
        melhor_nome = usuarios[melhor_idx]["nome"]

        for i, u in enumerate(usuarios):
            print(f"  {u['nome']:20s} → {sims[i]:.4f}")

        porcentagem = int(melhor_sim * 100)
        print(f"\n  🏆 {melhor_nome} ({melhor_sim:.4f})  |  limiar: {SIMILARIDADE_MINIMA}")

        if melhor_sim >= SIMILARIDADE_MINIMA:
            self.tts.falar(
                f"Usuário reconhecido: {melhor_nome}. "
                f"Similaridade de {porcentagem} porcento."
            )
        else:
            self.tts.falar(
                f"Rosto não reconhecido. "
                f"Melhor resultado foi {melhor_nome} com {porcentagem} porcento, "
                f"abaixo do mínimo de {int(SIMILARIDADE_MINIMA * 100)} porcento."
            )

    # ------------------------------------------------------------------
    # EXECUÇÃO PRINCIPAL
    # ------------------------------------------------------------------
    def executar(self):
        self.tts.falar("Sistema de reconhecimento facial iniciado.")
        self.tts.falar("Por favor, posicione seu rosto em frente à câmera.")

        embedding = self.camera.capturar_embedding()
        if embedding is None:
            self.tts.falar("Nenhum rosto detectado. Encerrando.")
            return

        self.tts.falar("Rosto detectado com sucesso.")

        opcao = self._obter_opcao()
        if not opcao:
            self.tts.falar("Não foi possível entender a opção após 3 tentativas. Encerrando.")
            return

        if opcao == "c":
            self._fluxo_cadastro(embedding)
        elif opcao == "r":
            self._fluxo_reconhecimento(embedding)

        self.tts.falar("Processo finalizado. Obrigado.")

    def finalizar(self):
        self.ger_audio.cleanup()
        print("Sistema encerrado.")


# =========================
# EXECUÇÃO
# =========================
if __name__ == "__main__":
    sistema = SistemaReconhecimento()
    try:
        sistema.executar()
    finally:
        sistema.finalizar()