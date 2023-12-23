import time
import serial
import pymysql
from datetime import datetime
import speech_recognition as sr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump, load
from sqlalchemy import create_engine
import tkinter as tk
import threading
import cv2
import os
import time

lecturas_activas = True  # Variable de control para indicar si las lecturas están activas
Key_Pass = False
puerta_abierta = 0
puerta_cerrada = 0
correo_enviado = False
nombre='VACIO'

# Crear una nueva instancia del modelo
model = IsolationForest(contamination=0.1)

try:
    # Intentar cargar el modelo desde un archivo
    model = load('model.joblib')
except FileNotFoundError:
    # Si el archivo no existe, entrenar el modelo y guardarlo en un archivo
    engine = create_engine('mysql+pymysql://root:@localhost/proyecto_arduino')
    df = pd.read_sql('SELECT * FROM lecturas', con=engine)
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    
    # Agregar nuevas características temporales
    df['segundos_desde_la_medianoche'] = df['tiempo'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
    df['hora'] = df['tiempo'].apply(lambda x: x.hour)
    df['dia_semana'] = df['tiempo'].dt.dayofweek

    # Seleccionar las características relevantes para el modelo
    features = df[['segundos_desde_la_medianoche', 'hora', 'dia_semana']]

    # Entrenar el modelo con las nuevas características
    model.fit(features)
    dump(model, 'model.joblib')

def enviar_correo(mensaje, image_path, subject):
    global nombre
    msg = MIMEMultipart()
    password = "pnbt zmaq fjad antz"
    msg['From'] = "cloudentertaining@gmail.com"
    msg['To'] = "sonicx5100@gmail.com"
    msg['Subject'] = subject
    msg.attach(MIMEText(mensaje, 'plain'))

    # Adjuntar el archivo de audio solo si se proporciona
    #if audio_file is not None:
     #   with open(audio_file, 'rb') as f:
      #      part = MIMEBase('application', 'octet-stream')
       #     part.set_payload(f.read())
        #    encoders.encode_base64(part)
         #   part.add_header('Content-Disposition', 'attachment; filename="audio.wav"')
          #  msg.attach(part)

    # Adjuntar la imagen
    img_data = open(image_path, 'rb').read()
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image)

    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    server.login(msg['From'], password)
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    server.quit()
    print("Correo enviado exitosamente!")

#def reconocer_voz():
 #   global lecturas_activas  # Accede a la variable global
  #  global Key_Pass
   # recognizer = sr.Recognizer()
#
 #   while True:
  #      with sr.Microphone() as source:
   #         print("Dime 'quién es':")
    #        recognizer.adjust_for_ambient_noise(source)
     #       audio = recognizer.listen(source)
#
 #       try:
  #          texto = recognizer.recognize_google(audio, language="es-ES")
   #         print(f"Has dicho: {texto}")
    #        if "5102" in texto.lower():
     #           print("Hola, Bryan")
      #          Key_Pass = True #Reconoce la entrada
       #         break  # Sale del bucle si la respuesta es reconocida
        #    else:
         #       print("Respuesta no reconocida. Intenta nuevamente.")
#
 #       except sr.UnknownValueError:
  #          print("No se pudo entender el audio. Intenta nuevamente.")
   #     except sr.RequestError as e:
    #        print(f"Error en la solicitud a Google Speech Recognition service; {e}")
#
 #   lecturas_activas = True  # Restablece la variable a True cuando la función termina

    # Guarda el audio en un archivo
  #  with open("audio.wav", "wb") as f:
   #     f.write(audio.get_wav_data())

    #return "audio.wav"


def reconocimiento_facial():
    global nombre
    global lecturas_activas  # Accede a la variable global
    global Key_Pass
    dataPath = 'ReconocimientoFacial\Data' #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Leyendo el modelo
    face_recognizer.read('ReconocimientoFacial\modeloLBPHFace.xml')

    #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('ReconocimientoFacial\Videos\dotro.mp4')

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
  
    reconocido = False
    tiempo_inicio = time.time()  # Registra el tiempo de inicio

    while True:
        ret,frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            # LBPHFace
            if result[1] < 70:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                nombre = imagePaths[result[0]]
                Key_Pass = True #Reconoce la entrada
                reconocido = True

                # Restablecer el tiempo de inicio
                tiempo_inicio = time.time()

                 # Guardar la foto de la persona reconocida
                cv2.imwrite('persona_reconocida.jpg', frame[y:y + h, x:x + w])


            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            
        cv2.imshow('frame',frame)

        # Verifica si ha pasado un tiempo determinado (por ejemplo, 10 segundos) sin reconocer a nadie
        tiempo_actual = time.time()
        tiempo_transcurrido = tiempo_actual - tiempo_inicio
        if tiempo_transcurrido > 5:  # Cambia 2 por el número de segundos deseado
            print("No se reconoció a nadie durante un tiempo.")
            cv2.imwrite('persona_reconocida.jpg', frame[y:y + h, x:x + w])  # Ejecuta la lógica deseada
            tiempo_inicio = time.time()  # Restablecer el tiempo de inicio
            nombre='UNA PERSONA DESCONOCIDA'
            lecturas_activas = True  # Restablece la variable a True cuando la función termina
            Key_Pass = True
            break


        k = cv2.waitKey(1)
        if k == 27 or reconocido:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(nombre)
    lecturas_activas = True  # Restablece la variable a True cuando la función termina
    return ('persona_reconocida.jpg')



def codigo_principal(text_widget):
    global lecturas_activas
    global Key_Pass
    global puerta_abierta
    global puerta_cerrada
    global correo_enviado
    foto=''

    i = 1

    try:
        arduino = serial.Serial('COM4', 9600)
        time.sleep(4)
        myConexion = pymysql.connect(host='localhost', user='root', passwd='', db='proyecto_arduino')
        cur = myConexion.cursor()

        while True:
            if lecturas_activas:  # Verifica si las lecturas están activas
                mensaje = arduino.readline().decode('utf-8').strip()

                # Assuming the valid messages are either '0' or '1'
                if mensaje in ['0', '1']:
                    valor = int(mensaje)

                    # Obtener la marca de tiempo actual
                    tiempo_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Establecer la marca de tiempo a una fecha y hora específicas
                    #tiempo_actual = "2023-11-29 07:10:00"


                    # Para el sensor magnético
                    if valor == 1 and Key_Pass == False:
                        mensaje = f"Iteración {i}: La puerta está abierta, Tiempo: {tiempo_actual}"
                        print(mensaje)
                        text_widget.insert(tk.END, mensaje + "\n")  # Agrega el mensaje a la interfaz gráfica
                        puerta_cerrada += 1
                        i+=1
                        cur.execute('INSERT INTO lecturas (valor, tiempo) VALUES (%s, %s)', (valor, tiempo_actual))
                        myConexion.commit()
                        if Key_Pass == False:
                            lecturas_activas = False  # Desactiva las lecturas mientras se ejecuta la función
                            #audio_file = reconocer_voz()  # Llama a la función de reconocimiento de voz
                            foto = reconocimiento_facial()
                            subject = f"HA ENTRADO: {nombre}" 
                            enviar_correo(mensaje, foto, subject)  # Envía un correo electrónico con la alerta

                        # Define df en cada iteración
                        engine = create_engine('mysql+pymysql://root:@localhost/proyecto_arduino')
                        df = pd.read_sql('SELECT * FROM lecturas', con=engine)
                        df['tiempo'] = pd.to_datetime(df['tiempo'])
                        df['dia_semana'] = df['tiempo'].dt.dayofweek
                        df['hora'] = df['tiempo'].dt.hour
                        df['fecha'] = df['tiempo'].dt.date
                        df['tiempo'] = df['tiempo'].dt.time
                        df['segundos_desde_la_medianoche'] = df['tiempo'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
                        
                        # Seleccionar las características relevantes para el modelo
                        features = df[['segundos_desde_la_medianoche', 'hora', 'dia_semana']]

                        # Cada 50 iteraciones, entrena el modelo de detección de anomalías
                        if i % 50 == 0:
                            model.fit(features)
                            dump(model, 'model.joblib')
                            

                        # En cada iteración, predice si la última lectura es una anomalía
                        ultima_lectura = df.iloc[-1][['segundos_desde_la_medianoche', 'hora', 'dia_semana']].values.reshape(1, -1)
                        print(ultima_lectura)
                        prediccion = model.predict(ultima_lectura)
                        print(prediccion)
                        if prediccion == -1:
                            # Obtén la hora y la fecha de la última lectura
                            ultima_hora = df.iloc[-1]['tiempo']
                            ultima_fecha = df.iloc[-1]['fecha']
                            
                            mensaje = f"Se ha detectado una anomalía en tus horarios de entrada. La anomalía ocurrió el {ultima_fecha} a las {ultima_hora}."
                            subject ='ANOMALÍA'
                            enviar_correo(mensaje, foto, subject)  # Envía un correo electrónico con la alerta

                    elif valor == 0 and Key_Pass == True:
                        mensaje = f"Iteración {i}: La puerta está cerrada, Tiempo: {tiempo_actual}"
                        print(mensaje)
                        text_widget.insert(tk.END, mensaje + "\n")  # Agrega el mensaje a la interfaz gráfica
                        puerta_abierta += 1
                        i+=1
                        cur.execute('INSERT INTO lecturas (valor, tiempo) VALUES (%s, %s)', (valor, tiempo_actual))
                        myConexion.commit()
                        foto = reconocimiento_facial()
                        Key_Pass=False
                        subject = f"HA SALIDO: {nombre}" 
                        enviar_correo(mensaje, foto, subject)  # Envía un correo electrónico con la alerta

                        # Define df en cada iteración
                        engine = create_engine('mysql+pymysql://root:@localhost/proyecto_arduino')
                        df = pd.read_sql('SELECT * FROM lecturas', con=engine)
                        df['tiempo'] = pd.to_datetime(df['tiempo'])
                        df['dia_semana'] = df['tiempo'].dt.dayofweek
                        df['hora'] = df['tiempo'].dt.hour
                        df['fecha'] = df['tiempo'].dt.date
                        df['tiempo'] = df['tiempo'].dt.time
                        df['segundos_desde_la_medianoche'] = df['tiempo'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
                        
                        # Seleccionar las características relevantes para el modelo
                        features = df[['segundos_desde_la_medianoche', 'hora', 'dia_semana']]

                        # Cada 50 iteraciones, entrena el modelo de detección de anomalías
                        if i % 50 == 0:
                            model.fit(features)
                            dump(model, 'model.joblib')

                        # En cada iteración, predice si la última lectura es una anomalía
                        ultima_lectura = df.iloc[-1][['segundos_desde_la_medianoche', 'hora', 'dia_semana']].values.reshape(1, -1)
                        prediccion = model.predict(ultima_lectura)
                        print(prediccion)
                        if prediccion == -1:
                            # la fecha de la última lectura
                            ultima_hora = df.iloc[-1]['tiempo']
                            ultima_fecha = df.iloc[-1]['fecha']
                            
                            mensaje = f"Se ha detectado una anomalía en tus horarios de salida. La anomalía ocurrió el {ultima_fecha} a las {ultima_hora}."
                            enviar_correo(mensaje)  # Envía un correo electrónico con la alerta

    except serial.SerialException as e:
        print("Error de comunicación con el puerto serial:", str(e))
        text_widget.insert(tk.END, "Error de comunicación con el puerto serial: " + str(e) + "\n")  # Agrega el mensaje a la interfaz gráfica
    except KeyboardInterrupt:
        arduino.close()

def mostrar_mensajes():
    ventana = tk.Tk()
    ventana.title("Mensajes del Código Principal")
    
    # Crea un widget de texto y añade un scrollbar
    scrollbar = tk.Scrollbar(ventana)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text_widget = tk.Text(ventana, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_widget.pack()
    
    scrollbar.config(command=text_widget.yview)
    
    # Crea un hilo para el código principal
    hilo = threading.Thread(target=codigo_principal, args=(text_widget,))
    
    # Inicia el hilo
    hilo.start()
    
    ventana.mainloop()

mostrar_mensajes()
