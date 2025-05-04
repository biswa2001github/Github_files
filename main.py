import sensor, image, time, ml, math, uos, gc
import network, socket

### ----- Wi-Fi Setup -----
SSID = "vivo1951"
KEY = "12345678q"
HOST = ""
PORT = 8080

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, KEY)
while not wlan.isconnected():
    print('Trying to connect to "{}"...'.format(SSID))
    time.sleep_ms(1000)
print("WiFi Connected ", wlan.ifconfig())

### ----- Model Setup -----
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

net = None
labels = None
min_confidence = 0.75

try:
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load model: ' + str(e))

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load labels.txt: ' + str(e))

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]
threshold_list = [(math.ceil(min_confidence * 255), 255)]

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]
    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)
    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]
    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1)
        for b in blobs:
            x, y, w, h = b.rect()
            score = img.get_statistics(thresholds=threshold_list, roi=b.rect()).l_mean() / 255.0
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

### ----- MJPEG Streaming Setup -----
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
s.bind([HOST, PORT])
s.listen(5)
s.setblocking(True)

def start_streaming(s):
    print("Waiting for connections...")
    client, addr = s.accept()
    client.settimeout(5.0)
    print("Connected to " + addr[0] + ":" + str(addr[1]))

    client.sendall(
        "HTTP/1.1 200 OK\r\n"
        "Server: OpenMV\r\n"
        "Content-Type: multipart/x-mixed-replace;boundary=openmv\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n\r\n"
    )

    clock = time.clock()
    while True:
        clock.tick()
        img = sensor.snapshot()

        # ---- Run Inference & Draw ----
        for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
            if i == 0: continue  # background class
            if len(detection_list) == 0: continue
            for x, y, w, h, score in detection_list:
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                img.draw_circle((center_x, center_y, 20), color=colors[i % len(colors)])
                img.draw_string(x, y - 10, labels[i], color=colors[i % len(colors)], scale=4)
                print(labels[i])

        # ---- Encode and Stream ----
        cframe = img.to_jpeg(quality=35, copy=True)
        header = (
            "\r\n--openmv\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length:" + str(cframe.size()) + "\r\n\r\n"
        )
        client.sendall(header)
        client.sendall(cframe)
        # print(clock.fps())

while True:
    try:
        start_streaming(s)
    except OSError as e:
        print("socket error:", e)
